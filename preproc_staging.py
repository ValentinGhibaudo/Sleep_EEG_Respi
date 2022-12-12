import numpy as np
import pandas as pd 
import xarray as xr
import yasa
import glob
import mne
from datetime import datetime
import matplotlib.pyplot as plt
from params import subjects, dérivations, chan_sleep_staging

def init_da(coords):
    dims = list(coords.keys())
    coords = coords

    def size_of(element):
        element = np.array(element)
        size = element.size
        return size

    shape = tuple([size_of(element) for element in list(coords.values())])
    data = np.zeros(shape)
    da = xr.DataArray(data=data, dims=dims, coords=coords)
    return da

def eeg_mono_to_bipol(da, dérivations):
    da_bipol = []
    for bipol in dérivations : 
        pole1, pole2 = bipol.split('-')[0] , bipol.split('-')[1]
        if pole1 in ['EOGDt','EOGG']:
            chan1 = pole1
        
            chan2 = f'EEG {pole2}'
        else:
            chan1 = f'EEG {pole1}'
            chan2 = f'EEG {pole2}'
        sig1 = da.loc[chan1,:]
        sig2 = da.loc[chan2,:]

        bipol_sig = sig1 - sig2
        da_bipol.append(bipol_sig)
    da_bipolaire = xr.concat(da_bipol, dim = 'chan')
    da_bipolaire = da_bipolaire.assign_coords({'chan':dérivations})
    return da_bipolaire

def eeg_mono_reref_masto(da):
    mastos = ['EEG A1','EEG A2'] # mastoid raw chan names
    eeg_chans = [chan for chan in da.coords['chan'].values if 'EEG' in chan] # only eeg chans
    eeg_chans_without_masto = [chan for chan in eeg_chans if not chan in mastos] # only eeg chans without matoïds
    eeg_chans_without_masto_clean = [chan.split(' ')[1] for chan in eeg_chans_without_masto] # cleaned labels

    reref = da.sel(chan = mastos).mean('chan') # mean of the two signals
    da_reref = init_da({'chan':eeg_chans_without_masto_clean, 'time':da.coords['time'].values})
    for channel, channel_clean in zip(eeg_chans_without_masto,eeg_chans_without_masto_clean): # loop on eeg chans but not mastoïd
        sig = da.sel(chan = channel)
        sig_reref = sig - reref
        da_reref.loc[channel_clean, :] = sig_reref
    return da_reref

def da_to_mne_object(da, srate):
    ch_names = list(da.coords['chan'].values)
    sfreq = srate
    info = mne.create_info(ch_names, sfreq, ch_types='misc', verbose=False)
    raw = mne.io.RawArray(data = da.values, info=info, first_samp=0, copy='auto', verbose=False)
    return raw

concat_sleep_stats = [] # init list that will contain sleep statistics df from each subject
concat_metadata = [] # list that will contain metadata df from each subject

for subject in subjects:
    print(subject)
    
    input_file = glob.glob(f'../data/{subject}/*.edf')[0]
    raw = mne.io.read_raw_edf(glob.glob(f'../data/{subject}/*.edf')[0], verbose = False, preload = True) # read in Volts
    info = raw.info # info object
    
    # get participant metadata
    name = raw.__dict__['_raw_extras'][0]['subject_info']['name']
    gender = raw.__dict__['_raw_extras'][0]['subject_info']['sex']
    birthday = raw.__dict__['_raw_extras'][0]['subject_info']['birthday']
    age = int((datetime.now() - birthday).days / 365)
    
    # get recording metadata
    srate = info['sfreq'] # get srate
    hp = info['highpass'] # get highpass
    lp = info['lowpass'] # get lowpass
    chans = info['ch_names'] # get chans
    eeg_chans = [ chan for chan in chans if 'EEG' in chan] # get eeg chans
    eeg_chans_clean = [ chan.split(' ')[1] for chan in chans if 'EEG' in chan] # get eeg chan names clean = without "eeg etc.."
    eog_chans = [ chan for chan in chans if 'EOG' in chan]  # get eog chan names
    physio_chans = ['ECG','Menton','EOGDt','EOGG','DEBIT','THERM'] # physio chans
    time = raw.times # get time vector
    duration = round(time[-1], 2) # get last time point in secs
    
    data = [subject, name, gender, age, srate , hp , lp  , duration , round((duration / 60),2) , round((duration / 3600) , 2)]
    index = ['subject','name','gender','age','srate','HP','LP','sec duration','min duration','hour duration']
    metadata = pd.Series(data, index=index).to_frame().T # metadata

    # NOTCH 
    raw_notch = raw.copy()
    raw_notch.notch_filter(freqs = [50,100], verbose = False) # unit = V 
    data = raw_notch.get_data(units = 'uV') # extract data in unit = uV
    
    # BIPOLARIZATION
    da = xr.DataArray(data = data, dims = ['chan','time'] , coords = {'chan':chans , 'time':time}) # store notched data in data array
    da_bipol = eeg_mono_to_bipol(da, dérivations) # store notched eeg_eog data in data array
    da_physios = da.loc[physio_chans,:]
    data_preproc_bipol_with_physio = xr.concat([da_bipol , da_physios], dim = 'chan').assign_attrs({'unit':'uV', 'srate':srate}) # concat eeg data bipol with physio data

    # MONOPOLAR REREF MASTOÏD
    da_reref_masto = eeg_mono_reref_masto(da)
    data_reref_masto_with_physio = xr.concat([da_reref_masto , da_physios], dim = 'chan').assign_attrs({'unit':'uV', 'srate':srate})

    # SLEEP STAGING
    raw_bipol = da_to_mne_object(data_preproc_bipol_with_physio, srate) # remake an mne object in V for sleep staging from YASA
    sls = yasa.SleepStaging(raw_bipol , eeg_name = chan_sleep_staging , eog_name = 'EOGG-A2', emg_name='Menton') # make sleep staging with yasa
    hypno_str = sls.predict()
    hypno_int = yasa.hypno_str_to_int(hypno_str)
    hypno_df = pd.DataFrame(columns = ['str','int']) # concat a version with strings labels and with int
    hypno_df['str'] = hypno_str
    hypno_df['int'] = hypno_int
    hypno_upsampled = yasa.hypno_upsample_to_data(hypno = hypno_int, sf_hypno=1/30, data=da_bipol, sf_data=srate) # upsample to data
    hypno_df_upsampled = pd.DataFrame(columns = ['str','int'])
    hypno_df_upsampled['int'] = hypno_upsampled
    hypno_df_upsampled['str'] = yasa.hypno_int_to_str(hypno_upsampled)

    # SPECTROGRAM WITH HYPNOGRAM
    fig = yasa.plot_spectrogram(data_preproc_bipol_with_physio.sel(chan = chan_sleep_staging).values, sf=srate, hypno=hypno_upsampled, cmap='Spectral_r')
    fig.suptitle(subject, fontsize = 20)
    plt.savefig(f'../spectrograms/{subject}_spectrogram')
    plt.close()

    # SLEEP STATISTICS
    sleep_stats = yasa.sleep_statistics(hypno_int, sf_hyp=1/30) # compute usual stats
    sleep_stats = pd.DataFrame.from_dict(sleep_stats, orient = 'index').T # put in in dataframe
    sleep_stats.insert(0, 'subject', subject)

    # SAVING DATA
    metadata.to_excel(f'../subject_characteristics/metadata_{subject}.xlsx') # save metadata from subjects
    data_preproc_bipol_with_physio.to_netcdf(f'../preproc/{subject}_bipol.nc') # save an xarray containing bipolarized preprocessed data
    data_reref_masto_with_physio.to_netcdf(f'../preproc/{subject}_reref.nc') # save an xarray containing reref to average matoid preprocessed data
    hypno_df.to_excel(f'../hypnos/hypno_{subject}.xlsx') # save hypnogram
    hypno_df_upsampled.to_csv(f'../hypnos/hypno_upsampled_{subject}.csv') #save an upsampled to data size version of hypnogram (in csv because too large for xlsx)
    sleep_stats.to_excel(f'../subject_characteristics/{subject}_sleep_stats.xlsx') # save sleep statistics

    concat_sleep_stats.append(sleep_stats)
    concat_metadata.append(metadata)

# SAVING CONCATENATED DATA
sleep_stats_all = pd.concat(concat_sleep_stats) # concat list of df
metadata_all = pd.concat(concat_metadata) # concat list of df
sleep_stats_all.to_excel('../subject_characteristics/global_sleep_stats.xlsx')
metadata_all.to_excel('../subject_characteristics/global_metadata.xlsx')

        