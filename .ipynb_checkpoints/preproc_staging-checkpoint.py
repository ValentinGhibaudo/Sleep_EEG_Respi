import numpy as np
import pandas as pd 
import xarray as xr
import yasa
import glob
import mne
from datetime import datetime
import time
import matplotlib.pyplot as plt
from params import subjects, dérivations, chan_sleep_staging, mapper_human_stages_to_yasa_stages, mapper_yasa_encoding


## FUNCTIONS

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

## RUNNING

light_times = pd.read_excel('../data/lights_recordings.xlsx', index_col = 0).set_index('subject', drop = True) # load a table containing hours of start recording, stop recording, light out, light on

concat_sleep_stats_yasa = [] # init list that will contain sleep statistics df from each subject computed on yasa hypnogram
concat_sleep_stats_human = [] # init list that will contain sleep statistics df from each subject computed on human hypnogram
concat_metadata = [] # list that will contain metadata df from each subject

for subject in subjects:
    print(subject)
    
    if subject == 'S8': # S8 has his recording divided into two parts so they are concatenated
        input_files = glob.glob(f'../data/{subject}/*.edf')
        raws = [mne.io.read_raw_edf(file, verbose = False, preload = True) for file in input_files]
        raw = mne.concatenate_raws(raws)
    else: # else, just load into memory
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
    duration = time[-1] # get last time point in secs
    
    data = [subject, name, gender, age, srate , hp , lp  , duration , round((duration / 60),2) , round((duration / 3600) , 2)]
    index = ['subject','name','gender','age','srate','HP','LP','sec duration','min duration','hour duration']
    metadata = pd.Series(data, index=index).to_frame().T # metadata
    
    # CROP
    crop_duration_at_beginning = (datetime.strptime(light_times.loc[subject, 'light out'], "%H:%M:%S") - datetime.strptime(light_times.loc[subject, 'recording start time'], "%H:%M:%S")).seconds # get n seconds to remove a beginning according to light out
    tmin = crop_duration_at_beginning # tmin = light out time
    if light_times.loc[subject, 'light on before stop recording'] == 'yes': # if light on is before stop recording
        crop_duration_at_end = (datetime.strptime(light_times.loc[subject, 'recording stop time'], "%H:%M:%S") - datetime.strptime(light_times.loc[subject, 'light on'], "%H:%M:%S")).seconds # get n seconds to remove at end according to light on
        tmax = duration - crop_duration_at_end # tmax = recording last time - light on time in secs
    else: # elif light on is after stop recording, tmax = recording last time
        tmax = duration 
    
    raw_crop = raw.copy()
    raw_crop.crop(tmin=tmin, tmax=tmax) # crop data from tmin to tmax to keep data from light out to light on
    raw_crop_time = raw_crop.times
    
    # NOTCH 
    raw_notch = raw_crop.copy()
    raw_notch.notch_filter(freqs = [50,100], verbose = False) # unit = V 
    data = raw_notch.get_data(units = 'uV') # extract data in unit = uV
    
    # BIPOLARIZATION
    da = xr.DataArray(data = data, dims = ['chan','time'] , coords = {'chan':chans , 'time':raw_crop_time}) # store notched data in data array
    da_bipol = eeg_mono_to_bipol(da, dérivations) # store notched eeg_eog data in data array
    da_physios = da.loc[physio_chans,:]
    da_eog = da_bipol.sel(chan = ['EOGDt-A1','EOGG-A2']) # EOG are referenced against opposite mastoid
    data_preproc_bipol_with_physio = xr.concat([da_bipol , da_physios], dim = 'chan').assign_attrs({'unit':'uV', 'srate':srate}) # concat eeg data bipol with physio data

    # MONOPOLAR REREF MASTOÏD
    da_reref_masto = eeg_mono_reref_masto(da)
    data_reref_masto_with_physio = xr.concat([da_reref_masto , da_physios, da_eog], dim = 'chan').assign_attrs({'unit':'uV', 'srate':srate})

    # SLEEP STAGING
    raw_for_sleep_staging = da_to_mne_object(data_reref_masto_with_physio, srate) # remake an mne object in V for sleep staging from YASA
    sls = yasa.SleepStaging(raw_for_sleep_staging , eeg_name = chan_sleep_staging , eog_name = 'EOGG-A2', emg_name='Menton') # make sleep staging with yasa according to set channel
    yasa_hypno_str = sls.predict()
    yasa_hypno_int = yasa.hypno_str_to_int(yasa_hypno_str)
    yasa_hypno_df = pd.DataFrame(columns = ['str','int']) # concat a version with strings labels and with int
    yasa_hypno_df.loc[:,'str'] = yasa_hypno_str
    yasa_hypno_df.loc[:,'int'] = yasa_hypno_int
    yasa_hypno_upsampled = yasa.hypno_upsample_to_data(hypno = yasa_hypno_int, sf_hypno=1/30, data=data_reref_masto_with_physio, sf_data=srate) # upsample to data
    yasa_hypno_df_upsampled = pd.DataFrame(columns = ['str','int'])
    yasa_hypno_df_upsampled.loc[:,'int'] = yasa_hypno_upsampled
    yasa_hypno_df_upsampled.loc[:,'str'] = yasa.hypno_int_to_str(yasa_hypno_upsampled)
    
    # PROCESS HUMAN SLEEP STAGING
    human_hypno_raw = pd.read_csv(f'../data/hypnograms_human_made/Hypno{subject}.txt' , sep = '\t', header = None).rename(columns = {0:'sec',1:'hour',2:'stage',3:'stage_code'}) # load human hypnogram and set colnames
    human_hypno = human_hypno_raw[ (human_hypno_raw['sec'] > tmin) & (human_hypno_raw['sec'] < tmax)] # crop human hypnogram according to tmin and tmax
    human_hypno.loc[:,'str'] = human_hypno['stage'].map(mapper_human_stages_to_yasa_stages) # replace human stage labels by yasa labels
    human_hypno.loc[:,'int'] = human_hypno['str'].map(mapper_yasa_encoding) # get an encoded version of stages relabeled (yasa code)
    human_hypno_upsampled = yasa.hypno_upsample_to_data(hypno=human_hypno['int'].values, sf_hypno = 1/30, data = data_reref_masto_with_physio, sf_data = srate) # upsample human hypno to data
    human_hypno_upsampled_df = pd.DataFrame(columns = ['str','int'])
    human_hypno_upsampled_df.loc[:,'int'] = human_hypno_upsampled
    human_hypno_upsampled_df.loc[:,'str'] = yasa.hypno_int_to_str(human_hypno_upsampled)
    
    # SPECTROGRAM WITH HYPNOGRAM
    fig = yasa.plot_spectrogram(data_reref_masto_with_physio.sel(chan = chan_sleep_staging).values, sf=srate, hypno=yasa_hypno_upsampled, cmap='Spectral_r')
    fig.suptitle(subject, fontsize = 20)
    plt.savefig(f'../spectrograms/{subject}_spectrogram_yasa')
    plt.close()

    # SLEEP STATISTICS
    sleep_stats_df = {}
    for encoder, hypno in zip(['yasa','human'],[yasa_hypno_df_upsampled['int'].values, human_hypno_upsampled_df['int'].values]):
        sleep_stats = yasa.sleep_statistics(hypno, sf_hyp=srate) # compute usual stats on upsampled hypno
        sleep_stats = pd.DataFrame.from_dict(sleep_stats, orient = 'index').T # put in in dataframe
        sleep_stats.insert(0, 'encoder', encoder)
        sleep_stats.insert(0, 'subject', subject)
        sleep_stats_df[encoder] = sleep_stats

    # SAVING DATA
    metadata.to_excel(f'../subject_characteristics/metadata_{subject}.xlsx') # save metadata from subjects
    data_preproc_bipol_with_physio.to_netcdf(f'../preproc/{subject}_bipol.nc') # save an xarray containing bipolarized preprocessed data
    data_reref_masto_with_physio.to_netcdf(f'../preproc/{subject}_reref.nc') # save an xarray containing reref to average matoid preprocessed data
    yasa_hypno_df.to_excel(f'../hypnos/hypno_{subject}_yasa.xlsx') # save yasa hypnogram
    yasa_hypno_df_upsampled.to_csv(f'../hypnos/hypno_upsampled_{subject}_yasa.csv') #save an upsampled to data size version of yasa hypnogram (in csv because too large for xlsx)
    human_hypno.to_excel(f'../hypnos/hypno_{subject}_human.xlsx') # save human hypnogram
    human_hypno_upsampled_df.to_csv(f'../hypnos/hypno_upsampled_{subject}_human.csv') #save an upsampled to data size version of human hypnogram (in csv because too large for xlsx)
    sleep_stats_df['yasa'].to_excel(f'../subject_characteristics/{subject}_sleep_stats_yasa.xlsx') # save sleep statistics computed from yasa hypnogram
    sleep_stats_df['human'].to_excel(f'../subject_characteristics/{subject}_sleep_stats_human.xlsx') # save sleep statistics computed from human hypnogram

    concat_sleep_stats_yasa.append(sleep_stats_df['yasa'])
    concat_sleep_stats_human.append(sleep_stats_df['human'])
    concat_metadata.append(metadata)

## SAVING CONCATENATED DATA
sleep_stats_all_yasa = pd.concat(concat_sleep_stats_yasa) # concat list of df
sleep_stats_all_human = pd.concat(concat_sleep_stats_human) # concat list of df
metadata_all = pd.concat(concat_metadata) # concat list of df
sleep_stats_all_yasa.to_excel('../subject_characteristics/global_sleep_stats_yasa.xlsx')
sleep_stats_all_human.to_excel('../subject_characteristics/global_sleep_stats_human.xlsx')
metadata_all.to_excel('../subject_characteristics/global_metadata.xlsx')

