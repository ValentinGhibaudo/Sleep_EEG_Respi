import numpy as np
import pandas as pd 
import xarray as xr
import yasa
import glob
import mne
from tqdm import tqdm
from datetime import datetime
from params import patients, dérivations

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

def da_to_mne_object(da, srate):
    ch_names = list(da.coords['chan'].values)
    sfreq = srate
    info = mne.create_info(ch_names, sfreq, ch_types='misc', verbose=False)
    raw = mne.io.RawArray(data = da.values, info=info, first_samp=0, copy='auto', verbose=False)
    return raw



save = True

for patient in tqdm(patients):
    print(patient)
    
    input_file = glob.glob(f'../data/{patient}/*.edf')[0]
    raw = mne.io.read_raw_edf(glob.glob(f'../data/{patient}/*.edf')[0], verbose = False, preload = True) # read in Volts
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
    
    data = [patient, name, gender, age, srate , hp , lp  , duration , round((duration / 60),2) , round((duration / 3600) , 2)]
    index = ['participant','name','gender','age','srate','HP','LP','sec duration','min duration','hour duration']
    metadata = pd.Series(data, index=index).to_frame().T # metadata

    # NOTCH 
    raw_notch = raw.copy()
    raw_notch.notch_filter(freqs = [50,100], verbose = False) # unit = V 
    data = raw_notch.get_data(units = 'uV') # extract data in unit = uV
    
    # BIPOLARIZATION
    da = xr.DataArray(data = data, dims = ['chan','time'] , coords = {'chan':chans , 'time':time}) # store notched data in data array
    da_bipol = eeg_mono_to_bipol(da, dérivations) # store notched eeg_eog data in data array
    da_physios = da.loc[physio_chans,:]
    data_preproc = xr.concat([da_bipol , da_physios], dim = 'chan').assign_attrs({'unit':'uV', 'srate':srate}) # concat eeg data bipol with physio data

    # SLEEP STAGING
    raw_bipol = da_to_mne_object(data_preproc, srate) # remake an mne object in V for sleep staging from YASA
    sls = yasa.SleepStaging(raw_bipol , eeg_name = 'C4-T4' , eog_name = 'EOGG-A2', emg_name='Menton') # make sleep staging with yasa
    hypno = sls.predict()
    hypno = pd.Series(hypno, name = 'yasa hypnogram')
    
    if save:
        metadata.to_excel(f'../participant_characteristics/metadata_{patient}.xlsx')
        data_preproc.to_netcdf(f'../preproc/{patient}.nc')
        hypno.to_excel(f'../hypnos/hypno_{patient}.xlsx')
        