import numpy as np
import pandas as pd 
import xarray as xr
import yasa
import glob
import mne
from datetime import datetime
import time
import matplotlib.pyplot as plt
from bibliotheque import init_da
from params import *
from configuration import *
import jobtools


# JOB METADATA 

def get_metadata(run_key, **p): # get metadata from raw
    if run_key == 'S8': # S8 has his recording divided into two parts so they are concatenated
        input_files = glob.glob(f'../data/{run_key}/*.edf')
        raws = [mne.io.read_raw_edf(file, verbose = False, preload = False) for file in input_files]
        raw = mne.concatenate_raws(raws)
    else: # else, just load into memory
        input_file = glob.glob(f'../data/{run_key}/*.edf')[0]
        raw = mne.io.read_raw_edf(input_file, verbose = False, preload = False) # read in Volts

    info = raw.info # info object
        
    # get participant metadata
    gender = raw.__dict__['_raw_extras'][0]['subject_info']['sex']
    birthday = raw.__dict__['_raw_extras'][0]['subject_info']['birthday']
    age = int((datetime.now() - birthday).days / 365)
    
    # get recording metadata
    srate = info['sfreq'] # get srate
    hp = info['highpass'] # get highpass
    lp = info['lowpass'] # get lowpass
    time = raw.times # get time vector
    duration = time[-1] # get last time point in secs
    
    data = [run_key, gender, age, srate , hp , lp  , duration , round((duration / 60),2) , round((duration / 3600) , 2)]
    index = ['subject','gender','age','srate','HP','LP','sec duration','min duration','hour duration']
    metadata = pd.Series(data, index=index).to_frame().T # metadata
    ds = xr.Dataset(metadata)
    return ds

metadata_job = jobtools.Job(precomputedir, 'metadata', metadata_params, get_metadata)
jobtools.register_job(metadata_job)

def test_get_metadata():
    run_key = 'S8'
    metadata = get_metadata(run_key, **metadata_params).to_dataframe()
    print(metadata)


# JOB PREPROC
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

def compute_preproc(run_key, **p): # notch filter & crop on light times & reref on mean of mastoid signals
    light_times = pd.read_excel(base_folder / 'data' / 'lights_recordings.xlsx', index_col = 0).set_index('subject', drop = True) # load a table containing hours of start recording, stop recording, light out, light on

    if run_key == 'S8': # S8 has his recording divided into two parts so they are concatenated
        input_files = glob.glob(f'../data/{run_key}/*.edf')
        raws = [mne.io.read_raw_edf(file, verbose = False, preload = True) for file in input_files]
        raw = mne.concatenate_raws(raws)
    else: # else, just load into memory
        input_file = glob.glob(f'../data/{run_key}/*.edf')[0]
        raw = mne.io.read_raw_edf(input_file, verbose = False, preload = True) # read in Volts

    info = raw.info # info object
        
    # get recording metadata
    srate = info['sfreq'] # get srate
    chans = info['ch_names'] # get chans
    physio_chans = p['physio_chans']
    time = raw.times # get time vector
    duration = time[-1] # get last time point in secs

    # CROP
    crop_duration_at_beginning = (datetime.strptime(light_times.loc[run_key, 'light out'], "%H:%M:%S") - datetime.strptime(light_times.loc[run_key, 'recording start time'], "%H:%M:%S")).seconds # get n seconds to remove at beginning according to light out
    tmin = crop_duration_at_beginning # tmin = light out time
    if light_times.loc[run_key, 'light on before stop recording'] == 'yes': # if light on is before stop recording
        crop_duration_at_end = (datetime.strptime(light_times.loc[run_key, 'recording stop time'], "%H:%M:%S") - datetime.strptime(light_times.loc[run_key, 'light on'], "%H:%M:%S")).seconds # get n seconds to remove at end according to light on
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
    
    # MONOPOLAR REREF MASTOÏD
    da = xr.DataArray(data = data, dims = ['chan','time'] , coords = {'chan':chans , 'time':raw_crop_time}) # store notched data in data array
    da_physios = da.loc[physio_chans,:]
    da_eog = init_da({'chan':['EOGG-A2','EOGDt-A1'], 'time':raw_crop_time})
    da_eog.loc['EOGG-A2',:] = da.loc['EOGG',:].values - da.loc['EEG A2',:].values # EOG are referenced against opposite mastoid
    da_eog.loc['EOGDt-A1',:] = da.loc['EOGDt',:].values - da.loc['EEG A1',:].values # EOG are referenced against opposite mastoid
 
    da_reref_masto = eeg_mono_reref_masto(da)
    data_reref_masto_with_physio = xr.concat([da_reref_masto , da_physios, da_eog], dim = 'chan').assign_attrs({'unit':'uV', 'srate':srate})

    ds = xr.Dataset()
    ds['preproc'] = data_reref_masto_with_physio
    return ds

preproc_job = jobtools.Job(precomputedir, 'preproc', preproc_params, compute_preproc)
jobtools.register_job(preproc_job)

def test_preproc():
    run_key = 'S8'
    preproc = compute_preproc(run_key, **preproc_params)['preproc']
    print(preproc)


# JOB SLEEP STAGING 

def da_to_mne_object(da, srate):
    ch_names = list(da.coords['chan'].values)
    sfreq = srate
    info = mne.create_info(ch_names, sfreq, ch_types='misc', verbose=False)
    raw = mne.io.RawArray(data = da.values, info=info, first_samp=0, copy='auto', verbose=False)
    return raw

def compute_hypnogram(run_key, **p):
    data_reref_masto_with_physio = preproc_job.get(run_key)['preproc']
    srate = data_reref_masto_with_physio.attrs['srate']
    raw_for_sleep_staging = da_to_mne_object(data_reref_masto_with_physio, srate) # remake an mne object in V for sleep staging from YASA
    sls = yasa.SleepStaging(raw_for_sleep_staging , eeg_name = p['chan_sleep_staging'] , eog_name = p['eog_name'], emg_name=p['emg_name']) # make sleep staging with yasa according to set channel
    yasa_hypno_str = sls.predict()
    yasa_hypno_int = yasa.hypno_str_to_int(yasa_hypno_str)
    yasa_hypno_df = pd.DataFrame(columns = ['str','int']) # concat a version with strings labels and with int
    yasa_hypno_df['str'] = yasa_hypno_str
    yasa_hypno_df['int'] = yasa_hypno_int
    return xr.Dataset(yasa_hypno_df)

hypnogram_job = jobtools.Job(precomputedir, 'hypnogram', sleep_staging_params, compute_hypnogram)
jobtools.register_job(hypnogram_job)

def test_compute_hypnogram():
    run_key = 'S1'
    print(compute_hypnogram(run_key, **sleep_staging_params).to_dataframe())


def upsample_hypnogram(run_key, **p):
    yasa_hypno_int = hypnogram_job.get(run_key).to_dataframe()['int']
    data_reref_masto_with_physio = preproc_job.get(run_key)['preproc']
    srate = data_reref_masto_with_physio.attrs['srate']
    yasa_hypno_upsampled = yasa.hypno_upsample_to_data(hypno = yasa_hypno_int, sf_hypno=1/30, data=data_reref_masto_with_physio, sf_data=srate) # upsample to data
    yasa_hypno_df_upsampled = pd.DataFrame(columns = ['str','int'])
    yasa_hypno_df_upsampled['int'] = yasa_hypno_upsampled
    yasa_hypno_df_upsampled['str'] = yasa.hypno_int_to_str(yasa_hypno_upsampled)
    return xr.Dataset(yasa_hypno_df_upsampled)

upsample_hypno_job = jobtools.Job(precomputedir, 'hypnogram_upsample', upsample_hypno_params, upsample_hypnogram)
jobtools.register_job(upsample_hypno_job)

def test_upsample_hypnogram():
    run_key = 'S1'
    print(upsample_hypnogram(run_key, **upsample_hypno_params).to_dataframe())



# JOB SPECTROGRAM

def compute_spectrogram(run_key, **p):
    data_reref_masto_with_physio = preproc_job.get(run_key)['preproc']
    srate = data_reref_masto_with_physio.attrs['srate']
    yasa_hypno_upsampled = upsample_hypno_job.get(run_key).to_dataframe()['int']
    chan_sleep_staging = p['sleep_staging_params']['chan_sleep_staging']
    fig = yasa.plot_spectrogram(data_reref_masto_with_physio.sel(chan = chan_sleep_staging).values, sf=srate, hypno=yasa_hypno_upsampled, cmap='Spectral_r')
    fig.suptitle(run_key, fontsize = 20)
    fig.savefig(base_folder / 'results' / 'spectrograms' / f'{run_key}_spectrogram.png', bbox_inches = 'tight')
    plt.close()
    return None

spectrogram_job = jobtools.Job(precomputedir, 'spectrogram', spectrogram_params, compute_spectrogram)
jobtools.register_job(spectrogram_job)

def test_compute_spectrogram():
    run_key = 'S1'
    compute_spectrogram(run_key, **spectrogram_params)



# JOB SLEEP STATS

def compute_sleep_stats(run_key, **p):
    hypno = hypnogram_job.get(run_key).to_dataframe()['int']
    sleep_stats = yasa.sleep_statistics(hypno, sf_hyp=1/30) # compute usual stats hypno
    sleep_stats = pd.DataFrame.from_dict(sleep_stats, orient = 'index').T # put in in dataframe
    sleep_stats.insert(0, 'subject', run_key)
    return xr.Dataset(sleep_stats)

sleep_stats_job = jobtools.Job(precomputedir, 'sleep_stats', sleep_stats_params, compute_sleep_stats)
jobtools.register_job(sleep_stats_job)

def test_compute_sleep_stats():
    run_key = 'S1'
    print(compute_sleep_stats(run_key, **sleep_stats_params).to_dataframe())



def save_useful_outputs(): # save concatenated version of sleep stats and metadata of all subjects
    concat_sleep_stats = []
    concat_metadata = []

    for run_key in run_keys:
        concat_sleep_stats.append(sleep_stats_job.get(run_key).to_dataframe())
        concat_metadata.append(metadata_job.get(run_key).to_dataframe())
        # spectrogram_job.get(run_key)
    
    all_sleep_stats = pd.concat(concat_sleep_stats).set_index('subject')
    all_sleep_stats_save = all_sleep_stats.copy()
    all_sleep_stats_save.loc['Mean',:] = all_sleep_stats.mean(axis = 0)
    all_sleep_stats_save.loc['SD',:] = all_sleep_stats.std(axis = 0)
    
    all_metadata = pd.concat(concat_metadata)

    all_sleep_stats_save.round(2).reset_index().to_excel(base_folder / 'results' / 'subject_characteristics' / 'sleep_stats.xlsx', index = False)
    all_metadata.to_excel(base_folder / 'results' / 'subject_characteristics' / 'metadata.xlsx', index = False)


def compute_all():
    jobtools.compute_job_list(metadata_job, run_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(preproc_job, run_keys, force_recompute=False, engine='loop')


if __name__ == '__main__':
    # test_get_metadata()
    # test_preproc()
    # test_compute_hypnogram()
    # test_upsample_hypnogram()
    # test_compute_spectrogram()
    # test_compute_sleep_stats()

    save_useful_outputs()

    # compute_all()
