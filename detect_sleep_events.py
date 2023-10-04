import numpy as np
import pandas as pd
import xarray as xr
import yasa
from params import *
from configuration import *
import jobtools
from preproc_staging import preproc_job, upsample_hypno_job, hypnogram_job

# JOB SPINDLES DETECTION

def spindle_detection(run_key, **p):
    data = preproc_job.get(run_key)['preproc'] # slow-waves will be detected on reref data on average mastoid
    srate = data.attrs['srate']
    data_eeg = data.sel(chan = p['chans']).values # sel reref eeg data only (without physio)
    hypno_upsampled_int = upsample_hypno_job.get(run_key).to_dataframe()['int']

    detec = yasa.spindles_detect( # detection spindles by yasa
        data=data_eeg, 
        sf=srate, 
        ch_names=p['chans'],
        freq_sp=p['freq_sp'],
        duration=p['duration'], 
        min_distance=p['min_distance'], 
        thresh=p['thresh'],
        multi_only=False,
        remove_outliers=p['remove_outliers'],
        hypno = hypno_upsampled_int,
        include = p['include'],
        verbose = False) 
    
    spindles = detec.summary() # get results summary
    spindles['Stage_Letter'] = spindles['Stage'].map(p['mapper'])
    spindles.insert(0 , 'subject', run_key)

    return xr.Dataset(spindles)


spindles_detect_job = jobtools.Job(precomputedir, 'spindle_detection', spindle_detection_params, spindle_detection)
jobtools.register_job(spindles_detect_job)

def test_spindles_detection():
    run_key = 'S1'
    spindles = spindle_detection(run_key, **spindle_detection_params).to_dataframe()
    print(spindles)


# JOB SLOWWAVES DETECTION

def slowwave_detection(run_key, **p):
    data = preproc_job.get(run_key)['preproc'] # slow-waves will be detected on reref data on average mastoid
    srate = data.attrs['srate']
    data_eeg = data.sel(chan = p['chans']).values # sel reref eeg data only (without physio)
    hypno_upsampled_int = upsample_hypno_job.get(run_key).to_dataframe()['int']

    detec = yasa.sw_detect( # detection slow-waves by yasa
        data=data_eeg, 
        sf=srate, 
        ch_names=p['chans'], 
        hypno=hypno_upsampled_int,
        include=p['include'],
        freq_sw=p['freq_sw'],
        dur_neg=p['dur_neg'],
        dur_pos=p['dur_pos'],
        amp_neg=p['amp_neg'],
        amp_pos=p['amp_pos'],
        amp_ptp=p['amp_ptp'],
        coupling=False,
        remove_outliers=p['remove_outliers'],
        verbose=False) 
    
    slowwaves = detec.summary() # get results summary
    slowwaves['Stage_Letter'] = slowwaves['Stage'].map(p['mapper'])
    slowwaves.insert(0 , 'subject', run_key)

    return xr.Dataset(slowwaves)


slowwaves_detect_job = jobtools.Job(precomputedir, 'slowwave_detection', slowwaves_detection_params, slowwave_detection)
jobtools.register_job(slowwaves_detect_job)

def test_slowwave_detection():
    run_key = 'S1'
    slowwaves = slowwave_detection(run_key, **slowwaves_detection_params).to_dataframe()
    print(slowwaves)


# JOB EVENTS TAGGING
def cooccuring_sp_sw_df(spindles, slowwaves): 
    
    """
    This function will add columns to the initial dataframe giving information of presence of the spindle inside slowwave or not and in this case, when does it occur in the slowwave
    -------------
    Inputs =
    spindles : dataframe of spindles
    slowwaves : dataframe of slowwaves
    
    Outputs = 
    sp_return : dataframe of spindles labelized according to the co-occurence or not with slowwave
    sw_return : dataframe of slowwaves labelized according to the presence of spindles inside or not
    """
    
    features_cooccuring_sp = []
    sw_with_spindle_inside = []
    for ch in slowwaves['Channel'].unique(): # loop on chans
        for stage in slowwaves['Stage_Letter'].unique(): # loop on stages 
            sw_staged_ch = slowwaves[(slowwaves['Channel'] == ch)&(slowwaves['Stage_Letter'] == stage)]  # mask slowwave of the chan and the stage
            sp_staged_ch = spindles[(spindles['Channel'] == ch)&(spindles['Stage_Letter'] == stage)]  # mask spindles of the chan and the stage

            if not sw_staged_ch.shape[0] == 0: # if masked slowwave df is not empty ...
                for i, row in sw_staged_ch.iterrows(): # ... loop on rows = on slowwaves 

                    start_window = row['Start'] # get start time of the slowwave
                    stop_window = row['End'] # get stop time of the slowwave
                    negpeak = row['NegPeak'] # get time of the neg peak of the slowwave
                    duration = row['Duration'] # get duration of the slowwave

                    co_occuring_spindles = sp_staged_ch[(sp_staged_ch['Peak'] >= start_window) & (sp_staged_ch['Peak'] < stop_window)] # mask spindle present between start and stop time of slowwave
                    
                    if not co_occuring_spindles.shape[0] == 0: # if masked spindle df is not empty ...
                        sw_with_spindle_inside.append(i) # ... add to a list the index of the slowwave 
                        for s, sp in co_occuring_spindles.iterrows(): # loop on spindles of the slowwave

                            t = sp['Peak'] # get peak time of the spindle
                            cooccuring_with_sw = True # set boolean information that spindle is in a slowwave
                            features_cooccuring_sp.append([s, cooccuring_with_sw]) 
                            
                            
                        
    cooccurors = pd.DataFrame(features_cooccuring_sp, columns = ['index','cooccuring']).set_index('index') # spindle cooccuring df
    
    sp_return = spindles.reindex(columns = list(spindles.columns) + list(cooccurors.columns)) # extend the columns of the initial spindle df
    sp_return.loc[cooccurors.index,cooccurors.columns] = cooccurors # add the cooccuring df to initial spindle df
    sp_return.loc[:,'cooccuring'] = sp_return['cooccuring'].fillna(False) # fill na with False
    
    slowwaves_return = slowwaves.copy()
    slowwaves_return['cooccuring'] = np.nan 
    slowwaves_return.loc[sw_with_spindle_inside, 'cooccuring'] = True # add a column in the slowwave df with idea of presence of spindle inside or not
    slowwaves_return.loc[:,'cooccuring'] = slowwaves_return['cooccuring'].fillna(False)
    
    return sp_return, slowwaves_return

# job spindle tagging
def spindle_tagging(run_key, **p):
    spindles = spindles_detect_job.get(run_key).to_dataframe()
    slowwaves = slowwaves_detect_job.get(run_key).to_dataframe()
    
    
    sp_cooccuring, sw_cooccuring = cooccuring_sp_sw_df(spindles, slowwaves)
    
    sp_speed = sp_cooccuring.copy()
    sp_speed['Sp_Speed'] = np.nan
    sp_speed.loc[:,'Sp_Speed'] = (sp_speed['Frequency'] >= p['spindles_freq_threshold'][run_key]).map({False:'SS',True:'FS'}) # add a column on spindle df setting if the spindle is a slow or a fast spindle according to the set threshold manually chosen for each subject (bimodal distribution of frequency of spindles)
    
    hypno = hypnogram_job.get(run_key).to_dataframe()

    q1 = hypno['time'].quantile(0.25)
    median_time = hypno['time'].median()
    q3 = hypno['time'].quantile(0.75)

    sp_speed['night_quartile'] = np.nan

    ev_label = timestamps_labels['spindles']

    for i, row in sp_speed.iterrows():
        if row[ev_label] <= q1:
            sp_speed.loc[i,'night_quartile'] = 'q1'
        elif row[ev_label] > q1 and row[ev_label] <= median_time:
            sp_speed.loc[i,'night_quartile'] = 'q2'
        elif row[ev_label] > median_time and row[ev_label] <= q3:
            sp_speed.loc[i,'night_quartile'] = 'q3'
        elif row[ev_label] > q3:
            sp_speed.loc[i,'night_quartile'] = 'q4'

    sp_speed['cooccuring'] = sp_speed['cooccuring'].map({False:'notcooccur',True:'cooccur'})

    return xr.Dataset(sp_speed)

spindles_tag_job = jobtools.Job(precomputedir, 'spindle_tag', spindles_tagging_params, spindle_tagging)
jobtools.register_job(spindles_tag_job)

def test_spindle_tagging():
    run_key = 'S1'
    print(spindle_tagging(run_key, **spindles_tagging_params).to_dataframe())

# job slowwaves tagging
def slowwave_tagging(run_key, **p):
    spindles = spindles_detect_job.get(run_key).to_dataframe()
    slowwaves = slowwaves_detect_job.get(run_key).to_dataframe()
    sp_cooccuring, sw_cooccuring = cooccuring_sp_sw_df(spindles, slowwaves)
    
    hypno = hypnogram_job.get(run_key).to_dataframe()

    q1 = hypno['time'].quantile(0.25)
    median_time = hypno['time'].median()
    q3 = hypno['time'].quantile(0.75)

    sw_cooccuring['night_quartile'] = None

    ev_label = timestamps_labels['spindles']

    for i, row in sw_cooccuring.iterrows():
        if row[ev_label] <= q1:
            sw_cooccuring.loc[i,'night_quartile'] = 'q1'
        elif row[ev_label] > q1 and row[ev_label] <= median_time:
            sw_cooccuring.loc[i,'night_quartile'] = 'q2'
        elif row[ev_label] > median_time and row[ev_label] <= q3:
            sw_cooccuring.loc[i,'night_quartile'] = 'q3'
        elif row[ev_label] > q3:
            sw_cooccuring.loc[i,'night_quartile'] = 'q4'

    sw_cooccuring['cooccuring'] = sw_cooccuring['cooccuring'].map({False:'notcooccur',True:'cooccur'})

    return xr.Dataset(sw_cooccuring)

slowwaves_tag_job = jobtools.Job(precomputedir, 'slowwave_tag', slowwaves_tagging_params, slowwave_tagging)
jobtools.register_job(slowwaves_tag_job)

def test_slowwave_tagging():
    run_key = 'S1'
    print(slowwave_tagging(run_key, **slowwaves_tagging_params).to_dataframe())








def compute_all():
    # jobtools.compute_job_list(spindles_detect_job, run_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(spindles_detect_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 5)
    
    # jobtools.compute_job_list(slowwaves_detect_job, run_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(slowwaves_detect_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 5)

    jobtools.compute_job_list(spindles_tag_job, run_keys, force_recompute=True, engine='loop')
    jobtools.compute_job_list(slowwaves_tag_job, run_keys, force_recompute=True, engine='loop')


if __name__ == '__main__':
    # test_spindles_detection()
    # test_slowwave_detection()
    # test_spindle_tagging()
    # test_slowwave_tagging()

    compute_all()
