import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import yasa
from params import *

save = True

for patient in patients:
    print(patient)
    data = xr.load_dataarray(f'/crnldata/cmo/Etudiants/Valentin_G/Sleep_EEG_Respi/preproc/{patient}.nc')
    hypno = pd.read_excel(f'/crnldata/cmo/Etudiants/Valentin_G/Sleep_EEG_Respi/hypnos/hypno_{patient}.xlsx', index_col = 0)['yasa hypnogram'].values
    sf_hypno = 1/30 # sampling of the yasa hypnogram
    hypno_upsample = yasa.hypno_upsample_to_data(hypno, sf_hypno, data, sf_data=srate, verbose=True) # get one hypno value by time point
    hypno_upsampled_int = yasa.hypno_str_to_int(hypno_upsample) # transform letter to int code (0 = Wake, 1 = N1 sleep, 2 = N2 sleep, 3 = N3 sleep, 4 = REM sleep)
    data_eeg = data.sel(chan = eeg_chans).values # sel eeg data only (without physio)
    
    mapper = {0:'W',1:'N1',2:'N2',3:'N3',4:'REM'}
    
    for event_type in ['spindle','slow_wave']:
        if event_type == 'spindle':
            detec = yasa.spindles_detect(data=data_eeg, sf=srate, ch_names=eeg_chans, duration=sp_duration, min_distance=sp_min_distance, thresh=sp_thresh, multi_only=False, remove_outliers=True, hypno = hypno_upsampled_int, include = (0,1,2,3,4)) # detection
            destination_file = f'../event_detection/{patient}_spindles.xlsx'
        elif event_type == 'slow_wave':
            detec = yasa.sw_detect(data=data_eeg, sf=srate, ch_names=eeg_chans, hypno=hypno_upsampled_int, include=(0,1,2,3,4), freq_sw=freq_sw, dur_neg=sw_dur_neg, dur_pos=sw_dur_pos, amp_neg=sw_amp_neg, amp_pos=sw_amp_neg, amp_ptp=sw_amp_ptp, coupling=True, remove_outliers=True, verbose=False)
            destination_file = f'../event_detection/{patient}_slowwaves.xlsx'
            
        events = detec.summary() # get results summary
        events['Stage_Letter'] = events['Stage'].map(mapper) # add a column with letters instead of numbers for staging
        
        if save:
            events.to_excel(destination_file) # save

    