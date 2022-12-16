import numpy as np
import pandas as pd
import xarray as xr
import yasa
from params import *

mapper = {0:'W',1:'N1',2:'N2',3:'N3',4:'R'} # mapper from int stage to str stage (same than YASA)

for subject in subjects:
    print(subject)
    data = xr.open_dataarray(f'../preproc/{subject}_reref.nc') # slow-waves will be detected on reref data on average mastoid
    data_eeg = data.sel(chan = eeg_mono_chans).values # sel reref eeg data only (without physio)
    
    for encoder in ['yasa','human']: # loop on both types of encoders of hypnogram to get both version of staging of events
        print(encoder)
        hypno_upsampled = pd.read_csv(f'../hypnos/hypno_upsampled_{subject}_{encoder}.csv', index_col = 0) # load upsampled hypnogram of the subject
        hypno_upsampled_int =  hypno_upsampled['int'] # get the integer version of hypnogram upsampled

        for event_type in ['spindle','slow_wave']: # loop on both types of events (spindles or slow waves)
            print(event_type)
            
            if event_type == 'spindle':
                detec = yasa.spindles_detect(data=data_eeg, sf=srate, ch_names=eeg_mono_chans,freq_sp=freq_sp,
                                             duration=sp_duration, min_distance=sp_min_distance, thresh=sp_thresh,
                                             multi_only=False, remove_outliers=True, hypno = hypno_upsampled_int,
                                             include = (0,1,2,3,4)) # detection spindles
                destination_file = f'../event_detection/{subject}_spindles_reref_{encoder}.xlsx'
            elif event_type == 'slow_wave':
                detec = yasa.sw_detect(data=data_eeg, sf=srate, ch_names=eeg_mono_chans, hypno=hypno_upsampled_int,
                                       include=(0,1,2,3,4), freq_sw=freq_sw, dur_neg=sw_dur_neg, dur_pos=sw_dur_pos,
                                       amp_neg=sw_amp_neg, amp_pos=sw_amp_neg, amp_ptp=sw_amp_ptp, coupling=True,
                                       remove_outliers=True, verbose=False) # detection slow-waves
                destination_file = f'../event_detection/{subject}_slowwaves_reref_{encoder}.xlsx'

            events = detec.summary() # get results summary
            events['Stage_Letter'] = events['Stage'].map(mapper) # add a column with letters instead of numbers for staging

            events.to_excel(destination_file) # save

    