import numpy as np
import pandas as pd
import xarray as xr
import yasa
from params import *

mode = 'bipol'
if mode == 'bipol':
    channels = eeg_chans
elif mode == 'reref':
    channels = eeg_mono_chans

mapper = {0:'W',1:'N1',2:'N2',3:'N3',4:'REM'} # mapper from int stage to str stage (same than YASA)

for subject in subjects:
    print(subject)
    hypno_upsampled = pd.read_csv(f'../hypnos/hypno_upsampled_{subject}.csv', index_col = 0) # load upsampled hypnogram of the subject
    hypno_upsampled_int =  hypno_upsampled['int'] # get the integer version of hypnogram upsampled
    
    data = xr.load_dataarray(f'../preproc/{subject}_{mode}.nc') # slow-waves will be detected on reref data on average mastoid
    data_eeg = data.sel(chan = channels).values # sel reref eeg data only (without physio)

    detec = yasa.sw_detect(data=data_eeg, sf=srate, ch_names=channels, hypno=hypno_upsampled_int, include=(0,1,2,3,4), freq_sw=freq_sw, dur_neg=sw_dur_neg, dur_pos=sw_dur_pos, amp_neg=sw_amp_neg, amp_pos=sw_amp_neg, amp_ptp=sw_amp_ptp, coupling=True, remove_outliers=True, verbose=False) # detection slow-waves
    destination_file = f'../event_detection/{subject}_slowwaves_{mode}.xlsx'
        
    events = detec.summary() # get results summary
    events['Stage_Letter'] = events['Stage'].map(mapper) # add a column with letters instead of numbers for staging
    
    events.to_excel(destination_file) # save

    