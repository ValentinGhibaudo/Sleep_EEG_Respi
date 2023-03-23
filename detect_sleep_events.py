import numpy as np
import pandas as pd
import xarray as xr
import yasa
import json
from params import *

mapper = {0:'W',1:'N1',2:'N2',3:'N3',4:'R'} # mapper from int stage to str stage (same than YASA)

def save_dict(dictionnary, path):
    file = json.dumps(dictionnary, indent = 4) # create json object from dictionary
    with open(path, 'w',encoding='utf-8') as convert_file:
        convert_file.write(file)

save_running_params = {'run_keys':subjects,
                       'srate':srate,
                       'ch_names':eeg_mono_chans,
                       'sp':{
                           'freq_sp':freq_sp,
                           'duration':sp_duration,
                           'min_distance':sp_min_distance,
                           'thresh':sp_thresh,
                           'remove_outliers':remove_outliers_sp,
                           'include':include_sp_stages},
                        'sw':{
                            'freq_sw':freq_sw,
                            'dur_neg':sw_dur_neg,
                            'dur_pos':sw_dur_pos,
                            'amp_neg':sw_amp_neg,
                            'amp_pos':sw_amp_pos,
                            'amp_ptp':sw_amp_ptp,
                            'remove_outliers':remove_outliers_sw,
                            'include':include_sw_stages
                        }
                       }
                      
print('RUNNING PARAMS')
print(save_running_params)

save_dict(save_running_params, path = '../event_detection/running_params')
                           

for subject in subjects:
    print(subject)
    data = xr.open_dataarray(f'../preproc/{subject}_reref.nc') # slow-waves will be detected on reref data on average mastoid
    data_eeg = data.sel(chan = eeg_mono_chans).values # sel reref eeg data only (without physio)

    hypno_upsampled = pd.read_csv(f'../hypnos/hypno_upsampled_{subject}_yasa.csv', index_col = 0) # load upsampled hypnogram of the subject
    hypno_upsampled_int =  hypno_upsampled['int'] # get the integer version of hypnogram upsampled

    for event_type in ['spindle','slow_wave']: # loop on both types of events (spindles or slow waves)
        print(event_type)

        if event_type == 'spindle':
            detec = yasa.spindles_detect(data=data_eeg, sf=srate, ch_names=eeg_mono_chans,freq_sp=freq_sp,
                                         duration=sp_duration, min_distance=sp_min_distance, thresh=sp_thresh,
                                         multi_only=False, remove_outliers=remove_outliers_sp, hypno = hypno_upsampled_int,
                                         include = include_sp_stages, verbose = False) # detection spindles
            destination_file = f'../event_detection/{subject}_spindles_reref_yasa.xlsx'
        elif event_type == 'slow_wave':
            detec = yasa.sw_detect(data=data_eeg, sf=srate, ch_names=eeg_mono_chans, hypno=hypno_upsampled_int,
                                   include=include_sw_stages, freq_sw=freq_sw, dur_neg=sw_dur_neg, dur_pos=sw_dur_pos,
                                   amp_neg=sw_amp_neg, amp_pos=sw_amp_pos, amp_ptp=sw_amp_ptp, coupling=False,
                                   remove_outliers=remove_outliers_sw, verbose=False) # detection slow-waves
            destination_file = f'../event_detection/{subject}_slowwaves_reref_yasa.xlsx'

        events = detec.summary() # get results summary

        events['Stage_Letter'] = events['Stage'].map(mapper) # add a column with letters instead of numbers for staging

        min_in_N2 = hypno_upsampled['str'].value_counts()['N2'] / srate / 60
        min_in_N3 = hypno_upsampled['str'].value_counts()['N3'] / srate / 60

        density_N2 = events[(events['Channel'] == 'Fz') & (events['Stage_Letter'] == 'N2')].shape[0] / min_in_N2 # detection density in Fz in N2
        density_N3 = events[(events['Channel'] == 'Fz') & (events['Stage_Letter'] == 'N3')].shape[0] / min_in_N3 # detection density in Fz in N3

        print(f'{subject} - {event_type} density by minute in Fz - N2 : {round(density_N2, 2)} ; N3 : {round(density_N3, 2)}') # print detection density in Fz

        events.to_excel(destination_file) # save
        
        
        

    