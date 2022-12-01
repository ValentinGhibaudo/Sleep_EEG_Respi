import numpy as np
import pandas as pd
import yasa
import xarray as xr
from params import patients, srate, timestamps_labels

save = True

for patient in patients: # loop on run keys
    print(patient)
    data = xr.open_dataarray(f'../preproc/{patient}.nc') # open lazy data for hypno upsampling
    hypno = pd.read_excel(f'../hypnos/hypno_{patient}.xlsx', index_col = 0) # load hypno of the run key
    hypno_upsampled = yasa.hypno_upsample_to_data(hypno = hypno['yasa hypnogram'].values, sf_hypno=1/30, data=data, sf_data=srate) # upsample hypno
    rsp_features = pd.read_excel(f'../resp_features/{patient}_resp_features.xlsx', index_col = 0) # load rsp features
    idx_start = rsp_features['start'].values # select start indexes of the cycles
    stage_of_the_start_idxs = hypno_upsampled[idx_start] # keep stages corresponding to start resp cycle indexes
    rsp_features_staged = rsp_features.copy()
    rsp_features_staged['sleep_stage'] = stage_of_the_start_idxs # append sleep stage column to resp features
    
    event_in_resp = {'sp':[],'sw':[]} # list to encode if an event is present is in the resp cycle or not
    
    for event, event_load in zip(['sw','sp'],['slowwaves','spindles']): # loop on both types of events (slow waves and spindles)
        event_df = pd.read_excel(f'../event_detection/{patient}_{event_load}.xlsx', index_col = 0) # load dataframe of detected events
        event_times = event_df[timestamps_labels[event]].values # get np array of events timings that summarize the whole events (set in params)
        
        for c, row in rsp_features_staged.iterrows(): # loop on rsp cycles
            start = row['start_time'] # get start time of the cycle
            stop = row['stop_time'] # get stop time of the cycle
            
            events_of_the_cycle = event_times[(event_times > start) & (event_times < stop)] # keep event times present in the cycle
            
            if events_of_the_cycle.size != 0: # if at least one event is present in the cycle...
                event_in_resp[event].append(1) # ... append a 1
            else:
                event_in_resp[event].append(0)  # ... else append a 0
                
    
    rsp_features_tagged = rsp_features_staged.copy()
    rsp_features_tagged['Spindle_Tag'] = event_in_resp['sp'] # append spindle tagging column to resp features
    rsp_features_tagged['SlowWave_Tag'] = event_in_resp['sw'] # append slowwave tagging column to resp features
    
    if save:
        rsp_features_tagged.to_excel(f'../resp_features/{patient}_resp_features_tagged.xlsx') # save



