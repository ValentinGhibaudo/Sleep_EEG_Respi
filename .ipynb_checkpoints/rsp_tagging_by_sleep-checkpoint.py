import pandas as pd
from params import subjects, timestamps_labels, channels_events_select

"""
This script tags respiratory cycles with corresponding sleep stage, and notion of spindle or slow wave present inside 
"""

for subject in subjects: # loop on run keys
    print(subject)
    hypno_upsampled = pd.read_csv(f'../hypnos/hypno_upsampled_{subject}_yasa.csv', index_col = 0) # load upsampled hypnogram of the subject
    rsp_features = pd.read_excel(f'../resp_features/{subject}_resp_features.xlsx', index_col = 0) # load resp features
    idx_start = rsp_features['start'].values # select start indexes of the cycles
    stage_of_the_start_idxs = hypno_upsampled['str'][idx_start] # keep stages (strings) corresponding to start resp cycle indexes
    rsp_features_staged = rsp_features.copy()
    rsp_features_staged['sleep_stage'] = stage_of_the_start_idxs.values # append sleep stage column to resp features
    
    event_in_resp = {'sp':[],'sw':[]} # list to encode if an event is present is in the resp cycle or not
    
    
    # rsp_features_tagged['Spindle_Tag'] = None
    # rsp_features_tagged['SlowWave_Tag'] = None
    
    for event, event_load in zip(['sw','sp'],['slowwaves','spindles']): # loop on both types of events (slow waves and spindles)
        event_df = pd.read_excel(f'../event_detection/{subject}_{event_load}_reref_yasa.xlsx', index_col = 0) # load dataframe of detected events
        events = event_df[event_df['Channel'].isin(channels_events_select)] # keep only events detected in 'channels_events_select'
        event_times = events[timestamps_labels[event]].values # get np array of events timings that summarize the whole events (set in params)
        
        for c, row in rsp_features_staged.iterrows(): # loop on rsp cycles
            start = row['start_time'] # get start time of the cycle
            stop = row['stop_time'] # get stop time of the cycle
            
            events_of_the_cycle = event_times[(event_times >= start) & (event_times < stop)] # keep event times present in the cycle
            
            if events_of_the_cycle.size != 0: # if at least one event is present in the cycle...
                event_in_resp[event].append(1) # ... append a 1
            else:
                event_in_resp[event].append(0)  # ... else append a 0
            
            # rsp_features_tagged.at[c, event] = 
            # print(subject, event, sum(event_in_resp))
                
    
    rsp_features_tagged = rsp_features_staged.copy()
    rsp_features_tagged['Spindle_Tag'] = event_in_resp['sp'] # append spindle tagging column to resp features
    rsp_features_tagged['SlowWave_Tag'] = event_in_resp['sw'] # append slowwave tagging column to resp features
    

    rsp_features_tagged.to_excel(f'../resp_features/{subject}_resp_features_tagged.xlsx') # save



