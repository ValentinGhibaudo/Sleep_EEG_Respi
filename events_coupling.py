import numpy as np
import pandas as pd
import json
import xarray as xr
from params import subjects, timestamps_labels, channels_events_select, stages_events_select

###### USEFUL FUNCTIONS

def load_resp_features(subject):
    rsp = pd.read_excel(f'../resp_features/{subject}_resp_features_tagged.xlsx', index_col = 0)
    return rsp

def load_events(subject, event_type):
    if event_type == 'sp':
        return pd.read_excel(f'../event_detection/{subject}_spindles_reref_human.xlsx', index_col = 0)
    elif event_type == 'sw':
        return pd.read_excel(f'../event_detection/{subject}_slowwaves_reref_human.xlsx', index_col = 0)

def get_phase_angles(rsp_features, event_times):
    list_of_angles = [] # phase angles of events found in each cycle are stored in a list
    for i, cycle in rsp_features.iterrows(): # loop on respi cycles
        start = cycle['start_time'] # get start time of the cycle
        stop = cycle['stop_time'] # get stop time of the cycle
        duration = cycle['cycle_duration'] # get duration of the cycle
        mask_of_the_cycle = (event_times >= start) & (event_times < stop) # mask events between start and stop
        if sum(mask_of_the_cycle) != 0: # do the next steps if some events have been found during the cycle
            events_times_of_the_cycle = event_times[mask_of_the_cycle] # keep the events of the cycle
            relative_times = (events_times_of_the_cycle - start) / duration # = time after start / duration = relative times between 0 and 1
            phase_angles = relative_times * 2*np.pi # relative times to phase angles between 0 and 2Pi
            list_of_angles.append(phase_angles)
            
    pooled_angles = []
    for element in list_of_angles:
        if type(element) is float: # if element is just a float, it is appended to a list
            pooled_angles.append(element)
        else:
            for angle in element: # else, looping on angles of the element variable
                pooled_angles.append(angle)
        
    return np.array(pooled_angles)

def save_dict(dictionnary, path):
    file = json.dumps(dictionnary) # create json object from dictionary
    with open(path, 'w',encoding='utf-8') as convert_file:
        convert_file.write(file)


##### COMPUTING

pooled_angles = {'sp':{'N2':[],'N3':[]} , 'sw':{'N2':[],'N3':[]}} # pooling angles from all subjects

for subject in subjects: # loop on run keys
    print(subject)
    rsp_features = load_resp_features(subject) # load rsp features
    
    for event_type in ['sp','sw']: # sp = spindles ; sw = slowwaves
        events = load_events(subject, event_type) # load sp or sw
        mask_channels = events['Channel'].isin(channels_events_select) # select events only detected in the dict "channels_events_select" of event
        events_of_sel_chans = events[mask_channels] # keep events of set channels
        if event_type == 'sp': 
            rsp_features_with_the_event = rsp_features[rsp_features['Spindle_Tag'] == 1] # keep only rsp cycles with spindles inside
        elif event_type == 'sw':
            rsp_features_with_the_event = rsp_features[rsp_features['SlowWave_Tag'] == 1] # keep only rsp cycles with slowwaves inside

        for stage in stages_events_select: # loop only on events detected in the list "stages_events_select"
            mask_stages = events_of_sel_chans['Stage_Letter'] == stage # mask on the stage
            events_of_chan_of_stage = events_of_sel_chans[mask_stages]  # keep event of set stage
            event_times = events_of_chan_of_stage[timestamps_labels[event_type]].values # get np array of events timings that summarize the whole events (set in params)
            rsp_features_of_the_stage = rsp_features_with_the_event[rsp_features_with_the_event['sleep_stage'] == stage] # keep only respi cycles of the stage (and with events inside)
            
            phase_angles_rsp = get_phase_angles(rsp_features_of_the_stage, event_times) # compute phase angles of event for each respi cycle
            np.save(f'../events_coupling/{subject}_{event_type}_{stage}_phase_angles.npy', phase_angles_rsp) # save angles from the subject & event type & stage as a .npy
            
            pooled_angles[event_type][stage].append(phase_angles_rsp.tolist()) # append to the right keys (event type & stage) a list of angles
            
save_dict(pooled_angles, '../events_coupling/pooled_angles.txt')
            

            

            
            

            





    

        

