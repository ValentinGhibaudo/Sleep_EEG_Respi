import numpy as np
import pandas as pd
import json
from params import subjects, timestamps_labels, channels_events_select, stages_events_select

###### USEFUL FUNCTIONS

def load_resp_features(subject):
    rsp = pd.read_excel(f'../resp_features/{subject}_resp_features_tagged.xlsx', index_col = 0)
    return rsp

def load_events(subject, event_type):
    if event_type == 'sp':
        return pd.read_excel(f'../event_detection/{subject}_spindles.xlsx', index_col = 0)
    elif event_type == 'sw':
        return pd.read_excel(f'../event_detection/{subject}_slowwaves.xlsx', index_col = 0)

def get_phase_angles(rsp_features, event_times):
    angles_cycles = {} # phase angles of events found in each cycle are stored in a dict (keys = cycles idxs and values = phases angles)
    for i, cycle in rsp_features.iterrows(): # loop on respi cycles
        start = cycle['start_time'] # get start time of the cycle
        stop = cycle['stop_time'] # get stop time of the cycle
        duration = cycle['cycle_duration'] # get duration of the cycle
        mask_of_the_cycle = (event_times >= start) & (event_times < stop) # mask events between start and stop
        if sum(mask_of_the_cycle) != 0: # do the next steps if some events have been found during the cycle
            events_times_of_the_cycle = event_times[mask_of_the_cycle] # keep the events of the cycle
            relative_times = (events_times_of_the_cycle - start) / duration # = time after start / duration = relative times between 0 and 1
            phase_angles = relative_times * 2*np.pi # relative times to phase angles between 0 and 2Pi
            angles_cycles[i] = phase_angles.tolist() # store phase angles as lists to be stored in json (np.array is not serializable)
        else:
            angles_cycles[i] = None # None if no event found in the cycle
    return angles_cycles

def save_dict(dictionnary, path):
    file = json.dumps(dictionnary) # create json object from dictionary
    with open(path, 'w',encoding='utf-8') as convert_file:
        convert_file.write(file)

##### COMPUTING

for subject in subjects: # loop on run keys
    print(subject)
    rsp_features = load_resp_features(subject) # load rsp features

    for event_type in ['sp','sw']: # sp = spindles ; sw = slowwaves
        events = load_events(subject, event_type) # load sp or sw
        mask_channels = events['Channel'].isin(channels_events_select[event_type]) # select events only detected in the dict "channels_events_select" of event
        events_of_sel_chans = events[mask_channels] # keep events of set channels

        for stage in stages_events_select: # loop only on events detected in the list "stages_events_select"
            mask_stages = events_of_sel_chans['Stage_Letter'] == stage # mask on the stage
            events_of_chan_of_stage = events_of_sel_chans[mask_stages]  # keep event of set stage
            event_times = events_of_chan_of_stage[timestamps_labels[event_type]].values # get np array of events timings that summarize the whole events (set in params)
            phase_angles_rsp = get_phase_angles(rsp_features, event_times) # compute phase angles of event for each respi cycle
    
            save_dict(phase_angles_rsp, path = f'../events_coupling/{subject}_{event_type}_{stage}_phase_angles.txt') # save dict (rsp cycles * phase angles)




    

        

