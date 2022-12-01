import numpy as np
import pandas as pd
import json
from params import patients, timestamps_labels, channels_events_select, stages_events_select

######

def load_resp_features(patient):
    rsp = pd.read_excel(f'../resp_features/{patient}_resp_features_tagged.xlsx', index_col = 0)
    return rsp

def load_events(patient, event_type):
    if event_type == 'sp':
        return pd.read_excel(f'../event_detection/{patient}_spindles.xlsx', index_col = 0)
    elif event_type == 'sw':
        return pd.read_excel(f'../event_detection/{patient}_slowwaves.xlsx', index_col = 0)

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

#####

save = True

for patient in patients: # loop on run keys
    print(patient)
    rsp_features = load_resp_features(patient) # load rsp features

    for event_type in ['sp','sw']: # sp = spindles ; sw = slowwaves
        events = load_events(patient, event_type) # load sp or sw
        mask_channels = events['Channel'].isin(channels_events_select) # select events only detected in the list "channels_events_select"
        mask_stages = events['Stage_Letter'].isin(stages_events_select) # select events only detected in the list "stages_events_select"
        whole_mask = mask_channels & mask_stages # unify the two previous masks (chans & stages)
        events = events[whole_mask] # keep events of set channels and stages
        event_times = events[timestamps_labels[event_type]].values # get np array of events timings that summarize the whole events (set in params)
        phase_angles_rsp = get_phase_angles(rsp_features, event_times) # compute phase angles of event for each respi cycle
   
        if save:
            save_dict(phase_angles_rsp, path = f'../events_coupling/{patient}_{event_type}_phase_angles.txt') # save dict (rsp cycles * phase angles)




    

        

