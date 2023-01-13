import numpy as np
import pandas as pd
import json
import xarray as xr
from params import subjects, timestamps_labels, channels_events_select, stages_events_select, encoder_events

###### USEFUL FUNCTIONS

def load_resp_features(subject):
    rsp = pd.read_excel(f'../resp_features/{subject}_resp_features_tagged.xlsx', index_col = 0)
    return rsp

def load_events(subject, event_type, encoder_events): # load events according to the yasa hypnogram labelling or human hypnogram labelling
    if event_type == 'sp':
        return pd.read_excel(f'../event_detection/{subject}_spindles_cooccuring.xlsx', index_col = 0)
    elif event_type == 'sw':
        return pd.read_excel(f'../event_detection/{subject}_slowwaves_cooccuring.xlsx', index_col = 0)

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
            list_of_angles.extend(phase_angles)
    return np.array(list_of_angles)

def save_dict(dictionnary, path):
    file = json.dumps(dictionnary) # create json object from dictionary
    with open(path, 'w',encoding='utf-8') as convert_file:
        convert_file.write(file)


##### COMPUTING

pooled_angles = {'sp':{'N2':{'SS':{'inslowwave':[],'outslowwave':[]},'FS':{'inslowwave':[],'outslowwave':[]}},
                       'N3':{'SS':{'inslowwave':[],'outslowwave':[]},'FS':{'inslowwave':[],'outslowwave':[]}},
                      },
                 'sw':{'N2':{'withSp':[],'NoSp':[]},'N3':{'withSp':[],'NoSp':[]}}
                }
                 # pooling angles from all subjects

for subject in subjects: # loop on run keys
    print(subject)
    rsp_features = load_resp_features(subject) # load rsp features
    
    for event_type in ['sp','sw']: # sp = spindles ; sw = slowwaves
        events = load_events(subject, event_type, encoder_events) # load sp or sw
        mask_channels = events['Channel'].isin(channels_events_select) # select events only detected in the dict "channels_events_select" of event
        events_of_sel_chans = events[mask_channels] # keep events of set channels
        
        if event_type == 'sp': 
            rsp_features_with_the_event = rsp_features[rsp_features['Spindle_Tag'] == 1] # keep only rsp cycles with spindles inside
        elif event_type == 'sw':
            rsp_features_with_the_event = rsp_features[rsp_features['SlowWave_Tag'] == 1] # keep only rsp cycles with slowwaves inside

        for stage in stages_events_select: # loop only on events detected in the list "stages_events_select"
            mask_stages = events_of_sel_chans['Stage_Letter'] == stage # mask on the stage
            events_of_chan_of_stage = events_of_sel_chans[mask_stages]  # keep event of set stage
            
            rsp_features_of_the_stage = rsp_features_with_the_event[rsp_features_with_the_event['sleep_stage'] == stage] # keep only respi cycles of the stage (and with events inside)
            
            if event_type == 'sp':
                
                for sp_speed in ['SS','FS']:
                    
                    mask_speed = events_of_chan_of_stage['Sp_Speed'] == sp_speed
                    spindles_speed = events_of_chan_of_stage[mask_speed]
                    
                    
                    for sp_cooccuring in [True, False]:
                        
                        spindles_cooccuring = spindles_speed[spindles_speed['in_slowwave'] == sp_cooccuring]
                        sp_times = spindles_cooccuring[timestamps_labels[event_type]].values
                        
                        if sp_cooccuring:
                            occuring_save_label = 'inslowwave'
                        else:
                            occuring_save_label = 'outslowwave'
                        
                        if sp_times.size != 0:
                            phase_angles_rsp = get_phase_angles(rsp_features_of_the_stage, sp_times)
                            np.save(f'../events_coupling/{subject}_{event_type}_{stage}_{occuring_save_label}_{sp_speed}_phase_angles.npy', phase_angles_rsp) # save angles from the subject & event type & stage as a .npy
                            
                            pooled_angles['sp'][stage][sp_speed][occuring_save_label].extend(phase_angles_rsp.tolist())
                        
            elif event_type == 'sw':
                
                for sp_inside in [True, False]:
                    
                    mask_occuring = events_of_chan_of_stage['sp_inside'] == sp_inside
                    sw_occuring = events_of_chan_of_stage[mask_occuring]
                    sw_times = sw_occuring[timestamps_labels[event_type]].values
                    
                    if sp_inside:
                        sp_inside_save_label = 'withSp'
                    else:
                        sp_inside_save_label = 'NoSp'
                    
                    phase_angles_rsp = get_phase_angles(rsp_features_of_the_stage, sw_times) # compute phase angles of event for each respi cycle
                    np.save(f'../events_coupling/{subject}_{event_type}_{stage}_{sp_inside_save_label}_phase_angles.npy', phase_angles_rsp) # save angles from the subject & event type & stage as a .npy
                    
                    pooled_angles['sw'][stage][sp_inside_save_label].extend(phase_angles_rsp.tolist())
                    
save_dict(pooled_angles, '../events_coupling/pooled_angles.txt')
            

            

            
            

            





    

        

