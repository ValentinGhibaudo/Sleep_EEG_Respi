import numpy as np
import xarray as xr
import pandas as pd
from params import *
from configuration import *
from detect_sleep_events import spindles_tag_job, slowwaves_tag_job
from rsp_detection import resp_tag_job
import jobtools


###### USEFUL FUNCTIONS

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


# JOB SPINDLES DETECTION

def events_to_resp_coupling(run_key, **p):

    events_dict = {}
    events_dict['spindles'] = spindles_tag_job.get(run_key).to_dataframe()
    events_dict['slowwaves'] = slowwaves_tag_job.get(run_key).to_dataframe()
    rsp_features = resp_tag_job.get(run_key).to_dataframe()

    ds = xr.Dataset()

    for chan in p['chans']:
        for event_label in ['spindles','slowwaves']:
            events = events_dict[event_label] # get events df
            mask_channels = events['Channel'] == chan # select events only detected in the channel
            events_of_chan = events[mask_channels] # keep events of set channels

            if event_label == 'spindles': 
                rsp_features_with_the_event = rsp_features[rsp_features['Spindle_Tag'] == 1] # keep only rsp cycles with spindles inside
            elif event_label == 'slowwaves':
                rsp_features_with_the_event = rsp_features[rsp_features['SlowWave_Tag'] == 1] # keep only rsp cycles with slowwaves inside

            rsp_features_of_the_stage = rsp_features_with_the_event[rsp_features_with_the_event['sleep_stage'] == p['stage']] # keep only respi cycles of the stage (and with events inside)

            if event_label == 'spindles':

                for sp_speed in ['SS','FS']:

                    mask_speed = events_of_chan['Sp_Speed'] == sp_speed
                    spindles_speed = events_of_chan[mask_speed]

                    for sp_cooccuring in [1, 0]:

                        spindles_cooccuring = spindles_speed[spindles_speed['cooccuring'] == sp_cooccuring]
                        
                        if sp_cooccuring:
                            occuring_save_label = 'cooccur'
                        else:
                            occuring_save_label = 'notcooccur'
                            
                        for q in ['q1','q2','q3','q4']:
                            sp_half = spindles_cooccuring[spindles_cooccuring['night_quartile'] == q]
                            
                            sp_times = sp_half[p['timestamps_labels'][event_label]].values

                            if sp_times.size != 0:
                                phase_angles_rsp = get_phase_angles(rsp_features_of_the_stage, sp_times)
                                ds[f'{run_key}_{event_label}_{occuring_save_label}_{sp_speed}_{q}_{chan}'] = phase_angles_rsp # store angles from the subject & event type in a dataset


            elif event_label == 'slowwaves':

                for sp_inside in [1, 0]:

                    mask_occuring = events_of_chan['cooccuring'] == sp_inside
                    sw_occuring = events_of_chan[mask_occuring]
                    
                    if sp_inside:
                        sp_inside_save_label = 'cooccur'
                    else:
                        sp_inside_save_label = 'notcooccur'
                        
                    for q in ['q1','q2','q3','q4']:
                        sw_half = sw_occuring[sw_occuring['night_quartile'] == q]
                        sw_times = sw_half[p['timestamps_labels'][event_label]].values
                        
                        if sw_times.size != 0:
                            phase_angles_rsp = get_phase_angles(rsp_features_of_the_stage, sw_times) # compute phase angles of event for each respi cycle
                            ds[f'{run_key}_{event_label}_{sp_inside_save_label}_{q}_{chan}'] = phase_angles_rsp # store angles from the subject & event type in a dataset

    return ds

event_coupling_job = jobtools.Job(precomputedir, 'events_resp_coupling', events_coupling_params, events_to_resp_coupling)
jobtools.register_job(event_coupling_job)

def test_events_to_resp_coupling():
    run_key = 'S1'
    ds_coupling = events_to_resp_coupling(run_key, **events_coupling_params)
    print(ds_coupling)



def compute_all():
    jobtools.compute_job_list(event_coupling_job, run_keys, force_recompute=False, engine='loop')

if __name__ == '__main__':
    # test_events_to_resp_coupling()

    compute_all()
