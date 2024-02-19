import numpy as np
import xarray as xr
import pandas as pd
from params import *
from configuration import *
from detect_sleep_events import spindles_tag_job, slowwaves_tag_job
from rsp_detection import resp_tag_job
import jobtools


# JOB EVENTS COUPLING TO RESPI CYCLE

def events_to_resp_coupling(run_key, **p):

    events_dict = {}
    events_dict['Spindles'] = spindles_tag_job.get(run_key).to_dataframe()
    events_dict['SlowWaves'] = slowwaves_tag_job.get(run_key).to_dataframe()
    rsp_features = resp_tag_job.get(run_key).to_dataframe()
    ts_labels = p['timestamps_labels']

    concat = []

    for event_label in ['Spindles','SlowWaves']:
        ts_label = ts_labels[event_label]
        events = events_dict[event_label] # get events df
        events_angles = events.copy()
        events_angles['Participant'] = run_key
        events_angles['Resp_Angle'] = np.nan
        events_angles['Event_type'] = event_label
        if event_label == 'SlowWaves':
            events_angles['Sp_Speed'] = np.nan

        if event_label == 'Spindles': 
            rsp_features_with_the_event = rsp_features[rsp_features['Spindle_Tag'] == 1] # keep only rsp cycles with spindles inside
        elif event_label == 'SlowWaves':
            rsp_features_with_the_event = rsp_features[rsp_features['SlowWave_Tag'] == 1] # keep only rsp cycles with slowwaves inside

        for i, row in rsp_features_with_the_event.iterrows():
            mask_ev_in_cycle = (events[ts_label] >= row['start_time']) & (events[ts_label] < row['stop_time']) # mask events of the current respiratory cycle time window
            ev_in_cycle = events[mask_ev_in_cycle] # keep events of the current respiratory cycle time window by applying the mask
            events_angles.loc[ev_in_cycle.index, 'Resp_Angle'] = ((ev_in_cycle[ts_label].values - row['start_time']) / row['cycle_duration']) * 2 * np.pi # Transform absolute time of event to relative time in the resp cycle

        concat.append(events_angles[['Participant','Event_type','Channel','Stage_Letter','night_quartile','cooccuring','Sp_Speed','Resp_Angle']])

    df_angles = pd.concat(concat)
    df_angles = df_angles[~df_angles['Resp_Angle'].isna()]
    return xr.Dataset(df_angles)

event_coupling_job = jobtools.Job(precomputedir, 'events_resp_coupling', events_coupling_params, events_to_resp_coupling)
jobtools.register_job(event_coupling_job)

def test_events_to_resp_coupling():
    run_key = 'S1'
    ds_coupling = events_to_resp_coupling(run_key, **events_coupling_params)
    print(ds_coupling.to_dataframe())

def concat_events_coupling(global_key, **p):
    concat = [event_coupling_job.get(run_key).to_dataframe() for run_key in run_keys]
    concat = pd.concat(concat).reset_index(drop = True)
    return xr.Dataset(concat)

def test_concat_events_coupling():
    print(concat_events_coupling('global_key', **concat_events_coupling_params).to_dataframe())

concat_events_coupling_job = jobtools.Job(precomputedir, 'concat_events_coupling', concat_events_coupling_params, concat_events_coupling)
jobtools.register_job(concat_events_coupling_job)




def compute_all():
    # jobtools.compute_job_list(event_coupling_job, run_keys, force_recompute=True, engine='loop')
    # jobtools.compute_job_list(event_coupling_job, run_keys, force_recompute=True, engine='joblib', n_jobs = 10)
    jobtools.compute_job_list(concat_events_coupling_job, [('global_key',)], force_recompute=True, engine='loop')

if __name__ == '__main__':
    # test_events_to_resp_coupling()
    # test_concat_events_coupling()

    compute_all()
