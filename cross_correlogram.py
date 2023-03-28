import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from params import *
import jobtools
from rsp_detection import resp_tag_job
from detect_sleep_events import spindles_tag_job, slowwaves_tag_job
from configuration import *

p = cross_correlogram_params
stage = p['stage']
chan = p['chan']
sp_time_label = p['timestamps_labels']['spindles']
sw_time_label = p['timestamps_labels']['slowwaves']

# CROSS-CORRELOGRAM SPINDLES VS SLOWWAVES
def crosscorrelogram(a,b):
    """
    Compute combinatorial difference between a vs b (a - b with all possibilities)
    
    ------------------
    INPUTS :
    a : 1D numpy vector
    b : 1D numpy vector
    
    OUTPUT :
    c : crosscorrelogram vector of shape (a.size*b.size,)
    
    """
    c = a[:, np.newaxis] - b[np.newaxis, :]
    return c.reshape(-1)



# FIG 1 : SPINDLES vs SLOWWAVES
delta = p['delta_spsw']
delta_t_by_bin = p['delta_t_by_bin_spsw']
nbins = int(delta * 2 / delta_t_by_bin)

nrows = 4
ncols = 5
subjects_array = np.array(run_keys).reshape(nrows, ncols)

fig, axs = plt.subplots(nrows, ncols, figsize = (20,15), constrained_layout = True)
fig.suptitle(f'Cross-correlogram of spindles {sp_time_label} - slowwaves {sw_time_label} in {chan}', fontsize = 20, y = 1.04)

for r in range(nrows):
    for c in range(ncols):
        ax = axs[r,c]
        subject = subjects_array[r,c]

        spindles = spindles_tag_job.get(subject).to_dataframe()
        slowwaves = slowwaves_tag_job.get(subject).to_dataframe()

        sp_times = spindles[(spindles['Channel'] == chan) & (spindles['Stage_Letter'] == stage)][sp_time_label].values
        sw_times = slowwaves[(slowwaves['Channel'] == chan) & (slowwaves['Stage_Letter'] == stage)][sw_time_label].values

        cross = crosscorrelogram(sp_times, sw_times)
        cross_sel = cross[(cross < delta) & (cross > -delta)]
        N = cross_sel.size
        ax.set_title(f'{subject} - N : {N}')
        ax.hist(cross_sel, bins = nbins, align = 'mid')
        ax.set_xlim(-delta,delta)
        
fig.savefig(base_folder / 'results' / 'cross_correlogram' / 'cross_correlogram_spindles_slowwaves.png', bbox_inches = 'tight')
plt.close()
    
    
# FIG 2 : EVENTS - RESP
delta = p['delta_resp']
delta_t_by_bin = p['delta_t_by_bin_resp']
nbins = int(delta * 2 / delta_t_by_bin)

resp_transition_dict = {'expi-inspi':'start_time','inspi-expi':'transition_time'}

event_labels = ['spindles','slowwaves']
for ev in event_labels:
    for resp_transition_label, resp_transition_time_label in resp_transition_dict.items():
        fig, axs = plt.subplots(nrows, ncols, figsize = (20,15), constrained_layout = True)
        event_time_label = p['timestamps_labels'][ev]
        fig.suptitle(f'Cross-correlogram of {ev} {event_time_label} - resp {resp_transition_label} transition in {chan}', fontsize = 20, y = 1.04)

        for r in range(nrows):
            for c in range(ncols):
                ax = axs[r,c]
                subject = subjects_array[r,c]

                resp = resp_tag_job.get(subject).to_dataframe()
                
                if ev == 'spindles':
                    events = spindles_tag_job.get(subject).to_dataframe()
                elif ev == 'slowwaves':
                    events = slowwaves_tag_job.get(subject).to_dataframe()


                ev_sel = events[(events['Channel'] == chan)]
                resp_stage = resp[resp['sleep_stage'] == stage]

                cross = crosscorrelogram(ev_sel[p['timestamps_labels'][ev]].values, resp_stage[resp_transition_time_label].values)
                cross_sel = cross[(cross < delta) & (cross > -delta)]
                N = cross_sel.size

                ax.set_title(f'{subject} - N : {N}')
                ax.hist(cross_sel, bins = nbins, align = 'mid')
                ax.set_xlim(-delta,delta)
            
        fig.savefig(base_folder / 'results' / 'cross_correlogram' / f'cross_correlogram_{ev}_{resp_transition_label}', bbox_inches = 'tight')
        plt.close()