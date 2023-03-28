import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from params import *
from configuration import base_folder
from rsp_detection import resp_tag_job
from detect_sleep_events import spindles_tag_job, slowwaves_tag_job


def corr_matrix_resp_events(resp, events, event_type, ax=None):
    resp_features_to_include = ['cycle_duration','inspi_duration','expi_duration','cycle_freq','cycle_ratio','inspi_amplitude','expi_amplitude','inspi_volume','expi_volume']
    if event_type == 'spindles':
        event_peak_label = 'Peak'
        event_features_to_include = ['Duration', 'Amplitude', 'RMS', 'AbsPower','RelPower', 'Frequency', 'Oscillations', 'Symmetry']
    elif event_type == 'slowwaves':
        event_peak_label = 'NegPeak'
        event_features_to_include = ['Duration','ValNegPeak', 'ValPosPeak', 'PTP', 'Slope', 'Frequency']
    rows = []
    for c, row in resp.iterrows():
        tmin = row['start_time']
        tmax = row['stop_time']
        mask_window = (events[event_peak_label] > tmin) & (events[event_peak_label] <= tmax)
        ev_in_window = events[mask_window]
        N_in_window = ev_in_window.shape[0]
        if not N_in_window == 0:
            resp_concat = row[resp_features_to_include]
            ev_concat = ev_in_window[event_features_to_include].mean()
            entire_row = pd.concat([resp_concat,ev_concat])
            entire_row['N_ev_in_window'] = N_in_window
            rows.append(entire_row.to_frame().T)

    to_corr = pd.concat(rows).astype(float)
    Ncorr = to_corr.shape[0]
    
    if ax is None:
        fig, ax = plt.subplots(figsize = (15,15))
    corr_matrix = to_corr.corr('spearman')
    sns.heatmap(corr_matrix, annot = True, ax=ax, vmin = -1, vmax = 1, cmap = 'seismic')
    ax.set_title(f'{event_type} - N : {Ncorr}')
    return corr_matrix, ax


corr_matrices_dict = {'spindles':[], 'slowwaves':[]}

columns = {}
indexes = {}

for subject in run_keys:
    resp = resp_tag_job.get(subject).to_dataframe()
    resp_staged = resp[resp['sleep_stage']== compute_stage]

    fig, axs = plt.subplots(ncols = 2, figsize = (30,10))
    fig.suptitle(subject, fontsize = 20, y = 1.05)

    for col, event_type in enumerate(['spindles','slowwaves']):
        if event_type == 'spindles':
            events = spindles_tag_job.get(subject).to_dataframe()
        elif event_type == 'slowwaves':
            events = slowwaves_tag_job.get(subject).to_dataframe()

        ax = axs[col]
        corr_matrix, ax = corr_matrix_resp_events(resp_staged, events, event_type, ax=ax)
        columns[event_type] = corr_matrix.columns
        indexes[event_type] = corr_matrix.index
        corr_matrices_dict[event_type].append(corr_matrix.values[:,:,np.newaxis])
    fig.savefig(base_folder / 'results' / 'events_stats' / f'{subject}_corr_features.png', bbox_inches = 'tight')
    plt.close()




fig, axs = plt.subplots(ncols = 2, figsize = (30,10))
fig.suptitle(f'Mean correlation matrices of {len(run_keys)} subjects', fontsize = 20, y = 1.05)

for col, event_type in enumerate(['spindles','slowwaves']):
    ax = axs[col]
    ax.set_title(event_type)
    corr_matrices_subjects = np.concatenate(corr_matrices_dict[event_type], axis = 2)
    corr_matrix_subjects = np.mean(corr_matrices_subjects, axis = 2)
    df_corr_matrix_subjects = pd.DataFrame(corr_matrix_subjects, columns = columns[event_type], index =indexes[event_type])
    sns.heatmap(df_corr_matrix_subjects, annot = True, ax=ax, vmin = -1, vmax = 1, cmap = 'seismic')

fig.savefig(base_folder / 'results' / 'events_stats' / 'all_corr_features.png', bbox_inches = 'tight')
plt.close()