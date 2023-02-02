import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from params import subjects, stages_events_select


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
    sns.heatmap(to_corr.corr('spearman'), annot = True, ax=ax, vmin = -1, vmax = 1, cmap = 'seismic')
    ax.set_title(f'{event_type} - N : {Ncorr}')
    return ax


for subject in subjects:
    print(subject)
    resp = pd.read_excel(f'../resp_features/{subject}_resp_features_tagged.xlsx', index_col = 0).reset_index(drop = True)
    resp_staged = resp[resp['sleep_stage'].isin(stages_events_select)]

    fig, axs = plt.subplots(ncols = 2, figsize = (30,10))
    fig.suptitle(subject, fontsize = 20, y = 1.05)

    for col, event_type in enumerate(['spindles','slowwaves']):
        events = pd.read_excel(f'../event_detection/{subject}_{event_type}_reref_yasa.xlsx', index_col = 0)
        ev_staged = events[events['Stage_Letter'].isin(stages_events_select)]
        ax = axs[col]
        corr_matrix_resp_events(resp_staged, ev_staged, event_type, ax=ax)

    plt.savefig(f'../events_stats/{subject}_corr_features', bbox_inches = 'tight')
    plt.close()