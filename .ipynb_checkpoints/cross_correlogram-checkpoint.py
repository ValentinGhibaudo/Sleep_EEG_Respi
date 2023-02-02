import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from params import subjects, timestamps_labels


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


chans = ['Fp2','Fp1','Fz','C4','C3','Cz','Pz']
stages = ['N2','N3']




delta = 1.5
delta_t_by_bin = 0.05
nbins = int(delta * 2 / delta_t_by_bin)

for subject in subjects:
    print(subject)
    spindles = pd.read_excel(f'../event_detection/{subject}_spindles_reref_yasa.xlsx', index_col = 0)
    slowwaves = pd.read_excel(f'../event_detection/{subject}_slowwaves_reref_yasa.xlsx', index_col = 0)

    chans = [chan for chan in spindles['Channel'].unique() if not chan in ['T4','T3','O1','O2']]
    stages = ['N2','N3']

    nrows = len(stages)
    ncols = len(chans)

    fig, axs = plt.subplots(nrows, ncols, figsize = (20,5), constrained_layout = True)
    fig.suptitle(f'{subject} : sp - sw', fontsize = 20)

    for c, ch in enumerate(chans):
        for r, stage in enumerate(stages):

            sp = spindles[(spindles['Channel'] == ch) & (spindles['Stage_Letter'] == stage)]
            sw = slowwaves[(slowwaves['Channel'] == ch) & (slowwaves['Stage_Letter'] == stage)]
            cross = crosscorrelogram(sp[timestamps_labels['sp']].values, sw[timestamps_labels['sw']].values)
            cross_sel = cross[(cross < delta) & (cross > -delta)]
            N = cross_sel.size

            ax = axs[r,c]
            ax.set_title(f'{ch} - {stage} - N : {N}')
            ax.hist(cross_sel, bins = nbins, align = 'mid')
            ax.set_xlim(-delta,delta)
    
    plt.savefig(f'../cross_correlogram/{subject}_cross_correlogram_sp_sw', bbox_inches = 'tight')
    plt.close()
    
    
    
    
    
delta = 4
delta_t_by_bin = 0.2
nbins = int(delta * 2 / delta_t_by_bin)

peak_labels = {'spindles':timestamps_labels['sp'],'slowwaves':timestamps_labels['sw']}
resp_transition_label = {'ei':'start_time','ie':'transition_time'}

for subject in subjects:
    print(subject)

    resp = pd.read_excel(f'../resp_features/{subject}_resp_features_tagged.xlsx', index_col = 0)

    nrows = len(stages)
    ncols = len(chans)
    
    for resp_transition in ['ei','ie']:
    
        for ev in ['spindles','slowwaves']:
            events = pd.read_excel(f'../event_detection/{subject}_{ev}_reref_yasa.xlsx', index_col = 0)

            fig, axs = plt.subplots(nrows, ncols, figsize = (20,5), constrained_layout = True)
            fig.suptitle(f'{subject} - {ev} - {peak_labels[ev]} vs resp {resp_transition}', fontsize = 20)

            for c, ch in enumerate(chans):
                for r, stage in enumerate(stages):

                    ev_sel = events[(events['Channel'] == ch) & (events['Stage_Letter'] == stage)]
                    resp_stage = resp[resp['sleep_stage'] == stage]

                    cross = crosscorrelogram(ev_sel[peak_labels[ev]].values, resp_stage[resp_transition_label[resp_transition]].values)
                    cross_sel = cross[(cross < delta) & (cross > -delta)]
                    N = cross_sel.size

                    ax = axs[r,c]
                    ax.set_title(f'{ch} - {stage} - N : {N}')
                    ax.hist(cross_sel, bins = nbins, align = 'mid')
                    ax.set_xlim(-delta,delta)
            
            plt.savefig(f'../cross_correlogram/{subject}_cross_correlogram_{resp_transition}_{ev}', bbox_inches = 'tight')
            plt.close()