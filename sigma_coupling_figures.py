import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import jobtools
from configuration import *
from params import *
from sigma_coupling import sigma_coupling_job
from bibliotheque import init_da

"""
This script generate 4 types of figs:
- 1 : The more detailed is a 8*4 subplots figure for each subject with phase frequency sigma power plotted for each chan and resp cycle type (all cycles, just cycles with at least one spindle found inside, just cycles without any spindle found inside, and difference between both)
- 2 : A less detailed fig that plots the selected resp cycle type mean phase-freq for each channel with all subjects in one fig
- 4 : The less detailed one that plot average across subject of the matrices of the 3rd figure
"""

sigma_coupling_chan = sigma_coupling_figs_params['sigma_coupling_chan']
transition_ratio = sigma_coupling_figs_params['sigma_coupling_params']['transition_ratio']
fig_global_cycle_type = sigma_coupling_figs_params['fig_global_cycle_type']
save_article = sigma_coupling_figs_params['save_article']

if save_article:
    save_folder = article_folder 
    fig_format = '.tif'
    dpis = 150
else:
    save_folder = base_folder / 'results' / 'sigma_coupling_figures'
    fig_format = '.png'
    dpis = 100

concat_phase_freqs = []
concat_Ns = []

for run_key in run_keys:
    phase_freq_run_key = sigma_coupling_job.get(run_key)['sigma_coupling']
    df_n_run_key = pd.Series(data = phase_freq_run_key.attrs['data_n_cycle_averaged'],
                             index = phase_freq_run_key.coords['cycle_type'].values).to_frame().T
    df_n_run_key.insert(0, 'subject', run_key)
    # print(df_n_run_key)
    concat_Ns.append(df_n_run_key)
    concat_phase_freqs.append(phase_freq_run_key)


mean_phase_freqs = xr.concat(concat_phase_freqs, dim = 'subject').assign_coords({'subject':run_keys})
points = mean_phase_freqs.coords['point'].values
freqs = mean_phase_freqs.coords['freq'].values
Ns = pd.concat(concat_Ns).set_index('subject')


cycle_types = mean_phase_freqs.coords['cycle_type'].values

## 1st FIGURE - One by subject : 6 subplots phase-freq for the 6 types of cycles (all, spindled, unspindled, N2,N3, diff)

if not save_article:
    for subject in run_keys:

        fig, axs = plt.subplots(ncols = cycle_types.size, sharex = True, sharey = True, constrained_layout = True, figsize = (20,4))
        fig.suptitle(f'{subject} - {sigma_coupling_chan}', fontsize = 20)

        for col, cycle_type in enumerate(cycle_types):

            ax = axs[col]
            N = Ns.loc[subject,cycle_type]
            data = mean_phase_freqs.loc[subject, cycle_type,sigma_coupling_chan,:,:].data.T
            # print(points.shape, freqs.shape, data.shape)
            im = ax.pcolormesh(points, freqs, data)
            ax.axvline(x = transition_ratio, color = 'r')
            ax.set_title(f'{subject} - {cycle_type} - N = {N}')
            if col == 0:
                ax.set_ylabel('Freq [Hz]')
            ax.set_xlabel('Phase')
            ax.set_xticks([0, 0, transition_ratio, 1])
            ax.set_xticklabels([0, 0, 'inspi-expi', '2*Pi'], rotation=45, fontsize=10)
            # if col == cycle_types.size - 1:
            #     plt.colorbar(im, ax = axs[col], label = 'Power in µV**2')
            # else:
            #     plt.colorbar(im, ax = axs[col])

        fig.savefig(save_folder / f'{subject}_phase_freq{fig_format}', bbox_inches = 'tight', dpi = dpis)
        plt.close()



## 2nd FIGURE - 20 subplots for the 20 subjects, with phase-freq corresponding phase-freq maps of the chosen cycle type
subject_array = np.array(run_keys).reshape(4,5)
nrows = subject_array.shape[0]
ncols = subject_array.shape[1]


fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (20,10), sharex = True, sharey = True, constrained_layout = True)
if not save_article:
    fig.suptitle(f'{sigma_coupling_chan} sigma power during respiration cycle by subject ({fig_global_cycle_type} cycles mode)', fontsize = 20, y = 1.05)
for r in range(nrows):
    for c in range(ncols):
        ax = axs[r,c]
        subject = subject_array[r,c]
        data = mean_phase_freqs.loc[subject, fig_global_cycle_type,sigma_coupling_chan,:,:].data.T

        im = ax.pcolormesh(points, freqs, data)
        ax.axvline(x = transition_ratio, color = 'r')
        ax.set_title(subject)
        if c == 0:
            ax.set_ylabel('Freq [Hz]')
        if r == nrows-1:
            ax.set_xlabel('Phase')
            ax.set_xticks([0, 0, transition_ratio, 1])
            ax.set_xticklabels([0, 0, 'inspi-expi', '2*Pi'], rotation=45, fontsize=10)

        plt.colorbar(im, ax = ax, label = 'Power in µV**2')

fig.savefig(save_folder / f'mean_phase_freq_subjects_detailed{fig_format}', bbox_inches = 'tight', dpi = dpis)
plt.close()


# ## 3th FIGURE - Mean across subjects of the phase-freq matrix of set cycles types and channel
def zscore(da):
    return (da - da.mean()) / da.std()

mean_phase_freqs_zscored = init_da({'subject':run_keys, 'chan':channels_events_select, 'point':points, 'freq':freqs})

for subject in run_keys:
    for chan in channels_events_select:
        mean_phase_freqs_zscored.loc[subject, chan , : ,: ] = zscore(mean_phase_freqs.sel(cycle_type = fig_global_cycle_type, subject = subject, chan = chan))

nrows = 3
ncols = 4

fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (22,10), constrained_layout = False, sharex = False, sharey = False)

axs[2,3].remove()

delta = 0.1
vmin = mean_phase_freqs_zscored.quantile(delta)
vmax = mean_phase_freqs_zscored.quantile(1 - delta)

for ax, chan in zip(axs.flat, channels_events_select):

    data = mean_phase_freqs_zscored.sel(chan = chan).mean('subject').data.T
    im = ax.pcolormesh(points, freqs, data, vmin=vmin, vmax = vmax)
    ax.axvline(x = transition_ratio, color = 'r')

    ax.set_ylabel('Freq [Hz]')
    # ax.set_xlabel('Respiration phase')
    ax.set_xticks([ 0, transition_ratio, 1])
    ax.set_xticklabels([ 0, 'inspi-expi', '2*Pi'],fontsize=10)
    ax.set_title(chan)

ax_x_start = 1.02
ax_x_width = 0.01
ax_y_start = 0
ax_y_height = 1
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
clb.ax.set_title('Normalized power [AU]',fontsize=10)

fig.savefig(save_folder / f'mean_phase_freq_across_subjects{fig_format}', bbox_inches = 'tight', dpi = dpis)
plt.close()
