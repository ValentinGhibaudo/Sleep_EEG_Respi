import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import ghibtools as gh
import pandas as pd
from params import subjects, dpis, channels_sigma_select, nb_point_by_cycle, transition_ratio, sigma_coupling_chan, fig_global_cycle_type

"""
This script generate 4 types of figs:
- 1 : The more detailed is a 8*4 subplots figure for each subject with phase frequency sigma power plotted for each chan and resp cycle type (all cycles, just cycles with at least one spindle found inside, just cycles without any spindle found inside, and difference between both)
- 2 : A less detailed fig that plots the selected resp cycle type mean phase-freq for each channel with all subjects in one fig
- 4 : The less detailed one that plot average across subject of the matrices of the 3rd figure
"""

mean_phase_freqs = xr.load_dataarray('../sigma_coupling/mean_phase_freq_sigma.nc')
points = mean_phase_freqs.coords['point'].values
freqs = mean_phase_freqs.coords['freq'].values
Ns = pd.read_excel('../sigma_coupling/n_cycles_by_type.xlsx', index_col = 0).set_index(['subject','cycle_type'])

cycle_types = mean_phase_freqs.coords['cycle_type'].values

## 1st FIGURE - One by subject : 6 subplots phase-freq for the 6 types of cycles (all, spindled, unspindled, N2,N3, diff)

for subject in subjects:
    
    fig, axs = plt.subplots(ncols = cycle_types.size, sharex = True, sharey = True, constrained_layout = True, figsize = (20,4))
    fig.suptitle(f'{subject} - {sigma_coupling_chan}', fontsize = 20)
    
    for col, cycle_type in enumerate(cycle_types):

        ax = axs[col]
        N = Ns.loc[(subject, cycle_type),'N']
        data = mean_phase_freqs.loc[subject, cycle_type,:,:].data
        im = ax.pcolormesh(points, freqs, data)
        ax.axvline(x = transition_ratio, color = 'r')
        ax.set_title(f'{subject} - {cycle_type} - N = {N}')
        if col == 0:
            ax.set_ylabel('Freq [Hz]')
        ax.set_xlabel('Phase')
        ax.set_xticks([0, 0, transition_ratio, 1])
        ax.set_xticklabels([0, 0, 'inspi-expi', '2*Pi'], rotation=45, fontsize=10)
        if col == cycle_types.size - 1:
            plt.colorbar(im, ax = axs[col], label = 'Power in µV**2')
        else:
            plt.colorbar(im, ax = axs[col])

    plt.savefig(f'../sigma_coupling_figures/{subject}_phase_freq', bbox_inches = 'tight', dpi = dpis)
    plt.close()


## 2nd FIGURE - 20 subplots for the 20 subjects, with phase-freq corresponding phase-freq maps of the chosen cycle type
subject_array = np.array(subjects).reshape(4,5)
nrows = subject_array.shape[0]
ncols = subject_array.shape[1]


fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (20,10), sharex = True, sharey = True, constrained_layout = True)
fig.suptitle(f'{sigma_coupling_chan} sigma power during respiration cycle by subject ({fig_global_cycle_type} cycles mode)', fontsize = 20, y = 1.05)
for r in range(nrows):
    for c in range(ncols):
        ax = axs[r,c]
        subject = subject_array[r,c]
        data = mean_phase_freqs.loc[subject, fig_global_cycle_type,:,:].data

        im = ax.pcolormesh(points, freqs, data)
        ax.axvline(x = transition_ratio, color = 'r')
        ax.set_title(subject)
        if c == 0:
            ax.set_ylabel('Freq [Hz]')
        if r == nrows-1:
            ax.set_xlabel('Phase')
            ax.set_xticks([0, 0, transition_ratio, 1])
            ax.set_xticklabels([0, 0, 'inspi-expi', '2*Pi'], rotation=45, fontsize=10)
        if c == ncols - 1:
            plt.colorbar(im, ax = axs[r,c], label = 'Power in µV**2')
        else:
            plt.colorbar(im, ax = axs[r,c])
            
plt.savefig(f'../sigma_coupling_figures/mean_phase_freq_subjects_detailed', bbox_inches = 'tight', dpi = dpis)
plt.close()


# ## 3th FIGURE - Mean across subjects of the phase-freq matrix of set cycles types and channel
def zscore(da):
    return (da - da.mean()) / da.std()

mean_phase_freqs_zscored = gh.init_da({'subject':subjects, 'freq':freqs, 'point':points})
for subject in subjects:
    mean_phase_freqs_zscored.loc[subject,: ,: ] = zscore(mean_phase_freqs.sel(cycle_type = fig_global_cycle_type, subject = subject))

fig, ax = plt.subplots(figsize = (15,7))

data = mean_phase_freqs_zscored.mean('subject').data
im = ax.pcolormesh(points, freqs, data)
ax.axvline(x = transition_ratio, color = 'r')
ax.set_ylabel('Freq [Hz]')
ax.set_xlabel('Respiration phase')
ax.set_xticks([0, 0, transition_ratio, 1])
ax.set_xticklabels([0, 0, 'inspi-expi', '2*Pi'], rotation=45, fontsize=10)
plt.colorbar(im, ax = ax, label = 'Power in A.U.')
plt.savefig(f'../sigma_coupling_figures/mean_phase_freq_across_subjects', bbox_inches = 'tight', dpi = dpis)
plt.close()