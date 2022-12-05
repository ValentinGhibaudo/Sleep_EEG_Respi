import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import ghibtools as gh
from params import patients, dpis, channels_sigma_select, nb_point_by_cycle, transition_ratio, sigma_coupling_chan_means, fig_global_cycle_type

"""
This script generate 4 types of figs:
- 1 : The more detailed is a 8*4 subplots figure for each subject with phase frequency sigma power plotted for each chan and resp cycle type (all cycles, just cycles with at least one spindle found inside, just cycles without any spindle found inside, and difference between both)
- 2 : A less detailed fig that plots the selected resp cycle type mean phase-freq for each channel with all subjects in one fig
- 3 : A much less detailed fig that plots the select resp cycle type mean phase-freq average across some chosen channels for each subject
- 4 : The less detailed one that plot average across subject of the matrices of the 3rd figure
"""

da_global = None # prepare an xarray that will keep phase-freq maps from subjects for figures 2, 3 , 4

## 1st FIGURE - One by subject : 8*4 subplots phase-freq for the 8 channels and the 4 types of cycles (all, spindled, unspindled, diff)
for patient in patients:
    print(patient)
    phase_freqs = xr.open_dataarray(f'../sigma_coupling/{patient}_phase_freq_sigma.nc') # 4 D xarray : chan * cycle * freq * time
    cycles = phase_freqs.coords['cycle']
    n_cycles = cycles.size
    mask_spindled = [bool(encoding) for encoding in phase_freqs.attrs['cycle_tagging']]
    mask_unspindled = [~bool(encoding) for encoding in phase_freqs.attrs['cycle_tagging']]
    spindled_cycles = cycles[mask_spindled]
    n_cycles_spindled = spindled_cycles.size
    unspindled_cycles = cycles[mask_unspindled]
    n_cycles_unspindled = unspindled_cycles.size
    phase_freq_mean_cycle = phase_freqs.mean('cycle')
    phase_freq_spindled_mean_cycle = phase_freqs.sel(cycle = spindled_cycles).mean('cycle')
    phase_freq_unspindled_mean_cycle = phase_freqs.sel(cycle = unspindled_cycles).mean('cycle')
    phase_freq_spindle_related_sigma = phase_freq_spindled_mean_cycle -  phase_freq_unspindled_mean_cycle

 

    phase_freq_means = {'all':phase_freq_mean_cycle,'spindled':phase_freq_spindled_mean_cycle,'unspindled':phase_freq_unspindled_mean_cycle,'spindle_related_sigma':phase_freq_spindle_related_sigma}
    n_titles = {'all':n_cycles, 'spindled':n_cycles_spindled, 'unspindled':n_cycles_unspindled, 'spindle_related_sigma':'NA'}
    p = phase_freqs.coords['point']
    f = phase_freqs.coords['freq']

    if da_global is None:
        da_global = gh.init_da({'patient':patients, 'chan':channels_sigma_select, 'freq':f, 'point':p})
    
    if fig_global_cycle_type == 'all':
        phase_freq_global_mean_to_norm = phase_freq_mean_cycle.data
    elif fig_global_cycle_type == 'spindled':
        phase_freq_global_mean_to_norm = phase_freq_spindled_mean_cycle.data
    elif fig_global_cycle_type == 'spindle_related_sigma':
        phase_freq_global_mean_to_norm = phase_freq_spindle_related_sigma.data
    
    da_global.loc[patient, :, :,:] = phase_freq_global_mean_to_norm # keep selected data type for each subject

    fig, axs = plt.subplots(nrows = len(channels_sigma_select), ncols = len(phase_freq_means.keys()), sharex = True, sharey = True, constrained_layout = True, figsize = (20,20))
    fig.suptitle(patient, fontsize = 20)
    # fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
    for c, cycle_type in enumerate(phase_freq_means.keys()):

        vmin = phase_freq_means[cycle_type].quantile(0.02)
        vmax = phase_freq_means[cycle_type].quantile(0.98)

        for r, channel in enumerate(channels_sigma_select):
            ax = axs[r, c]
            data = phase_freq_means[cycle_type].sel(chan = channel).data
            N = n_titles[cycle_type]
            ax.pcolormesh(p, f, data, vmin=vmin, vmax=vmax)
            ax.axvline(x = nb_point_by_cycle*transition_ratio, color = 'r')
            if r == 0:
                ax.set_title(f'{cycle_type} - N = {N}')
            if c == 0:
                ax.set_ylabel(f'{channel} \n Freq [Hz]')
            if c == len(phase_freq_means.keys())-1:
                ax.set_xlabel('Phase')
                ax.set_xticks([0, 0, int(nb_point_by_cycle*transition_ratio), int(nb_point_by_cycle/2), int(nb_point_by_cycle)])
                ax.set_xticklabels([0, 0, 'i-e', 'Pi', '2*Pi'], rotation=0, fontsize=10)

    plt.savefig(f'../sigma_coupling_figures/{patient}_phase_freq.tif', bbox_inches = 'tight', dpi = dpis)
    plt.close()


## 2nd FIGURE - One by channel, 20 subplots by fig for the 20 subjects, with phase-freq corresponding phase-freq maps of the chosen cycle type
patients = np.array(patients).reshape(4,5)
nrows = patients.shape[0]
ncols = patients.shape[1]

def zscore(da):
    return (da - da.mean()) / da.std()

for channel in channels_sigma_select:
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (20,10), sharex = True, sharey = True)
    fig.suptitle(channel, fontsize = 20)
    scaled_da = zscore(da_global.loc[:,channel,:,:]) #scale across subjects for the channel
    vmin = scaled_da.quantile(0.02)
    vmax = scaled_da.quantile(0.98)
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r,c]
            patient = patients[r,c]
            data = scaled_da.loc[patient, :,:].data
            ax.pcolormesh(p, f, data, vmin=vmin, vmax=vmax)
            ax.axvline(x = nb_point_by_cycle*transition_ratio, color = 'r')
            ax.set_title(patient)
            if c == 0:
                ax.set_ylabel('Freq [Hz]')
            if r == nrows-1:
                ax.set_xlabel('Phase')
                ax.set_xticks([0, 0, int(nb_point_by_cycle*transition_ratio), int(nb_point_by_cycle/2), int(nb_point_by_cycle)])
                ax.set_xticklabels([0, 0,'i-e', 'Pi', '2*Pi'], rotation=0, fontsize=10)
    plt.savefig(f'../sigma_coupling_figures/{channel}_global_phase_freq.tif', bbox_inches = 'tight', dpi = dpis)
    plt.close()


## 3rd FIGURE - One figure is generated, presenting for each subject a mean of selected channels phase-freq maps pf the chosen cycle type
fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (20,10), sharex = True, sharey = True)
scaled_das = xr.concat([zscore(da_global.loc[patient,sigma_coupling_chan_means, :,:].mean('chan')) for patient in patients], dim = 'patient')
vmin = scaled_das.quantile(0.02)
vmax = scaled_das.quantile(0.98)
for r in range(nrows):
    for c in range(ncols):
        ax = axs[r,c]
        patient = patients[r,c]
        data = zscore(scaled_das.loc[patient, :,:]).data
        ax.pcolormesh(p, f, data, vmin=vmin, vmax=vmax)
        ax.axvline(x = nb_point_by_cycle*transition_ratio, color = 'r')
        ax.set_title(patient)
        if c == 0:
            ax.set_ylabel('Freq [Hz]')
        if r == nrows-1:
            ax.set_xlabel('Phase')
            ax.set_xticks([0, 0, int(nb_point_by_cycle*transition_ratio), int(nb_point_by_cycle/2), int(nb_point_by_cycle)])
            ax.set_xticklabels([0, 0,'i-e', 'Pi', '2*Pi'], rotation=0, fontsize=10)
plt.savefig(f'../sigma_coupling_figures/choice_phase_freq.tif', bbox_inches = 'tight', dpi = dpis)
plt.close()


## 4th FIGURE - Mean across subjects of the 3rd figure/matrices
fig, ax = plt.subplots(figsize = (15,7))
data = scaled_das.mean('patient').data
ax.pcolormesh(p, f, data)
ax.axvline(x = nb_point_by_cycle*transition_ratio, color = 'r')
ax.set_ylabel('Freq [Hz]')
ax.set_xlabel('Respiration phase')
ax.set_xticks([0, 0, int(nb_point_by_cycle*transition_ratio), int(nb_point_by_cycle/2), int(nb_point_by_cycle)])
ax.set_xticklabels([0, 0,'i-e', 'Pi', '2*Pi'], rotation=0, fontsize=10)
plt.savefig(f'../sigma_coupling_figures/mean_choice_phase_freq.tif', bbox_inches = 'tight', dpi = dpis)
plt.close()