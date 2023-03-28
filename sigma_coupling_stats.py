import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
from params import *
from configuration import *
import jobtools
from sigma_coupling import sigma_coupling_job

def get_circ_features(angles, weights): # angles in radians and weights for each angle

    mu = pg.circ_mean(angles, weights) 
    mu = int(np.degrees(mu))
    r = round(pg.circ_r(angles, weights), 3)
    if mu < 0:
        mu = 360 + mu

    return mu, r

# JOB SIGMA COUPLING STATS
rows = []

def compute_stats_sigma_coupling(run_key, **p):

    phase_freqs = sigma_coupling_job.get(run_key)['sigma_coupling']
    
    rows = []
    for c_type in phase_freqs.coords['cycle_type'].values:
        for channel in phase_freqs.coords['chan'].values:
            sig = phase_freqs.sel(cycle = c_type, chan=channel).mean('freq')
            mi = (sig.max() - sig.min()) / sig.mean()
            angles = phase_freqs.coords['point'].values * 2 * np.pi
            weights = sig.values
            mu, r = get_circ_features(angles, weights)
            rows.append([run_key, channel, c_type,  float(mi), mu , r])

    stats = pd.DataFrame(rows, columns = ['subject','channel','cycle_type','Height Ratio','Mean Angle','Mean Vector Length'])
    return xr.Dataset(stats)

stats_sigma_coupling_job = jobtools.Job(precomputedir, 'stats_sigma_coupling', sigma_coupling_stats_params, compute_stats_sigma_coupling)
jobtools.register_job(stats_sigma_coupling_job)

def test_compute_stats_sigma_coupling():
    run_key = 'S2'
    sigma_coupling = compute_stats_sigma_coupling(run_key, **sigma_coupling_stats_params)['sigma_coupling']
    print(sigma_coupling)
            

# FIGS
def fig_polar_plot_sigma(run_keys, sigma_coupling_stats_params):
    p = sigma_coupling_stats_params
    concat = [stats_sigma_coupling_job(run_key).to_dataframe() for run_key in run_keys]
    sigma_coupling_stats = pd.concat(concat)
    sigma_coupling_stats.to_excel(base_folder / 'results' / 'sigma_coupling_stats' / 'sigma_modulation_stats.xlsx')

    nrows = 4
    ncols = 5
    subjects_array = np.array(run_keys).reshape(nrows, ncols)
    
    fig, axs = plt.subplots(nrows, ncols, subplot_kw = {'projection':'polar'}, constrained_layout = True, figsize = (20,20))
    df_chan = sigma_coupling_stats[sigma_coupling_stats['channel'] == p['chan_sel']].set_index(['subject','cycle_type'])
    hue_colors = {'all':'royalblue','N2':'forestgreen','N3':'pink','spindled':'red','unspindled':'black'}
    ymax = df_chan['Mean Vector Length'].max()

    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r,c]
            subject = subjects_array[r,c]

            for c_type in sigma_coupling_stats['cycle_type'].unique():
                angle = np.radians(df_chan.loc[(subject,c_type),'Mean Angle'])
                mvl = df_chan.loc[(subject,c_type),'Mean Vector Length']
                ax.arrow(angle, 0, 0, mvl, alpha = 0.6, width = 0.11, label = c_type, color=hue_colors[c_type], length_includes_head=False, head_width = 0.2, head_length =  0.001)
                ax.set_ylim(0,ymax)
            if r == 0 and c == 4:
                ax.legend(bbox_to_anchor=(1.1, 0.7))
            ax.set_title(subject)
    fig.savefig(base_folder / 'results' / 'sigma_coupling_stats' / 'polar_sigma.png', bbox_inches = 'tight')
    plt.close()
    return None

if __name__ == '__main__':
    # test_compute_stats_sigma_coupling()

    fig_polar_plot_sigma(run_keys, sigma_coupling_stats_params)

