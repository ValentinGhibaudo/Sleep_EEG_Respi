import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
from params import subjects

def get_circ_features(angles, weights): # angles in radians and weights for each angle

    mu = pg.circ_mean(angles, weights) 
    mu = int(np.degrees(mu))
    r = round(pg.circ_r(angles, weights), 3)
    if mu < 0:
        mu = 360 + mu

    return mu, r

rows=[]

for subject in subjects:
    phase_freqs = xr.open_dataarray(f'../sigma_coupling/{subject}_phase_freq_sigma.nc').dropna('cycle')
    resp_features_stretched = pd.read_excel(f'../resp_features/{subject}_resp_features_tagged_stretch.xlsx', index_col = 0) 
    
    c_spindled = resp_features_stretched[resp_features_stretched['Spindle_Tag'] == 1].index.to_numpy()
    c_unspindled = resp_features_stretched[resp_features_stretched['Spindle_Tag'] == 0].index.to_numpy()
    c_N2 = resp_features_stretched[resp_features_stretched['sleep_stage'] == 'N2'].index.to_numpy()
    c_N3 = resp_features_stretched[resp_features_stretched['sleep_stage'] == 'N3'].index.to_numpy()
    cycles = phase_freqs.coords['cycle'].values
    
    for c_label, c_type in zip(['all','N2','N3','spindled','unspindled'],[cycles, c_N2, c_N3, c_spindled, c_unspindled]):
        for channel in phase_freqs.coords['chan'].values:
            sig = phase_freqs.sel(cycle = c_type, chan=channel).mean(['cycle','freq'])
            mi = (sig.max() - sig.min()) / sig.mean()
            angles = phase_freqs.coords['point'].values * 2 * np.pi
            weights = sig.values
            mu, r = get_circ_features(angles, weights)
            rows.append([subject, channel, c_label,  float(mi), mu , r])
            
sigma_coupling_stats = pd.DataFrame(rows, columns = ['subject','channel','cycle_type','Height Ratio','Mean Angle','Mean Vector Length'])
sigma_coupling_stats.to_excel('../sigma_coupling_stats/sigma_modulation_stats.xlsx')



nrows = 4
ncols = 5
subjects_array = np.array(subjects).reshape(nrows,ncols)

for metric in ['Height Ratio','Mean Angle','Mean Vector Length']:
    fig, axs = plt.subplots(nrows, ncols, figsize = (20,15), constrained_layout = True)
    fig.suptitle(metric, fontsize = 20, y = 1.05)
    
    ymin = sigma_coupling_stats[metric].min()
    ymax = sigma_coupling_stats[metric].max()
    
    for r in range(nrows):
        for c in range(ncols):
            sub = subjects_array[r,c]
            ax = axs[r,c]
            data_plot = sigma_coupling_stats[sigma_coupling_stats['subject'] == sub]
            sns.pointplot(data=data_plot, x='channel', y = metric, hue = 'cycle_type', ax=ax)
            ax.set_ylim(ymin, ymax)
            ax.set_title(sub)
    plt.savefig(f'../sigma_coupling_stats/pointplot_modulation_detailed_{metric}', bbox_inches = 'tight')
    plt.close()
            
            

            
            
            
fig, axs = plt.subplots(ncols = 3, figsize = (15,5), constrained_layout = True)
for col, metric in enumerate(['Height Ratio','Mean Angle','Mean Vector Length']):
    ax = axs[col]
    sns.pointplot(data=sigma_coupling_stats, x = 'channel', y = metric, hue = 'cycle_type', ax=ax)
    ax.set_title(f'{metric} of sigma by respiration phase')
plt.savefig('../sigma_coupling_stats/pointplot_modulation_global')
plt.close()






fig, axs = plt.subplots(nrows, ncols, subplot_kw = {'projection':'polar'}, constrained_layout = True, figsize = (20,20))
df_Cz = sigma_coupling_stats[sigma_coupling_stats['channel'] == 'Cz'].set_index(['subject','cycle_type'])
hue_colors = {'all':'royalblue','N2':'forestgreen','N3':'pink','spindled':'red','unspindled':'black'}
ymax = df_Cz['Mean Vector Length'].max()

for r in range(nrows):
    for c in range(ncols):
        ax = axs[r,c]
        subject = subjects_array[r,c]

        for c_type in sigma_coupling_stats['cycle_type'].unique():
            angle = np.radians(df_Cz.loc[(subject,c_type),'Mean Angle'])
            mvl = df_Cz.loc[(subject,c_type),'Mean Vector Length']
            ax.arrow(angle, 0, 0, mvl, alpha = 0.6, width = 0.11, label = c_type, color=hue_colors[c_type], length_includes_head=False, head_width = 0.2, head_length =  0.001)
            ax.set_ylim(0,ymax)
        if r == 0 and c == 4:
            ax.legend(bbox_to_anchor=(1.1, 0.7))
        ax.set_title(subject)
plt.savefig('../sigma_coupling_stats/polar_sigma')
plt.close()

        
        

