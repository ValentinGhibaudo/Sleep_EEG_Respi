import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from fnmatch import filter
import xarray as xr
from params import *
from configuration import *
import jobtools
from events_coupling import event_coupling_job
from rsp_detection import resp_tag_job



"""
This script generate 4 types of figs:

"""

p = events_coupling_figs_params
stage = p['stage']
save_folder = base_folder / 'results' / 'events_coupling_figures'
subjects = run_keys

#####

def Kullback_Leibler_Distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def Modulation_Index(distrib):
    distrib = np.asarray(distrib, dtype = float)
        
    N = distrib.size
    uniform_distrib = np.ones(N) * (1/N)
    mi = Kullback_Leibler_Distance(distrib, uniform_distrib) / np.log(N)

    return mi

def get_circ_features(angles, bins = 18): # angles in radians
    
    z, pval = pg.circ_rayleigh(angles)
    count, bins = np.histogram(angles, bins = bins)
    tort_mi = Modulation_Index(count / sum(count))
    tort_significance = '*' if tort_mi >= 0.005 else 'ns'
    
    mu = pg.circ_mean(angles) #+ np.pi
    mu = int(np.degrees(mu))
    r = round(pg.circ_r(angles), 3)
    
    if mu < 0:
        mu = 360 + mu

    return pval, mu, r, tort_mi, tort_significance

def pval_stars(p):
    if not np.isnan(p):
        if p >= 0.05:
            stars = 'ns'
        elif p < 0.05 and p >= 0.01:
            stars = '*'
        elif p < 0.01 and p >= 0.001:
            stars = '**'
        elif p < 0.001:
            stars = '***'
    else:
        stars = np.nan
    return stars

def get_angles(ds, pattern):
    coords = list(ds.coords)
    filtered_coords = filter(coords, pattern)
    angles = []
    for coord in filtered_coords:
        angles.extend(list(ds[coord].values))
    return angles

def load_grouped_angles(subject, event, cooccuring, speed, chan):
    
    """
    High level function that load angles according to arguments and concatenate them if '*' argument
    
    Parameters (str)
    ----------
    subject : From 'S1' to 'S20', '*' to concatenate all
    event : 'spindles' or 'slowwaves' for spindles or slow-wave 
    cooccuring : 'cooccur' or 'notcoocur', '*' to concatenate both
    speed : 'SS' or 'FS' for slow or fast spindles, '*' to concatenate both (useful only for spindles)
    chan : 'Fz' for example
    """
    
    if subject == '*':
        concat = [event_coupling_job.get(run_key) for run_key in run_keys]
        ds_search = xr.concat(concat, dim = 'subject')
    else:
        ds_search = event_coupling_job.get(subject)

    if event == 'spindles':
        pattern = f'{subject}_spindles_{cooccuring}_{speed}_{chan}'
    elif event == 'slowwaves':
        pattern = f'{subject}_slowwaves_{cooccuring}_{chan}'
    
    return np.array(get_angles(ds_search, pattern))

def circular_plot_angles(angles, color = None, ax=None, ratio_plot = 0.42, bins = 18, with_rticks = True, with_title = False, with_arrow = True):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection = 'polar'), constrained_layout = True)
    pval, mu , r, tort_mi, tort_significance = get_circ_features(angles)
    stars = pval_stars(pval)
        
    N_events = angles.size # number of angles computed and distributed = number of events detected and computed (all subjects pooled)
    values, bins_hist, patches = ax.hist(angles, bins = bins, density = True, edgecolor = 'black', color = color ) # polar histogram of distribution of angles of all subjects (in radians)
    rmax = np.max(values)
    max_rticks = rmax + 0.05 * rmax
        
    if with_rticks:
        ax.set_rticks(np.arange(0,max_rticks,0.05))
        ax.set_rmax(max_rticks)
    else:
        ax.set_rticks([])
        
    tick = [max_rticks * 0.995, max_rticks * 0.999] # will display some markers at the exterior part of the polarplot (indicating respi phase)
    tick_pos = ratio_plot * 360 # angle where ticks will be colored differently = at the inspi to expi transition
    for t in np.deg2rad(np.arange(0, 360, 2)): # loop ticks displaying around the polar circle
        if t <= np.deg2rad(tick_pos) and t >= 0: # if angle < resp transition, color is red
            color = 'r'
        elif t >= np.deg2rad(tick_pos) and t <= np.deg2rad(360): # if angle > resp transition, color is black
            color = 'k'
        ax.plot([t, t], tick, lw=7, color=color, alpha = 1) # plot the ticks, lw = linewidth = width of each tick
    ax.set_xticks(np.deg2rad([0, 90 , tick_pos , 180 , 270])) # at this angles in degrees, ticks labels will be present
    ax.set_xticklabels(['Start', '90°', 'I>E', '180°','270°']) # labelize polar plot angles
    if with_title :
        ax.set_title(f'N : {N_events} \n Mean Angle : {mu}° - MVL : {r} - p : {stars} \n Tort MI : {round(tort_mi, 5)} ({tort_significance})')
    if with_arrow:
        if N_events > 1000 and tort_mi > 0.005:
            color_arrow = 'darkorange'
        else:
            color_arrow = 'red'
        ax.arrow(np.deg2rad(mu), 0, 0, r, alpha = 1, width = 0.3, label = 'r', color=color_arrow, length_includes_head=True, head_width = 0.4, head_length =  0.01)
    return ax

def get_cycles_ratios(run_keys):
    concat = []
    for run_key in run_keys:
        to_concat = resp_tag_job.get(run_key).to_dataframe()
        to_concat['subject'] = run_key
        concat.append(to_concat)
    pooled_features = pd.concat(concat)

    cycle_ratio_by_sub_by_stage = pooled_features.groupby(['subject','sleep_stage'])['cycle_ratio'].mean(numeric_only = True).reset_index()
    mean_cycle_ratio_by_stage = pooled_features.groupby('sleep_stage')['cycle_ratio'].mean(numeric_only = True).reset_index()
    mean_cycle_ratio_by_stage.insert(0, 'subject','mean')
    cycles_ratios = pd.concat([cycle_ratio_by_sub_by_stage, mean_cycle_ratio_by_stage])
    return cycles_ratios

def get_respi_ratio(subject , stage, ratio_df):
    if subject == '*':
        subject_ratio = 'mean'
    else:
        subject_ratio  = subject

    if stage == '*':
        ratio = ratio_df.set_index(['subject','sleep_stage']).loc[(subject_ratio , ['N2','N3']), 'cycle_ratio'].mean()
    else:
        ratio = ratio_df.set_index(['subject','sleep_stage']).loc[(subject_ratio , stage), 'cycle_ratio']  
    return ratio

##### 

event_types = ['spindles','slowwaves'] # run keys for spindles and slow waves
event_types_titles = {'spindles':'Spindles','slowwaves':'Slow-Waves'} # understandable labels for spindles and slowwaves
bins = 18 # histograms of distribution of events according to resp phase will be distributed in this number of bins

cycles_ratios = get_cycles_ratios(run_keys) # get cycles ratio for the phase transition in polar plots

dict_figure = {
    'spindles':{'SS':{'pos':0, 'color':None},'FS':{'pos':1, 'color':'skyblue'}},
    'slowwaves':{'pos':2, 'color':'forestgreen'},
}

# POOLED
print('FIG POOLED')

ncols = 3

for chan in channels_events_select:
    fig, axs = plt.subplots(ncols, figsize = (15,7), constrained_layout = True, subplot_kw=dict(projection = 'polar'))

    for ev in ['spindles','slowwaves']:
        ev_title = event_types_titles[ev]
            
        ratio = get_respi_ratio(subject = '*', stage = stage, ratio_df = cycles_ratios)

        if ev == 'spindles':
            for speed in ['SS','FS']:
                angles = load_grouped_angles(subject = '*' , event = ev, cooccuring = '*', speed = speed, chan = chan)
                
                if angles.size == 0:
                    continue

                pos = dict_figure[ev][speed]['pos']
                color = dict_figure[ev][speed]['color']
                ax = axs[pos]
                circular_plot_angles(angles, color=color, ax=ax, ratio_plot = ratio, with_title = True, with_arrow = True, with_rticks = True)
                title = f'{ev} - {speed} \n' + ax.get_title()
                ax.set_title(title, fontsize = 15, y = 1.1)  
                    
        elif ev == 'slowwaves':

            angles = load_grouped_angles(subject = '*' , event = ev,cooccuring = '*', speed = speed, chan = chan)
            
            if angles.size == 0:
                continue
                
            pos = dict_figure[ev]['pos']

            ax = axs[pos]
            color = dict_figure[ev]['color']
            circular_plot_angles(angles, color=color, ax=ax, ratio_plot = ratio, with_title = True, with_arrow = True, with_rticks = True)
            title = f'{ev} \n' + ax.get_title()
            ax.set_title(title, fontsize = 15, y = 1.1)

    fig.savefig(save_folder / 'global' / f'polar_plot_pooled_{chan}.png', bbox_inches = 'tight')
    plt.close()
    
  
    
# SUBJECT
print('FIG by SUBJECT')
colors = {'spindles':None , 'slowwaves':'forestgreen'}
for chan in channels_events_select:
    for event_type in ['spindles','slowwaves']:

        color = colors[event_type]

        nrows = 4
        ncols = 5

        subjects_array = np.array(run_keys).reshape(nrows, ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize = (20,20), constrained_layout = True, subplot_kw=dict(projection = 'polar'))
        fig.suptitle(f'{event_type} polar distributions along respiration phase')
        
        for r in range(nrows):
            for c in range(ncols):
                ax = axs[r,c]
                subject = subjects_array[r,c]

                ratio = get_respi_ratio(subject = subject, stage = stage, ratio_df = cycles_ratios)

                angles = load_grouped_angles(subject = subject , event = ev, cooccuring = '*', speed = '*', chan = chan)
                
                if angles.size == 0:
                    continue 
                    
                circular_plot_angles(angles, color=color, ax=ax, ratio_plot = ratio, with_title = False, with_arrow = True, with_rticks = True)
                title = subject
                ax.set_title(title)  

        fig.savefig(save_folder / 'subjects' / f'polar_plot_{chan}_{event_type}.png', bbox_inches = 'tight')
        plt.close()
        
print('SUCCESS')