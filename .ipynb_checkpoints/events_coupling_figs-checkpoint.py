import numpy as np
import pandas as pd
import json
import pingouin as pg
import matplotlib.pyplot as plt
import glob
from params import subjects, dpis, stages_events_select, channels_events_select


"""
This script generate 4 types of figs:

"""

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

def load_grouped_angles(subject, event, stage, cooccuring, speed, chan):
    
    """
    High level function that load angles according to arguments and concatenate them if '*' argument
    
    Parameters (str)
    ----------
    subject : From 'S1' to 'S20', '*' to concatenate all
    event : 'sp' or 'sw' for spindles or slow-wave 
    stage : 'N2 or 'N3', '*' to concatenate both
    cooccuring : 'cooccur' or 'notcoocur', '*' to concatenate both
    speed : 'SS' or 'FS' for slow or fast spindles, '*' to concatenate both (useful only for spindles)
    chan : 'Fz' for example
    """
    
    if event == 'sp':
        files = glob.glob(f'../events_coupling/{subject}_sp_{stage}_{cooccuring}_{speed}_phase_angles_{chan}*')
    elif event == 'sw':
        files = glob.glob(f'../events_coupling/{subject}_sw_{stage}_{cooccuring}_phase_angles_{chan}*')
    
    return np.concatenate([np.load(f) for f in files])

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

event_types = ['sp','sw'] # run keys for spindles and slow waves
event_types_titles = {'sp':'Spindles','sw':'Slow-Waves'} # understandable labels for spindles and slowwaves
bins = 18 # histograms of distribution of events according to resp phase will be distributed in this number of bins

cycles_ratios = pd.read_excel('../resp_stats/cycle_ratios.xlsx', index_col = 0) # get cycles ratio for the phase transition in polar plots

dict_figure = {'sp':
               {'N2':
                {'SS':
                 {'cooccur':{'pos':(0,0), 'color':None},
                  'notcooccur':{'pos':(0,1), 'color':None}},
                 'FS':
                 {'cooccur':{'pos':(0,2), 'color':None},
                  'notcooccur':{'pos':(0,3), 'color':None}}},
                'N3':{
                    'SS':{
                        'cooccur':{'pos':(1,0), 'color':'skyblue'},
                        'notcooccur':{'pos':(1,1), 'color':'skyblue'}},
                    'FS':{
                        'cooccur':{'pos':(1,2), 'color':'skyblue'},
                        'notcooccur':{'pos':(1,3), 'color':'skyblue'}}},
                      },
                 'sw':{
                     'N2':{
                         'cooccur':{'pos':(2,0), 'color':'forestgreen'},
                         'notcooccur':{'pos':(2,1), 'color':'forestgreen'}},
                     'N3':{'cooccur':{'pos':(2,2), 'color':'limegreen'},
                           'notcooccur':{'pos':(2,3), 'color':'limegreen'}}}
                }


# GLOBAL DETAILED
print('FIG GLOBAL DETAILED')

nrows = 3
ncols = 4

for chan in channels_events_select:
    fig, axs = plt.subplots(nrows, ncols, figsize = (20,10), constrained_layout = True, subplot_kw=dict(projection = 'polar'))

    for ev in ['sp','sw']:
        ev_title = event_types_titles[ev]
        for stage in ['N2','N3']:
            
            ratio = get_respi_ratio(subject = '*', stage = stage, ratio_df = cycles_ratios)

            if ev == 'sp':
                for speed in ['SS','FS']:
                    for cooccur in ['notcooccur','cooccur']:
                        angles = load_grouped_angles(subject = '*' , event = ev, stage = stage,cooccuring = cooccur, speed = speed, chan = chan)
                        
                        if angles.size == 0:
                            print(chan ,ev , cooccur)
                            continue
                        pos = dict_figure[ev][stage][speed][cooccur]['pos']
                        color = dict_figure[ev][stage][speed][cooccur]['color']
                        ax = axs[pos[0], pos[1]]
                        circular_plot_angles(angles, color=color, ax=ax, ratio_plot = ratio, with_title = True, with_arrow = True, with_rticks = True)
                        title = f'{ev} - {stage} - {speed} - {cooccur} \n' + ax.get_title()
                        ax.set_title(title, fontsize = 15, y = 1.1)  
                        
            elif ev == 'sw':
                for cooccur in ['notcooccur','cooccur']:

                    angles = load_grouped_angles(subject = '*' , event = ev, stage = stage,cooccuring = cooccur, speed = speed, chan = chan)
                    
                    if angles.size == 0:
                        print(chan ,ev , cooccur)
                        continue
                        
                    pos = dict_figure[ev][stage][cooccur]['pos']

                    ax = axs[pos[0], pos[1]]
                    color = dict_figure[ev][stage][cooccur]['color']
                    circular_plot_angles(angles, color=color, ax=ax, ratio_plot = ratio, with_title = True, with_arrow = True, with_rticks = True)
                    title = f'{ev} - {stage} - {cooccur} \n' + ax.get_title()
                    ax.set_title(title, fontsize = 15, y = 1.1)

    plt.savefig(f'../events_coupling_stats/global/detailed/polar_plot_{chan}', bbox_inches = 'tight')
    plt.close()
    
    
    
# GLOBAL MERGE STAGE & SPEED
print('FIG GLOBAL MERGE STAGE & SPEED')
colors = {'sp':{'notcooccur':None, 'cooccur':'skyblue'} , 'sw':{'notcooccur':'forestgreen', 'cooccur':'limegreen'}}
ratio = get_respi_ratio(subject = '*', stage = '*', ratio_df = cycles_ratios)

for chan in channels_events_select:
    fig, axs = plt.subplots(2, 2, figsize = (20,10), constrained_layout = True, subplot_kw=dict(projection = 'polar'))

    for r, ev in enumerate(['sp','sw']):
        for c, cooccur in enumerate(['notcooccur','cooccur']):
            ax = axs[r,c]

            color = colors[ev][cooccur]
            angles = load_grouped_angles(subject = '*' , event = ev, stage = '*',cooccuring = cooccur, speed = '*', chan = chan)
            
            if angles.size == 0:
                continue
                
            circular_plot_angles(angles, color=color, ax=ax, ratio_plot = ratio, with_title = True, with_arrow = True, with_rticks = True)
            title = f'{ev} - {cooccur} \n' + ax.get_title()
            ax.set_title(title, fontsize = 15, y = 1.1)  

    plt.savefig(f'../events_coupling_stats/global/merge_stage_speed/polar_plot_{chan}', bbox_inches = 'tight')
    plt.close()
    
    
    
# GLOBAL MERGE COOCCURING & SPEED
print('FIG GLOBAL MERGE OCCURING & SPEED')
colors = {'sp':{'N2':None, 'N3':'skyblue'} , 'sw':{'N2':'forestgreen', 'N3':'limegreen'}}
for chan in channels_events_select:
    fig, axs = plt.subplots(2, 2, figsize = (20,10), constrained_layout = True, subplot_kw=dict(projection = 'polar'))

    for r, ev in enumerate(['sp','sw']):
            for c, stage in enumerate(['N2','N3']):
                
                ratio = get_respi_ratio(subject = '*', stage = stage, ratio_df = cycles_ratios)
                
                ax = axs[r,c]
                
                color = colors[ev][stage]
                angles = load_grouped_angles(subject = '*' , event = ev, stage = stage,cooccuring = '*', speed = '*', chan = chan)
                
                if angles.size == 0:
                    continue
                    
                circular_plot_angles(angles, color=color, ax=ax, ratio_plot = ratio, with_title = True, with_arrow = True, with_rticks = True)
                title = f'{ev} - {stage} \n' + ax.get_title()
                ax.set_title(title, fontsize = 15, y = 1.1)  

    plt.savefig(f'../events_coupling_stats/global/merge_occuring_speed/polar_plot_{chan}', bbox_inches = 'tight')
    plt.close()
    
    
# SUBJECT MERGE
print('FIG by SUBJECT')
colors = {'sp':{'notcooccur':None, 'cooccur':'skyblue'} , 'sw':{'notcooccur':'forestgreen', 'cooccur':'limegreen'}}
for chan in channels_events_select:
    for subject in subjects:
        
        fig, axs = plt.subplots(2, 2, figsize = (20,10), constrained_layout = True, subplot_kw=dict(projection = 'polar'))
        
        ratio = get_respi_ratio(subject = subject, stage = '*', ratio_df = cycles_ratios)

        for r, ev in enumerate(['sp','sw']):
            for c, cooccur in enumerate(['notcooccur','cooccur']):

                ax = axs[r,c]
                
                color = colors[ev][cooccur]
                angles = load_grouped_angles(subject = subject , event = ev, stage = '*', cooccuring = cooccur, speed = '*', chan = chan)
                
                if angles.size == 0:
                    continue 
                    
                circular_plot_angles(angles, color=color, ax=ax, ratio_plot = ratio, with_title = True, with_arrow = True, with_rticks = True)
                title = f'{ev} - {cooccur} \n' + ax.get_title()
                ax.set_title(title, fontsize = 15, y = 1.1)  

        plt.savefig(f'../events_coupling_stats/subjects/{chan}/polar_plot_{subject}_{chan}', bbox_inches = 'tight')
        plt.close()