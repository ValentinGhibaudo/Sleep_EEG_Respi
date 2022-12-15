import numpy as np
import pandas as pd
import json
import pingouin as pg
import matplotlib.pyplot as plt
from params import subjects, dpis, stages_events_select

"""
This script generate 3 types of figs:
- 1 : One fig by subject comparing spindle to slow wave polar plot
- 2 : Two figs grouping polar plots from 20 subjects / fig , one fig by event type (spindle or slow-wave), polar plot
- 3 : Mean polar plots of all events pooled from all subjects
"""

#####

def load_dict(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data

def flatten_2d_list(list_2d):
    return [item for sublist in list_2d for item in sublist]

##### 

event_types = ['sp','sw'] # run keys for spindles and slow waves
event_types_titles = {'sp':'Spindles','sw':'Slow-Waves'} # understandable labels for spindles and slowwaves
bars_colors = {'sp':{'N2':None,'N3':'skyblue'},'sw':{'N2':'forestgreen','N3':'limegreen'}} # colors for polarplots
bins = 18 # histograms of distribution of events according to resp phase will be distributed in this number of bins
cycles_ratios = pd.read_excel('../resp_stats/cycle_ratios.xlsx', index_col = 0).set_index(['subject','sleep_stage']) # get cycles ratio for the phase transition in polar plots

# FIG TYPE 1 : SPINDLE vs SLOW-WAVE POLAR PLOT FOR EACH PARTICIPANT
print('FIG 1')

for subject in subjects:

    fig, axs = plt.subplots(nrows = len(stages_events_select), ncols = len(event_types), subplot_kw=dict(projection = 'polar'), figsize = (15,7), constrained_layout = True) # init polar plot figure
    fig.suptitle(subject, fontsize = 20) # suptitle

    for subrow, stage in enumerate(stages_events_select):
        
        mean_ratio = cycles_ratios.loc[(subject,stage), 'cycle_ratio'] # sel mean respi cycle ratio of the stage of the subject

        for subcol, event_type in enumerate(event_types): # loop on events types (sp and sw) that will be displayed in columns
            phase_angles = np.load(f'../events_coupling/{subject}_{event_type}_{stage}_phase_angles.npy')
            N_events = phase_angles.size
            
            ax = axs[subrow, subcol] # choose the right subplot 
            ax.hist(phase_angles, bins = bins, density = False, edgecolor = 'black', color = bars_colors[event_type][stage]) # polar histogram of distribution of angles (in radians)
            ax.set_rticks([]) # remove vector lengths of the plot
            ax.set_title(f'{stage} - {event_types_titles[event_type]} - N = {N_events}', fontsize = 15, y = 1.1) 

            tick = [ax.get_rmax()*0.995, ax.get_rmax() * 0.99] # will display some markers at the exterior part of the polarplot (indicating respi phase)
            tick_transition_pos = mean_ratio * 360 # angle where ticks will be colored differently = at the inspi to expi transition
            for t in np.deg2rad(np.arange(0, 360, 2)): # loop ticks displaying around the polar circle
                if t <= np.deg2rad(tick_transition_pos) and t >= 0: # if angle < resp transition, color is red
                    color = 'r'
                elif t > np.deg2rad(tick_transition_pos) and t <= np.deg2rad(360): # if angle > resp transition, color is black
                    color = 'k'
                ax.plot([t, t], tick, lw=8, color=color) # plot the ticks, lw = linewidth = width of each tick
                ax.set_xticks(np.deg2rad([0, 90 , tick_transition_pos , 180 , 270])) # at this angles in degrees, ticks labels will be present
                ax.set_xticklabels(['Start', '90°', 'I>E', '180°','270°']) # labelize polar plot angles

    plt.savefig(f'../events_coupling_stats/{subject}_polar_plots.tif', format = 'tif',  bbox_inches = 'tight', dpi = dpis) # save in tif with set dpi for good quality
    plt.close()
    
    
    
# FIG TYPE 2 : SPINDLE POLAR PLOT FOR EACH PARTICIPANT and SLOW-WAVE POLAR PLOT FOR EACH PARTICIPANT
print('FIG 2')

subject_grid = np.array(subjects).reshape(4,5) # array of subjects corresponding to subplots
nrows = subject_grid.shape[0]
ncols = subject_grid.shape[1]

for event_type in event_types: # loop on events types (sp and sw) that will be displayed in columns
    for stage in stages_events_select:

        fig, axs = plt.subplots(nrows=nrows, ncols = ncols, subplot_kw=dict(projection = 'polar'), figsize = (15,15)) # init polar plot figure
        fig.suptitle(f'{event_types_titles[event_type]} - {stage}', fontsize = 20)
        
        for row in range(nrows):
            for col in range(ncols):
                ax = axs[row, col]
                subject = subject_grid[row, col] # sel subject

                mean_ratio = cycles_ratios.loc[(subject,stage), 'cycle_ratio'] # sel mean respi cycle ratio of the stage of the subject
        
                phase_angles = np.load(f'../events_coupling/{subject}_{event_type}_{stage}_phase_angles.npy')
                N_events = phase_angles.size # number of angles computed and distributed = number of events detected and computed
 
                ax.hist(phase_angles, bins = bins, density = False, edgecolor = 'black', color = bars_colors[event_type][stage]) # polar histogram of distribution of angles (in radians)
                ax.set_rticks([]) # remove vector lengths of the plot
                ax.set_title(f'{subject} - N = {N_events}', fontsize = 15, y = 1.10) 

                tick = [ax.get_rmax()*0.995, ax.get_rmax() * 0.99] # will display some markers at the exterior part of the polarplot (indicating respi phase)
                tick_transition_pos = mean_ratio * 360 # angle where ticks will be colored differently = at the inspi to expi transition
                for t in np.deg2rad(np.arange(0, 360, 2)): # loop ticks displaying around the polar circle
                    if t <= np.deg2rad(tick_transition_pos) and t >= 0: # if angle < resp transition, color is red
                        color = 'r'
                    elif t > np.deg2rad(tick_transition_pos) and t <= np.deg2rad(360): # if angle > resp transition, color is black
                        color = 'k'
                    ax.plot([t, t], tick, lw=8, color=color) # plot the ticks, lw = linewidth = width of each tick
                    ax.set_xticks(np.deg2rad([0, 90 , tick_transition_pos , 180 , 270])) # at this angles in degrees, ticks labels will be present
                    ax.set_xticklabels(['Start', '90°', 'I>E', '180°','270°']) # labelize polar plot angles


        plt.savefig(f'../events_coupling_stats/{event_type}_{stage}_polar_plots.tif', format = 'tif',  bbox_inches = 'tight', dpi = dpis) # save in tif with set dpi for good quality
        plt.close()


    
# FIG TYPE 3 : SPINDLE vs SLOW-WAVE MEAN POLAR PLOT by STAGE
print('FIG 3')

pooled_angles = load_dict('../events_coupling/pooled_angles.txt')

fig, axs = plt.subplots(nrows=len(stages_events_select), ncols = len(event_types), subplot_kw=dict(projection = 'polar'), figsize = (15,7), constrained_layout = True) # polar plot figure

for row, stage in enumerate(stages_events_select):
    ratio_plot = cycles_ratios.loc[('mean',stage),'cycle_ratio']
    for col, event_type in enumerate(event_types): # loop on event types (sp and sw)  in columns

        
        ax = axs[row, col] # choose the right subplot
        angles = np.array(flatten_2d_list(pooled_angles[event_type][stage])) # array from list with all angles from all subjects from one stage
        N_events = angles.size # number of angles computed and distributed = number of events detected and computed (all subjects pooled)
        ax.hist(angles, bins = bins, density = False, edgecolor = 'black', color = bars_colors[event_type][stage]) # polar histogram of distribution of angles of all subjects (in radians)
        ax.set_rticks([]) # remove vector lengths
        ax.set_title(f'{event_types_titles[event_type]} - Stage = {stage} - N = {N_events}', fontsize = 15, y = 1.1) 
        
        tick = [ax.get_rmax()*0.995, ax.get_rmax() * 0.99] # will display some markers at the exterior part of the polarplot (indicating respi phase)
        tick_pos = ratio_plot * 360 # angle where ticks will be colored differently = at the inspi to expi transition
        for t in np.deg2rad(np.arange(0, 360, 2)): # loop ticks displaying around the polar circle
            if t <= np.deg2rad(tick_pos) and t >= 0: # if angle < resp transition, color is red
                color = 'r'
            elif t >= np.deg2rad(tick_pos) and t <= np.deg2rad(360): # if angle > resp transition, color is black
                color = 'k'
            ax.plot([t, t], tick, lw=8, color=color) # plot the ticks, lw = linewidth = width of each tick
        ax.set_xticks(np.deg2rad([0, 90 , tick_pos , 180 , 270])) # at this angles in degrees, ticks labels will be present
        ax.set_xticklabels(['Start', '90°', 'I>E', '180°','270°']) # labelize polar plot angles
    
plt.savefig('../events_coupling_stats/mean_polar_plot.tif', format = 'tif', bbox_inches = 'tight', dpi = dpis) # save in tif with set dpi for good quality
plt.close()


