import numpy as np
import pandas as pd
import json
import pingouin as pg
import matplotlib.pyplot as plt
from params import patients, subjects_from_patient, dpis

"""
This script generate 3 types of figs:
- 1 : One fig by subject comparing spindle to slow wave polar plot
- 2 : Two figs pooling events from 20 subjects / fig , one fig by event type (spindle or slow-wave), polar plot
- 3 : Mean polar plots of all events pooled from all subjects
- It also geenrates a table of circular statistics for all subjects, for the two types of events
"""

#####

def get_circ_features(angles): # angles in radians
    if angles.size != 0:
        z, pval = pg.circ_rayleigh(angles)
        mu = pg.circ_mean(angles) #+ np.pi
        mu = int(np.degrees(mu))
        r = round(pg.circ_r(angles), 3)
        if mu < 0:
            mu = 360 + mu
    else:
        pval, mu, r = np.nan, np.nan, np.nan
    return pval, mu, r

def load_angles(patient, event_type):
    path = f'../events_coupling/{patient}_{event_type}_phase_angles.txt'
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data

def p_stars(p):
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

##### 

savefig = True

event_types = ['sp','sw'] # run keys for spindles and slow waves
event_types_titles = {'sp':'Spindles','sw':'Slow-Waves'} # understandable labels for spindles and slowwaves
event_types_colors = {'sp':None,'sw':'forestgreen'} # colors for polarplots
bins = 18 # histograms of distribution of events according to resp phase will be distributed in this number of bins

# FIG TYPE 1 : SPINDLE vs SLOW-WAVE POLAR PLOT FOR EACH PARTICIPANT
all_events_angles = {'sp':[],'sw':[]} # dict where pooled angles from all subjects will be stored
rows_circular_stats = {'sp':[],'sw':[]} # dict of lists where circular stats from subjects will be stored
mean_resp_ratios = [] # list where mean inspi/duration cycle length will be stored
pooled_N_cycles_with_events = {'sp':[],'sw':[]} # dict of lists where number of resp cycle with at least one event is detected inside
pooled_N_cycles_without_events = {'sp':[],'sw':[]} # dict of lists where number of resp cycle without any event detected inside
pooled_N_cycles_total = {'sp':[],'sw':[]} # dict of lists where number of total resp cycle considered

for patient in patients:
    print(patient)
    subject = subjects_from_patient[patient]

    rsp_features = pd.read_excel(f'../resp_features/{patient}_resp_features.xlsx',index_col = 0) # load resp features
    mean_ratio = rsp_features['cycle_ratio'].mean() # compute mean of the inspi / duration = cycle ratio, across resp cycles
    mean_resp_ratios.append(mean_ratio) # add mean ratio to a list

    fig, axs = plt.subplots(ncols = len(event_types), subplot_kw=dict(projection = 'polar'), figsize = (15,7), constrained_layout = True) # init polar plot figure
    fig.suptitle(subject, fontsize = 20) # suptitle

    for col, event_type in enumerate(event_types): # loop on events types (sp and sw) that will be displayed in columns
        phase_angles = load_angles(patient,event_type) # dictionnary cycle:angles
        angles_in_radian = [] # list where angles of the event type will be pooled
        cycles_without_events = [] # pool cycle indexes where no event have been found
        for c in phase_angles.keys():
            angles = phase_angles[c] # list of angles
            if not angles is None:
                for angle in angles:
                    angles_in_radian.append(angle) # pool angles if events present during the respi cycle
            else:
                cycles_without_events.append(c) # store cycle index if not event present inside

        all_events_angles[event_type].append(angles_in_radian) # store angles of the event type in a list which pool angles from all subjects
        angles_in_radian = np.array(angles_in_radian) # array from list
        N_events = angles_in_radian.size # number of angles computed and distributed = number of events detected and computed
        N_cycles_without_event = len(cycles_without_events) # number of cycles without event present inside
        pooled_N_cycles_without_events[event_type].append(N_cycles_without_event)
        N_cycles_total = len(phase_angles.keys()) # total number of resp cycles
        pooled_N_cycles_total[event_type].append(N_cycles_total)
        N_cycles_with_event = N_cycles_total - N_cycles_without_event # number of cycles with at least one event detected inside
        pooled_N_cycles_with_events[event_type].append(N_cycles_with_event)
        proportion_cycles_with_event = N_cycles_with_event / N_cycles_total # proportion of cycles with at least one event detected inside
        
        ax = axs[col] # choose the right subplot 
        p, mu , r = get_circ_features(angles_in_radian) # compute circular stats from pingouin toolbox (p-value, mean angles, mean vector length)
        stars = p_stars(p) # pvalue to stars encoding
        if event_type == 'sp':
            row = [subject,  event_types_titles[event_type] , N_events, N_cycles_with_event,N_cycles_without_event, proportion_cycles_with_event, p, stars, mu , r] # tidy row for circular stats of the subject
        else:
            row = [event_types_titles[event_type] , N_events,N_cycles_with_event,N_cycles_without_event, proportion_cycles_with_event, p, stars, mu , r] # no 'subject' label for sw (already present in sp)
        rows_circular_stats[event_type].append(row) # add the tidy circualar subject's row to a list
        ax.hist(angles_in_radian, bins = bins, density = False, edgecolor = 'black', color = event_types_colors[event_type]) # polar histogram of distribution of angles (in radians)
        ax.set_rticks([]) # remove vector lengths of the plot
        # ax.set_title(f'# {event_type} # p-Rayleigh : {round(p, 4)} - mu : {int(mu)}° - MVL : {round(r, 3)}', fontsize = 15, y = 1.05)
        ax.set_title(f'{event_types_titles[event_type]} - N = {N_events}', fontsize = 15, y = 1.05) 

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

    if savefig:
        plt.savefig(f'../events_coupling_stats/{patient}_polar_plots.tif', format = 'tif',  bbox_inches = 'tight', dpi = dpis) # save in tif with set dpi for good quality

    plt.close()
    
    
    
# FIG TYPE 2 : SPINDLE POLAR PLOT FOR EACH PARTICIPANT and SLOW-WAVE POLAR PLOT FOR EACH PARTICIPANT
patient_grid = np.array(patients).reshape(4,5) # array of patients corresponding to subplots
nrows = patient_grid.shape[0]
ncols = patient_grid.shape[1]

for event_type in event_types: # loop on events types (sp and sw) that will be displayed in columns

    fig, axs = plt.subplots(nrows=nrows, ncols = ncols, subplot_kw=dict(projection = 'polar'), figsize = (15,10), constrained_layout = True, sharex = True, sharey = True) # init polar plot figure
    fig.suptitle(event_type)
    
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[row, col]
            patient = patient_grid[row, col] # sel patient
            subject = subjects_from_patient[patient] # convert patient to subject

            rsp_features = pd.read_excel(f'../resp_features/{patient}_resp_features.xlsx',index_col = 0) # load resp features
            mean_ratio = rsp_features['cycle_ratio'].mean() # compute mean of the inspi / duration = cycle ratio, across resp cycles
            mean_resp_ratios.append(mean_ratio) # add mean ratio to a list
    
            phase_angles = load_angles(patient,event_type) # dictionnary cycle:angles
            angles_in_radian = [] # list where angles of the event type will be pooled
            cycles_without_events = [] # pool cycle indexes where no event have been found
            
            for c in phase_angles.keys():
                angles = phase_angles[c] # list of angles
                if not angles is None:
                    for angle in angles:
                        angles_in_radian.append(angle) # pool angles if events present during the respi cycle
                else:
                    cycles_without_events.append(c) # store cycle index if not event present inside

            all_events_angles[event_type].append(angles_in_radian) # store angles of the event type in a list which pool angles from all subjects
            angles_in_radian = np.array(angles_in_radian) # array from list
            N_events = angles_in_radian.size # number of angles computed and distributed = number of events detected and computed
            N_cycles_without_event = len(cycles_without_events) # number of cycles without event present inside
            pooled_N_cycles_without_events[event_type].append(N_cycles_without_event)
            N_cycles_total = len(phase_angles.keys()) # total number of resp cycles
            pooled_N_cycles_total[event_type].append(N_cycles_total)
            N_cycles_with_event = N_cycles_total - N_cycles_without_event # number of cycles with at least one event detected inside
            pooled_N_cycles_with_events[event_type].append(N_cycles_with_event)
            proportion_cycles_with_event = N_cycles_with_event / N_cycles_total # proportion of cycles with at least one event detected inside
        
            ax.hist(angles_in_radian, bins = bins, density = False, edgecolor = 'black', color = event_types_colors[event_type]) # polar histogram of distribution of angles (in radians)
            ax.set_rticks([]) # remove vector lengths of the plot
            # ax.set_title(f'# {event_type} # p-Rayleigh : {round(p, 4)} - mu : {int(mu)}° - MVL : {round(r, 3)}', fontsize = 15, y = 1.05)
            ax.set_title(f'{event_types_titles[event_type]} - N = {N_events}', fontsize = 15, y = 1.05) 

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

    if savefig:
        plt.savefig(f'../events_coupling_stats/{event_type}_polar_plots.tif', format = 'tif',  bbox_inches = 'tight', dpi = dpis) # save in tif with set dpi for good quality

    plt.close()



    
    
# FIG TYPE 3 : SPINDLE vs SLOW-WAVE MEAN POLAR PLOT
mean_resp_ratio = np.mean(mean_resp_ratios) # mean of resp ratios across subjects
sum_cycles_with_events = {event_type:np.sum(pooled_N_cycles_with_events[event_type]) for event_type in pooled_N_cycles_with_events.keys()} # dict of number of resp cycles with at least one event detected inside, pooled on all subjects
sum_cycles_without_events ={event_type:np.sum(pooled_N_cycles_without_events[event_type]) for event_type in pooled_N_cycles_without_events.keys()} # dict of number of resp cycles without any event detected inside, pooled on all subjects
sum_cycles_total = {event_type:np.sum(pooled_N_cycles_total[event_type]) for event_type in pooled_N_cycles_total.keys()} # dict of number of resp cycles considered, all subjects pooled
global_proportion_cycles_with_event = {event_type:np.sum(sum_cycles_with_events[event_type]) / np.sum(sum_cycles_total[event_type])  for event_type in event_types} # dict of proportion of resp cycles with vs without at least one event detected inside, all subjects pooled

fig, axs = plt.subplots(ncols = len(event_types), subplot_kw=dict(projection = 'polar'), figsize = (15,7), constrained_layout = True) # polar plot figure

for col, event_type in enumerate(event_types): # loop on event types (sp and sw)  in columns

    concat_angles_all = [] # init a list that will be flattening the list of lists of subject's angles
    for angles in all_events_angles[event_type]:
        for angle in angles:
            concat_angles_all.append(angle) # flattened list of angles
    
    ax = axs[col] # choose the right subplot
    angles = np.array(concat_angles_all) # array from list with all angles from all subjects
    N_events = angles.size # number of angles computed and distributed = number of events detected and computed (all subjects pooled)
    p, mu , r = get_circ_features(angles) # compute global circular stats on all subjects with pingouin toolbox
    stars = p_stars(p)
    if event_type == 'sp':
        row = ['All pooled', event_types_titles[event_type], N_events,sum_cycles_with_events[event_type],sum_cycles_without_events[event_type], global_proportion_cycles_with_event[event_type], p, stars, mu , r] # tidy row for circular stats of all subjects
    else: 
        row = [event_types_titles[event_type] , N_events,sum_cycles_with_events[event_type],sum_cycles_without_events[event_type], global_proportion_cycles_with_event[event_type], p, stars, mu , r] # no mean label for sw (already present in sp)
    rows_circular_stats[event_type].append(row)
    ax.hist(angles, bins = bins, density = False, edgecolor = 'black', color = event_types_colors[event_type]) # polar histogram of distribution of angles of all subjects (in radians)
    ax.set_rticks([]) # remove vector lengths
    ax.set_title(f'{event_types_titles[event_type]} - N = {N_events}', fontsize = 15, y = 1.05) 
    
    tick = [ax.get_rmax()*0.995, ax.get_rmax() * 0.99] # will display some markers at the exterior part of the polarplot (indicating respi phase)
    tick_pos = mean_resp_ratio * 360 # angle where ticks will be colored differently = at the inspi to expi transition
    for t in np.deg2rad(np.arange(0, 360, 2)): # loop ticks displaying around the polar circle
        if t <= np.deg2rad(tick_pos) and t >= 0: # if angle < resp transition, color is red
            color = 'r'
        elif t >= np.deg2rad(tick_pos) and t <= np.deg2rad(360): # if angle > resp transition, color is black
            color = 'k'
        ax.plot([t, t], tick, lw=8, color=color) # plot the ticks, lw = linewidth = width of each tick
    ax.set_xticks(np.deg2rad([0, 90 , tick_pos , 180 , 270])) # at this angles in degrees, ticks labels will be present
    ax.set_xticklabels(['Start', '90°', 'I>E', '180°','270°']) # labelize polar plot angles
    
if savefig:
    plt.savefig('../events_coupling_stats/mean_polar_plot.tif', format = 'tif', bbox_inches = 'tight', dpi = dpis) # save in tif with set dpi for good quality

plt.close()


cols = {'sp':['Subject', 'Event', 'N Events', 'N Resp Cycles With Event', 'N Resp Cycles Without Event', 'Proportion Resp Cycles With Event', 'p-Rayleigh', 'Rayleigh Significance', 'Mean Direction (°)','Mean Vector Length'], # colnames
'sw':['Event', 'N Events', 'N Resp Cycles With Event', 'N Resp Cycles Without Event', 'Proportion Resp Cycles With Event', 'p-Rayleigh', 'Rayleigh Significance', 'Mean Direction (°)','Mean Vector Length']}

tables = [] # init future list of circular stats tables of all subjects and means
for event_type in event_types:
    stats_table = pd.DataFrame(rows_circular_stats[event_type], columns = cols[event_type])  # make df
    tables.append(stats_table)

whole_stats_table = pd.concat(tables, axis = 1) # concat in columns-axis the spindles and slowwaves circular stats
if savefig:
    whole_stats_table.to_excel('../events_coupling_stats/circular_stats_table.xlsx') # save the global circular stats dataframe
