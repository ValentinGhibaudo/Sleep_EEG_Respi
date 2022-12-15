import numpy as np
import pandas as pd
import json
import pingouin as pg
import matplotlib.pyplot as plt
from params import subjects, dpis, stages_events_select

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

def load_dict(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data

def flatten_2d_list(list_2d):
    return [item for sublist in list_2d for item in sublist]

pooled_angles = load_dict('../events_coupling/pooled_angles.txt') # load pooled angles from subjects
event_types_titles = {'sp':'Spindles','sw':'Slow-Waves'} # understandable labels for spindles and slowwaves
subjects_rows = subjects + ['all pooled'] # append an "all pooled" subject

event_types = ['sp','sw']
rows_circular_stats = {'sp':[],'sw':[]} # dict of lists where circular stats from subjects will be stored

for subject in subjects_rows:
    for stage in stages_events_select:
        for event_type in event_types:
            if not subject == 'all pooled':
                phase_angles = np.load(f'../events_coupling/{subject}_{event_type}_{stage}_phase_angles.npy') # angles of the subject for the right event_type and stage
            else:
                phase_angles = np.array(flatten_2d_list(pooled_angles[event_type][stage])) # pooled angles from all subjects for on event_type and stage
                
            p, mu , r = get_circ_features(phase_angles) # compute circular stats from pingouin toolbox (p-value, mean angles, mean vector length)
            stars = p_stars(p) # pvalue to stars encoding
            
            if event_type == 'sp':
                row = [subject, stage,  event_types_titles[event_type] , p, stars, mu , r] # tidy row for circular stats of the subject
            elif event_type == 'sw':
                row = [event_types_titles[event_type] , p, stars, mu , r] # no 'subject' label for sw (already present in sp)
            rows_circular_stats[event_type].append(row) # add the tidy circular subject's row to a list

sp_df = pd.DataFrame(rows_circular_stats['sp'], columns = ['subject','stage','event','p-Rayleigh','significance','Mean Angle','Mean Vector Length']) # circular stats of spindles
sw_df = pd.DataFrame(rows_circular_stats['sw'], columns = ['event','p-Rayleigh','significance','Mean Angle','Mean Vector Length']) # circular stats of slowwaves

circular_stats = pd.concat([sp_df, sw_df], axis = 1) # concat of spindles + slowwaves circular stats
circular_stats.to_excel('../events_coupling_stats/circular_stats_table.xlsx') # saving to excel

