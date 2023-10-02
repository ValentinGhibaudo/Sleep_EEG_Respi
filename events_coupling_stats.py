import numpy as np
import pandas as pd
import pingouin as pg
from fnmatch import filter
from params import *
import xarray as xr
from events_coupling import event_coupling_job
from configuration import *

def get_circ_features(angles): # angles in radians

    z, pval = pg.circ_rayleigh(angles)
    mu = pg.circ_mean(angles) #+ np.pi
    mu = int(np.degrees(mu))
    r = round(pg.circ_r(angles), 3)

    if mu < 0:
        mu = 360 + mu

    return pval, mu, r

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


def load_grouped_angles(subject, event, cooccuring, speed, chan, half):

    """
    High level function that load angles according to arguments and concatenate them if '*' argument

    Parameters (str)
    ----------
    subject : From 'S1' to 'S20', '*' to concatenate all
    event : 'spindles' or 'slowwaves' for spindles or slow-wave 
    cooccuring : 'cooccur' or 'notcoocur', '*' to concatenate both
    speed : 'SS' or 'FS' for slow or fast spindles, '*' to concatenate both (useful only for spindles)
    half : 'q1', 'q2','q3','q4', of night, '*' to concatenate all
    chan : 'Fz' for example
    """

    if subject == '*':
        concat = [event_coupling_job.get(run_key) for run_key in run_keys]
        ds_search = xr.concat(concat, dim = 'subject')
    else:
        ds_search = event_coupling_job.get(subject)

    if event == 'spindles':
        pattern = f'{subject}_spindles_{cooccuring}_{speed}_{half}_{chan}'
    elif event == 'slowwaves':
        pattern = f'{subject}_slowwaves_{cooccuring}_{half}_{chan}'

    return np.array(get_angles(ds_search, pattern))

def readable_pval(pval):
    return round(pval, 4) if pval >= 0.001 else '< 0.001'

p = events_coupling_stats_params
save_article = p['save_article']
stage = p['stage']
subjects = run_keys
chan = p['chan']

if save_article:
    save_folder = article_folder  
else:
    save_folder = base_folder / 'results' / 'events_coupling_figures'

# STATS SP SW INDIVIDUAL
rows = []
for subject in subjects:
    for event_type in ['spindles','slowwaves']:
        angles = load_grouped_angles(subject = subject , event = event_type, cooccuring = '*', speed = '*', chan = chan, half = '*')
        N = angles.size
        pval, mu, r = get_circ_features(angles)
        rows.append([subject, event_type, readable_pval(pval), pval_stars(pval), mu, r])
stats = pd.DataFrame(rows, columns = ['Subject','Event','p-Rayleigh','Rayleigh Significance','Mean Direction (°)','Mean Vector Length'])

spindles_stats = stats[stats['Event'] == 'spindles'].set_index('Subject')
slowwaves_stats = stats[stats['Event'] == 'slowwaves'].set_index('Subject')

concat_events = [spindles_stats,slowwaves_stats]
stats_return = pd.concat(concat_events, axis = 1)
stats_return.round(3).reset_index().to_excel(save_folder / 'circ_stats_spindles_slowwaves_individual.xlsx', index = False)

# STATS SP SPEED POOLED
rows = []
for speed in ['SS','FS']:
    angles = load_grouped_angles(subject = '*' , event = 'spindles', cooccuring = '*', speed = speed, chan = chan, half = '*')
    N = angles.size
    pval, mu, r = get_circ_features(angles)
    speed_label = 'Slow' if speed == 'SS' else 'Fast'
    rows.append(['spindles',speed_label , N  , readable_pval(pval), mu , r])
stats_pooled = pd.DataFrame(rows, columns = ['event','speed','N','p-Rayleigh','Mean Direction (°)','Mean Vector Length'])
stats_pooled.to_excel(save_folder / 'circ_stats_spindles_speed_pooled.xlsx', index = False)


# STATS SP and SW SPEED HALF-NIGHT POOLED
ev_looped = ['Slow spindles','Fast spindles','Slow-Waves']
ev_load_dict = {'Slow spindles':'spindles_SS','Fast spindles':'spindles_FS','Slow-Waves':'slowwaves'}
rows = []
for half in ['firsthalf','secondhalf']:
    for ev_clean in ev_looped:
        if not ev_clean == 'Slow-Waves':
            ev_load, speed = ev_load_dict[ev_clean].split('_')
        else:
            ev_load = ev_load_dict[ev_clean]
            speed = 'NA'

        angles = load_grouped_angles(subject = '*' , event = ev_load, cooccuring = '*', speed = speed, chan = chan, half = half)
        N = angles.size
        pval, mu, r = get_circ_features(angles)
        stars = pval_stars(pval)
        row = [ev_clean, half.split('h')[0] , N  , readable_pval(pval), stars, mu , r]
        rows.append(row)
stats_pooled = pd.DataFrame(rows, columns = ['Event','Half-Night','N','p-Rayleigh','Significance','Angle','MVL'])
stats_pooled.to_excel(save_folder / 'circ_stats_events_speed_halfnight_pooled.xlsx', index = False)


# STATS SP SW POOLED
rows = []
for ev in ['spindles','slowwaves']:
    angles = load_grouped_angles(subject = '*' , event = ev, cooccuring = '*', speed = '*', chan = chan, half = '*')
    N = angles.size
    pval, mu, r = get_circ_features(angles)
    rows.append([ev, N  , readable_pval(pval), mu , r])
stats_pooled = pd.DataFrame(rows, columns = ['event','N','p-Rayleigh','Mean Direction (°)','Mean Vector Length'])
stats_pooled.to_excel(save_folder / 'circ_stats_spindles_slowwaves_pooled.xlsx', index = False)



# STATS SP SPEED SW POOLED
ev_looped = ['Slow spindles','Fast spindles','Slow-Waves']
ev_load_dict = {'Slow spindles':'spindles_SS','Fast spindles':'spindles_FS','Slow-Waves':'slowwaves'}
rows = []
for chan in channels_events_select:
    for ev_clean in ev_looped:
        if not ev_clean == 'Slow-Waves':
            ev_load, speed = ev_load_dict[ev_clean].split('_')
        else:
            ev_load = ev_load_dict[ev_clean]
            speed = '--'

        angles = load_grouped_angles(subject = '*' , event = ev_load, cooccuring = '*', speed = speed, chan = chan, half = '*')
        N = angles.size
        pval, mu, r = get_circ_features(angles)
        stars = pval_stars(pval)
        row = [chan, ev_clean, N  , readable_pval(pval), stars, mu , r]
        rows.append(row)
stats_pooled = pd.DataFrame(rows, columns = ['Channel','Event','N','p-Rayleigh','Rayleigh Significance','Mean Direction (°)','Mean Vector Length'])
stats_pooled.to_excel(save_folder / 'circ_stats_spindles_speed_slowwaves_pooled.xlsx', index = False)




