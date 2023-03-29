import numpy as np
import pandas as pd
import pingouin as pg
from fnmatch import filter
from params import *
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


p = events_coupling_stats_params
save_article = p['save_article']
stage = p['stage']
subjects = run_keys
chan = p['chan']

if save_article:
    save_folder = article_folder / 'figs' 
else:
    save_folder = base_folder / 'results' / 'events_coupling_figures'

rows = []
for subject in subjects:
    for event_type in ['spindles','slowwaves']:
        angles = load_grouped_angles(subject = subject , event = event_type, cooccuring = '*', speed = '*', chan = chan)
        N = angles.size
        pval, mu, r = get_circ_features(angles)
        rows.append([subject, event_type, pval, pval_stars(pval), mu, r])
stats = pd.DataFrame(rows, columns = ['Subject','Event','p-Rayleigh','Rayleigh Significance','Mean Direction (°)','Mean Vector Length'])

spindles_stats = stats[stats['Event'] == 'spindles'].set_index('Subject')
slowwaves_stats = stats[stats['Event'] == 'slowwaves'].set_index('Subject')

concat_events = [spindles_stats,slowwaves_stats]
stats_return = pd.concat(concat_events, axis = 1)
stats_return.round(3).reset_index().to_excel(save_folder / 'events_coupling_stats.xlsx', index = False)


