import numpy as np
import pandas as pd
import pingouin as pg
from params import *
import xarray as xr
from events_coupling import event_coupling_job, concat_events_coupling_job
from configuration import *
from circular_stats import HR2P

def get_circ_features(angles, univals=1000, seed=None, progress_bar = False, resample=True, size_resample = 10000): # angles in radians
    if angles.size > 10000 and resample:
        rng = np.random.default_rng(seed=seed)
        angles_resampled = rng.choice(angles, size = size_resample)
    else:
        angles_resampled = angles.copy()
    
    pval = HR2P(angles_resampled, univals=univals, seed=seed, progress_bar=progress_bar)

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

def load_angles(event, subject = '*', stage = '*', cooccuring = '*', speed = '*', chan = '*', quart_night = '*'):

    """
    High level function that load angles according to arguments and concatenate them if '*' argument

    Parameters (str)
    ----------
    event : 'Spindles' or 'SlowWaves'
    subject : From 'S1' to 'S20', '*' to concatenate all
    stage : 'N2' or 'N3, or '*' to concatenate all
    cooccuring : 'cooccur' or 'notcoocur', '*' to concatenate both
    speed : 'SS' or 'FS' for slow or fast spindles, '*' to concatenate both (useful only for spindles)
    quart_night : 'firsthalf' ot 'secondhalf' of night, '*' to concatenate both
    chan : 'Fz' for example or '*' to concatenate all
    """

    if subject == '*':
        df_angles = concat_events_coupling_job.get('global_key').to_dataframe()
    else:
        df_angles = event_coupling_job.get(subject).to_dataframe()

    df_angles = df_angles[df_angles['Event_type'] == event]

    if cooccuring == '*':
        mask_cooccuring = df_angles['cooccuring'].isin(df_angles['cooccuring'].unique())
    else:
        mask_cooccuring = df_angles['cooccuring'] == cooccuring
        
    if stage == '*':
        mask_stage = df_angles['Stage_Letter'].isin(df_angles['Stage_Letter'].unique())
    else:
        mask_stage = df_angles['Stage_Letter'] == stage        
    
    if speed == '*':
        mask_speed = df_angles['Sp_Speed'].isin(df_angles['Sp_Speed'].unique())
    else:
        mask_speed = df_angles['Sp_Speed'] == speed

    if chan == '*':
        mask_chan = df_angles['Channel'].isin(df_angles['Channel'].unique())
    else:
        mask_chan = df_angles['Channel'] == chan

    if quart_night == '*':
        mask_night = df_angles['night_quartile'].isin(df_angles['night_quartile'].unique())
    else:
        mask_night = df_angles['night_quartile'] == quart_night

    if event == 'Spindles':
        mask =  mask_stage & mask_cooccuring & mask_speed & mask_chan & mask_night 
    elif event == 'SlowWaves':
        mask = mask_stage & mask_cooccuring & mask_chan & mask_night 
        
    df_angles = df_angles[mask]
    return df_angles['Resp_Angle'].values

def readable_pval(pval):
    if pval >= 0.01:  
        p_return = round(pval, 5)
    elif pval < 0.01 and pval >= 0.001:
        p_return = '< 0.01'
    else:
        p_return = '< 0.001'
    return  p_return 

def sub_to_p(sub):
    return 'P{}'.format(sub.split('S')[1])

p = events_coupling_stats_params

subjects = run_keys
univals = p['univals']

# save_folder = base_folder / 'results' / 'events_coupling_figures' / 'stats'
# save_folder = base_folder / 'autres' / 'article_N20' / 'clin_neurophy_submission2' / 'reviewing_2'
save_folder = base_folder / 'autres' / 'article_N20' / 'clin_neurophy_submission2' / 'send_valentin_27022024'



compute_S3 = True
compute_S2 = True
compute_fig3A = True
compute_fig3B = True

bonferroni_factor = 49 # number of p-values that are computed in the paper

# SUPPLEMENTARY S3

if compute_S3:
    print('S3')
    ev_looped = ['Slow spindles','Fast spindles','Slow-Waves']
    ev_load_dict = {'Slow spindles':'Spindles_SS','Fast spindles':'Spindles_FS','Slow-Waves':'SlowWaves'}

    subject = '*'
    chan = '*'
    stage = '*'
    cooccuring = '*'

    rows = []
    for quart in ['q1','q2','q3','q4']:
        print(quart)
        for ev_clean in ev_looped:
            if not ev_clean == 'Slow-Waves':
                ev_load, speed = ev_load_dict[ev_clean].split('_')
            else:
                ev_load = ev_load_dict[ev_clean]
                speed = 'NA'

            angles = load_angles(subject = subject , 
                                 event = ev_load, 
                                 cooccuring = cooccuring, 
                                 speed = speed, 
                                 chan = chan, 
                                 quart_night = quart, 
                                 stage=stage)
            N = angles.size
            if not N == 0:
                pval, mu, r = get_circ_features(angles, univals=univals)
                stars = pval_stars(pval)
                pval_corrected = pval * bonferroni_factor
                if pval_corrected >= 1:
                    pval_corrected = 1
                if ev_clean == 'Slow-Waves':
                    row = [ev_clean, quart , N  , readable_pval(pval), stars, 'NA' , 'NA', pval, pval_corrected]
                else:
                    row = [ev_clean, quart , N  , readable_pval(pval), stars, mu , r, pval, pval_corrected]
            else:
                row = [ev_clean, quart , N  , 'NA', 'NA', 'NA' , 'NA', 'NA', 'NA']                
            rows.append(row)
    stats_pooled = pd.DataFrame(rows, columns = ['Event','Night quartile','N','p-HR','Significance','Mean Direction (°)','Mean Vector Length','p-HR uncorrected', 'p-HR corrected'])
    stats_pooled.to_excel(save_folder / f'supplementary_table_3.xlsx', index = False)


  
# SUPPLEMENTARY S2
if compute_S3:
    print('S2')
    ev_looped = ['Slow spindles','Fast spindles','Slow-Waves']
    ev_load_dict = {'Slow spindles':'Spindles_SS','Fast spindles':'Spindles_FS','Slow-Waves':'SlowWaves'}

    subject = '*'
    cooccuring = '*'
    quart_night = '*'
    stage='*'


    rows = []
    for chan in channels_events_select:
        print(chan)
        for ev_clean in ev_looped:
            if not ev_clean == 'Slow-Waves':
                ev_load, speed = ev_load_dict[ev_clean].split('_')
            else:
                ev_load = ev_load_dict[ev_clean]
                speed = '--'

            angles = load_angles(subject = subject , 
                                 event = ev_load, 
                                 cooccuring = cooccuring, 
                                 speed = speed, 
                                 chan = chan, 
                                 quart_night = quart_night, 
                                 stage=stage)
            N = angles.size
            if not N == 0:
                pval, mu, r = get_circ_features(angles, univals=univals)
                stars = pval_stars(pval)
                pval_corrected = pval * bonferroni_factor
                if pval_corrected >= 1:
                    pval_corrected = 1
                if ev_clean == 'Slow-Waves':
                    row = [chan, ev_clean, N  , readable_pval(pval), stars, 'NA' , 'NA', pval, pval_corrected]
                else:
                    row = [chan, ev_clean, N  , readable_pval(pval), stars, mu , r , pval, pval_corrected]
            else:
                row = [chan, ev_clean, N  , 'NA', 'NA', 'NA' , 'NA', 'NA', 'NA']
            rows.append(row)
    stats_pooled = pd.DataFrame(rows, columns = ['Channel','Event','N','p-HR','Significance','Mean Direction (°)','Mean Vector Length','p-HR uncorrected', 'p-HR corrected'])
    stats_pooled.to_excel(save_folder / f'supplementary_table_2.xlsx', index = False)   





# FOR STATS IN RESULTS SECTION 3.4.	Spindles and slow waves events coupling with respiration 
# FIG 3A
if compute_fig3A:
    print('fig3A stats')
    subject = '*'
    chan = 'Fz'
    cooccuring = '*'
    quart_night = '*'
    stage='*'
    speed = '*'

    evs = ['Spindles','SlowWaves']
    evs_cleans = ['Spindles','Slow-Waves']
    
    rows = []
    for ev, ev_clean in zip(evs,evs_cleans):
        angles = load_angles(event = ev, chan = chan)
        N = angles.size
        if not N == 0:
            pval, mu, r = get_circ_features(angles, univals=univals)
            stars = pval_stars(pval)
            pval_corrected = pval * bonferroni_factor
            if pval_corrected >= 1:
                pval_corrected = 1
            if ev_clean == 'Slow-Waves':
                row = [ev_clean, N  , readable_pval(pval), stars, 'NA' , 'NA', pval, pval_corrected]
            else:
                row = [ev_clean, N  , readable_pval(pval), stars, mu , r, pval, pval_corrected]
        else:
            row = [ev_clean, N  , 'NA', 'NA', 'NA' , 'NA', 'NA', 'NA']
        rows.append(row)
    res = pd.DataFrame(rows, columns = ['Event','N','p-HR','Significance','Mean Direction (°)','Mean Vector Length','p-HR uncorrected', 'p-HR corrected'])
    res.to_excel(save_folder / f'circ_stats_fig3A.xlsx', index = False)   

# FIG 3B
if compute_fig3B:
    print('fig3B stats')
    subject = '*'
    chan = 'Fz'
    cooccuring = '*'
    quart_night = '*'
    stage='*'

    ev_looped = ['Slow spindles','Fast spindles']
    ev_load_dict = {'Slow spindles':'Spindles_SS','Fast spindles':'Spindles_FS'}
    
    rows = []
    for ev in ev_looped:
        ev_load, speed = ev_load_dict[ev].split('_')
        angles = load_angles(event = ev_load, chan = chan, speed=speed)
        N = angles.size
        if not N == 0:
            pval, mu, r = get_circ_features(angles, univals=univals)
            stars = pval_stars(pval)
            pval_corrected = pval * bonferroni_factor
            if pval_corrected >= 1:
                pval_corrected = 1
            row = [ev, N  , readable_pval(pval), stars, mu , r, pval, pval_corrected]
        else:
            row = [ev, N  , 'NA', 'NA', 'NA' , 'NA', 'NA', 'NA']
        rows.append(row)
    res = pd.DataFrame(rows, columns = ['Event','N','p-HR','Significance','Mean Direction (°)','Mean Vector Length','p-HR uncorrected', 'p-HR corrected'])
    res.to_excel(save_folder / f'circ_stats_fig3B.xlsx', index = False)   


