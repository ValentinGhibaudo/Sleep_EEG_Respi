import numpy as np
import pandas as pd
import pingouin as pg
from params import *
import xarray as xr
from events_coupling import event_coupling_job, concat_events_coupling_job
from configuration import *
from circular_stats import HR2P

def get_circ_features(angles, univals=1000, seed=None, progress_bar = False, resample=True, size_resample = 10000): # angles in radians
    if angles.size > 10000:
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
    return round(pval, 4) if pval >= 0.01 else '< 0.01'

def sub_to_p(sub):
    return 'P{}'.format(sub.split('S')[1])

p = events_coupling_stats_params

subjects = run_keys
univals = p['univals']

# save_folder = base_folder / 'results' / 'events_coupling_figures' / 'stats'
save_folder = base_folder / 'autres' / 'article_N20' / 'clin_neurophy_submission2' / 'reviewing_2'



compute_S3 = False
compute_S2 = False
compute_fig3A = False
compute_fig3B = False

compute_for_reviewer_3 = True


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
                if ev_clean == 'Slow-Waves':
                    row = [ev_clean, quart , N  , readable_pval(pval), stars, 'NA' , 'NA']
                else:
                    row = [ev_clean, quart , N  , readable_pval(pval), stars, mu , r]
            else:
                row = [ev_clean, quart , N  , 'NA', 'NA', 'NA' , 'NA']                
            rows.append(row)
    stats_pooled = pd.DataFrame(rows, columns = ['Event','Night quartile','N','p-HR','Significance','Mean Direction (°)','Mean Vector Length'])
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
                if ev_clean == 'Slow-Waves':
                    row = [chan, ev_clean, N  , readable_pval(pval), stars, 'NA' , 'NA']
                else:
                    row = [chan, ev_clean, N  , readable_pval(pval), stars, mu , r]
            else:
                row = [chan, ev_clean, N  , 'NA', 'NA', 'NA' , 'NA']
            rows.append(row)
    stats_pooled = pd.DataFrame(rows, columns = ['Channel','Event','N','p-HR','Significance','Mean Direction (°)','Mean Vector Length'])
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
            if ev_clean == 'Slow-Waves':
                row = [ev_clean, N  , readable_pval(pval), stars, 'NA' , 'NA']
            else:
                row = [ev_clean, N  , readable_pval(pval), stars, mu , r]
        else:
            row = [ev_clean, N  , 'NA', 'NA', 'NA' , 'NA']
        rows.append(row)
    res = pd.DataFrame(rows, columns = ['Event','N','p-HR','Significance','Mean Direction (°)','Mean Vector Length'])
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
            row = [ev, N  , readable_pval(pval), stars, mu , r]
        else:
            row = [ev, N  , 'NA', 'NA', 'NA' , 'NA']
        rows.append(row)
    res = pd.DataFrame(rows, columns = ['Event','N','p-HR','Significance','Mean Direction (°)','Mean Vector Length'])
    res.to_excel(save_folder / f'circ_stats_fig3B.xlsx', index = False)   



# WASTE 



# chan = 'Fz'
# ev_looped = ['Slow spindles','Fast spindles','Slow-Waves']
# ev_load_dict = {'Slow spindles':'Spindles_SS','Fast spindles':'Spindles_FS','Slow-Waves':'SlowWaves'}

# rows = []
# for chan in channels_events_select:
#     for ev_clean in ev_looped:
#         if not ev_clean == 'Slow-Waves':
#             ev_load, speed = ev_load_dict[ev_clean].split('_')
#         else:
#             ev_load = ev_load_dict[ev_clean]
#             speed = '--'

#         angles = load_angles(subject = '*' , event = ev_load, cooccuring = '*', speed = speed, chan = chan, quart_night = '*', stage='*')
#         N = angles.size
#         if not N == 0:
#             pval, mu, r = get_circ_features(angles, univals=univals)
#             stars = pval_stars(pval)
#             if ev_clean == 'Slow-Waves':
#                 row = [chan, ev_clean, N  , readable_pval(pval), stars, 'NA' , 'NA']
#             else:
#                 row = [chan, ev_clean, N  , readable_pval(pval), stars, mu , r]
#         else:
#             row = [chan, ev_clean, N  , 'NA', 'NA', 'NA' , 'NA']
#         rows.append(row)
# stats_pooled = pd.DataFrame(rows, columns = ['Channel','Event','N','p-HR','Significance','Mean Direction (°)','Mean Vector Length'])
# stats_pooled.to_excel(save_folder / f'supplementary_table_2_N2N3_{chan}.xlsx', index = False)



      
# ev_looped = ['Slow spindles','Fast spindles','Slow-Waves']
# ev_load_dict = {'Slow spindles':'Spindles_SS','Fast spindles':'Spindles_FS','Slow-Waves':'SlowWaves'}
# chan = '*'
# rows = []
# for quart in ['q1','q2','q3','q4']:
#     for ev_clean in ev_looped:
#         if not ev_clean == 'Slow-Waves':
#             ev_load, speed = ev_load_dict[ev_clean].split('_')
#         else:
#             ev_load = ev_load_dict[ev_clean]
#             speed = 'NA'

#         angles = load_angles(subject = '*' , event = ev_load, cooccuring = '*', speed = speed, chan = chan, quart_night = quart, stage='*')
#         N = angles.size
#         if not N == 0:
#             pval, mu, r = get_circ_features(angles, univals=univals)
#             stars = pval_stars(pval)
#             if ev_clean == 'Slow-Waves':
#                 row = [ev_clean, quart , N  , readable_pval(pval), stars, 'NA' , 'NA']
#             else:
#                 row = [ev_clean, quart , N  , readable_pval(pval), stars, mu , r]
#         else:
#             row = [ev_clean, quart , N  , 'NA', 'NA', 'NA' , 'NA']                
#         rows.append(row)
# stats_pooled = pd.DataFrame(rows, columns = ['Event','Quart-Night','N','p-HR','Significance','Mean Direction (°)','Mean Vector Length'])
# if chan == '*':
#     chan_save = 'allchan'
# stats_pooled.to_excel(save_folder / f'supplementary_table_3_N2N3_{chan_save}.xlsx', index = False)


# # STATS SP SW POOLED
# print('sp sw pooled')
# for stage in compute_stage:
#     rows = []
#     for ev in ['Spindles','SlowWaves']:
#         angles = load_angles(subject = '*' , event = ev, cooccuring = '*', speed = '*', chan = chan, quart_night = '*')
#         N = angles.size
#         if not N == 0:
#             pval, mu, r = get_circ_features(angles, univals=univals)
#             if ev == 'SlowWaves':
#                 rows.append([ev, N ,  readable_pval(pval), pval_stars(pval), 'NA', 'NA'])
#             else:
#                 rows.append([ev, N ,  readable_pval(pval), pval_stars(pval), mu, r])
#         else:
#             rows.append([ev, N  , 'NA', 'NA' , 'NA'])            
#     stats_pooled = pd.DataFrame(rows, columns = ['Event','N','p-HR','Significance','Mean Direction (°)','Mean Vector Length'])
#     stats_pooled.to_excel(save_folder / f'circ_stats_spindles_slowwaves_pooled_{stage}.xlsx', index = False)



# # STATS SP SW INDIVIDUAL
# print('sp sw individual')
# rows = []
# for stage in compute_stage:
#     for subject in subjects:
#         for event_type in ['Spindles','SlowWaves']:
#             angles = load_angles(subject = subject , event = event_type, cooccuring = '*', speed = '*', chan = chan, quart_night = '*', stage = stage)
#             N = angles.size
#             if N != 0:
#                 pval, mu, r = get_circ_features(angles, univals=univals)
#                 if event_type == 'SlowWaves':
#                     rows.append([sub_to_p(subject), event_type, N ,  readable_pval(pval), pval_stars(pval), 'NA', 'NA'])
#                 else:
#                     rows.append([sub_to_p(subject), event_type, N ,  readable_pval(pval), pval_stars(pval), mu, r])
#             else:
#                 rows.append([sub_to_p(subject), event_type, N ,  'NA', 'NA', 'NA', 'NA'])                
#     stats = pd.DataFrame(rows, columns = ['Participant','Event','N','p-HR','Significance','Mean Direction (°)','Mean Vector Length'])

#     spindles_stats = stats[stats['Event'] == 'Spindles'].set_index('Participant')
#     slowwaves_stats = stats[stats['Event'] == 'SlowWaves'].set_index('Participant')

#     concat_events = [spindles_stats,slowwaves_stats]
#     stats_return = pd.concat(concat_events, axis = 1)
#     stats_return.round(3).reset_index().to_excel(save_folder / f'circ_stats_spindles_slowwaves_{stage}_individual.xlsx', index = False)

# # STATS SP SPEED POOLED
# print('sp speed pooled')
# for stage in compute_stage:
#     rows = []
#     for speed in ['SS','FS']:
#         speed_label = 'Slow' if speed == 'SS' else 'Fast'
#         angles = load_angles(subject = '*' , event = 'Spindles', cooccuring = '*', speed = speed, chan = chan, quart_night = '*', stage=stage)
#         N = angles.size
#         if N != 0:
#             pval, mu, r = get_circ_features(angles, univals=univals)
#             rows.append(['Spindles',speed_label , N  , readable_pval(pval), mu , r])
#         else:
#             rows.append(['Spindles',speed_label , N  , 'NA', 'NA' , 'NA'])            
#     stats_pooled = pd.DataFrame(rows, columns = ['Event','speed','N','p-HR','Mean Direction (°)','Mean Vector Length'])
#     stats_pooled.to_excel(save_folder / f'circ_stats_spindles_speed_pooled_{stage}.xlsx', index = False)



# FOR ANSWER TO REVIEWER 3
if compute_for_reviewer_3:
    
    bonferroni_factor = 49 # number of p-values that are computed in the paper
    
#     print('S3')
    
#     ev_looped = ['Slow spindles','Fast spindles','Slow-Waves']
#     ev_load_dict = {'Slow spindles':'Spindles_SS','Fast spindles':'Spindles_FS','Slow-Waves':'SlowWaves'}

#     subject = '*'
#     chan = '*'
#     stage = '*'
#     cooccuring = '*'

#     rows = []
#     for quart in ['q1','q2','q3','q4']:
#         print(quart)
#         for ev_clean in ev_looped:
#             if not ev_clean == 'Slow-Waves':
#                 ev_load, speed = ev_load_dict[ev_clean].split('_')
#             else:
#                 ev_load = ev_load_dict[ev_clean]
#                 speed = 'NA'

#             angles = load_angles(subject = subject , 
#                                  event = ev_load, 
#                                  cooccuring = cooccuring, 
#                                  speed = speed, 
#                                  chan = chan, 
#                                  quart_night = quart, 
#                                  stage=stage)
#             N = angles.size
#             pval_uncorrected, mu, r = get_circ_features(angles, univals=univals)
#             pval_corrected = pval_uncorrected * bonferroni_factor

#             if ev_clean == 'Slow-Waves':
#                 row = [ev_clean, quart , N  , pval_uncorrected, pval_corrected, 'NA' , 'NA']
#             else:
#                 row = [ev_clean, quart , N  , pval_uncorrected, pval_corrected, mu , r]             
#             rows.append(row)
#     stats_pooled = pd.DataFrame(rows, columns = ['Event','Night quartile','N','p-HR uncorrected','p-HR Bonferroni corrected','Mean Direction (°)','Mean Vector Length'])
#     stats_pooled.to_excel(save_folder / f'supplementary_table_3_pvalues_corrected.xlsx', index = False)

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
            pval_uncorrected, mu, r = get_circ_features(angles, univals=univals)
            pval_corrected = pval_uncorrected * bonferroni_factor
            if ev_clean == 'Slow-Waves':
                row = [chan, ev_clean, N  , pval_uncorrected, pval_corrected, 'NA' , 'NA']
            else:
                row = [chan, ev_clean, N  , pval_uncorrected, pval_corrected, mu , r]
            rows.append(row)
    stats_pooled = pd.DataFrame(rows, columns = ['Channel','Event','N','p-HR uncorrected','p-HR Bonferroni corrected','Mean Direction (°)','Mean Vector Length'])
    stats_pooled.to_excel(save_folder / f'supplementary_table_2_pvalues_corrected.xlsx', index = False)   


    # FIG 3A
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

        pval_uncorrected, mu, r = get_circ_features(angles, univals=univals)
        pval_corrected = pval_uncorrected * bonferroni_factor
        if ev_clean == 'Slow-Waves':
            row = [ev_clean, N  , pval_uncorrected, pval_corrected, 'NA' , 'NA']
        else:
            row = [ev_clean, N  , pval_uncorrected, pval_corrected, mu , r]

        rows.append(row)
    res = pd.DataFrame(rows, columns = ['Event','N','p-HR uncorrected','p-HR Bonferroni corrected','Mean Direction (°)','Mean Vector Length'])
    res.to_excel(save_folder / f'circ_stats_fig3A_pvalues_corrected.xlsx', index = False)   

    # FIG 3B
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

        pval_uncorrected, mu, r = get_circ_features(angles, univals=univals)
        pval_corrected = pval_uncorrected * bonferroni_factor
        row = [ev, N  , pval_uncorrected, pval_corrected, mu , r]

        rows.append(row)
    res = pd.DataFrame(rows, columns = ['Event','N','p-HR uncorrected','p-HR Bonferroni corrected','Mean Direction (°)','Mean Vector Length'])
    res.to_excel(save_folder / f'circ_stats_fig3B_pvalues_corrected.xlsx', index = False) 
