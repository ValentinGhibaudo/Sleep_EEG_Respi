import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from fnmatch import filter
import xarray as xr
from circular_stats import HR2P
import jobtools
from params import *
from configuration import *
from events_coupling import event_coupling_job, concat_events_coupling_job
from rsp_detection import resp_tag_job



# p = events_coupling_figs_params
# stage = p['stage']
# subjects = run_keys
# save_article = p['save_article']
# univals = p['univals']
# with_stats = p['with_stats']

# if save_article:
#     save_folder = article_folder 
#     extension = '.tif'
#     dpis = 300
#     with_title = False
#     print('SAVING FIGURES IN ARTICLE FOLDER')
# else:
#     save_folder = base_folder / 'results' / 'events_coupling_figures'
#     extension = '.png'
#     dpis = 300
#     with_title = True
#     print('SAVING FIGURES IN RESULTS FOLDER')

#####

def get_circ_features(angles, univals=1000, seed=None, progress_bar = False): # angles in radians
    pval = HR2P(angles, univals=univals, seed=seed, progress_bar=progress_bar)

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
        mask = mask_stage & mask_cooccuring & mask_chan & mask_night & mask_stage

    df_angles = df_angles[mask]

    return df_angles['Resp_Angle'].values

def circular_plot_angles(
    angles, 
    color = None, 
    ax=None, 
    ratio_plot = 0.42, 
    bins = 18, 
    with_rticks = True, 
    with_title = False, 
    with_arrow = True, 
    polar_ticks = 'full',
    lw = 10,
    progress_bar = False,
    univals= 100,
    seed = None,
    with_stats = True):

    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection = 'polar'), constrained_layout = True)
    if with_stats:
        pval, mu , r = get_circ_features(angles, univals=univals, seed=seed, progress_bar=progress_bar)
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

    step = np.pi / 180
    theta_inspi = np.arange(0, ratio_plot * 2*np.pi, step)
    theta_expi = np.arange(ratio_plot * 2*np.pi, 2*np.pi, step)
    for theta, color in zip([theta_inspi, theta_expi],['r','k']):
        r_plot = np.ones(theta.size) * rmax * 1.05
        ax.plot(theta, r_plot, lw = lw, color =color)
    
    if polar_ticks == 'full':
        ax.set_xticks(np.deg2rad([0, 90 , ratio_plot * 360 , 180 , 270])) # at this angles in degrees, ticks labels will be present
        ax.set_xticklabels(['Start', '90°', 'I>E', '180°','270°']) # labelize polar plot angles
    elif polar_ticks == 'light':
        ax.set_xticks([ratio_plot * 2*np.pi]) # at this angles in degrees, ticks labels will be present
        ax.set_xticklabels(['I>E']) # labelize polar plot angles
    
    if with_title :
        if with_stats:
            ax.set_title(f'N : {N_events} \n Mean Angle : {mu}° - MVL : {r} - p-HermansRasson : {stars}')
        else:
            ax.set_title(f'N : {N_events}')
    if with_stats:
        if with_arrow:
            # if N_events > 1000:
            #     color_arrow = 'red'
            # else:
            #     color_arrow = 'red'
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



#########################
#########################
#########################


# event_types = ['spindles','slowwaves'] # run keys for spindles and slow waves
# event_types_titles = {'spindles':'Spindles','slowwaves':'Slow-Waves'} # understandable labels for spindles and slowwaves

# cycles_ratios = get_cycles_ratios(run_keys) # get cycles ratio for the phase transition in polar plots





# # POOLED MERGE

# ncols = 2

# colors = {'spindles':'dimgrey', 'slowwaves':'forestgreen'}

# if save_article:
#     chan_loop = ['Fz']
# else:
#     chan_loop = channels_events_select

# for chan in chan_loop:
#     fig, axs = plt.subplots(ncols=ncols, figsize = (15,7), constrained_layout = True, subplot_kw=dict(projection = 'polar'))

#     for c, ev in enumerate(['spindles','slowwaves']):
#         ax = axs[c]

#         ev_title = event_types_titles[ev]

#         ratio = get_respi_ratio(subject = '*', stage = stage, ratio_df = cycles_ratios)

#         angles = load_angles(subject = '*' , event = ev, cooccuring = '*', speed = '*', chan = chan, half = '*')

#         color = colors[ev]
#         circular_plot_angles(angles, color=color, ax=ax, ratio_plot = ratio, with_title = with_title, with_arrow = True, with_rticks = True)
#         ts_label = timestamps_labels[ev]
#         if save_article:
#             title = f'{event_types_titles[ev]} \n N : {angles.size}'
#         else:
#             title = f'{ev} {ts_label} \n' + ax.get_title()
#         ax.set_title(title, fontsize = 15, y = 1.1)  

#     if save_article:
#         fig.savefig(save_folder / f'polar_plot_pooled_{chan}{extension}', dpi = dpis, bbox_inches = 'tight')
#     else:
#         fig.savefig(save_folder / 'global' / f'polar_plot_pooled_{chan}{extension}', dpi = dpis, bbox_inches = 'tight')
#     plt.close()


# # POOLED WITH SPINDLE SPEED QUESTION

# dict_figure = {
#     'spindles':{'SS':{'pos':0, 'color':None},'FS':{'pos':1, 'color':'skyblue'}},
#     'slowwaves':{'pos':2, 'color':'forestgreen'},
# }

# ncols = 2

# if save_article:
#     chan_loop = ['Fz']
# else:
#     chan_loop = channels_events_select
    
# for chan in chan_loop:
#     fig, axs = plt.subplots(ncols=ncols, figsize = (15,7), constrained_layout = True, subplot_kw=dict(projection = 'polar'))

#     for ev in ['spindles']:
#         ev_title = event_types_titles[ev]

#         ratio = get_respi_ratio(subject = '*', stage = stage, ratio_df = cycles_ratios)

#         if ev == 'spindles':
#             for speed, speed_title in zip(['SS','FS'],['Slow','Fast']):
#                 angles = load_angles(subject = '*' , event = ev, cooccuring = '*', speed = speed, chan = chan, half = '*')

#                 if angles.size == 0:
#                     continue

#                 pos = dict_figure[ev][speed]['pos']
#                 color = dict_figure[ev][speed]['color']
#                 ax = axs[pos]
#                 circular_plot_angles(angles, color=color, ax=ax, ratio_plot = ratio, with_title = with_title, with_arrow = True, with_rticks = True)
#                 ts_label = timestamps_labels[ev]
#                 if save_article:
#                     title = f'{speed_title} {ev} \n N : {angles.size}'
#                 else:
#                     title = f'{ev} {ts_label} - {speed} \n' + ax.get_title()
#                 ax.set_title(title, fontsize = 15, y = 1.1)  

#         elif ev == 'slowwaves':

#             angles = load_angles(subject = '*' , event = ev,cooccuring = '*', speed = speed, chan = chan, half = '*')

#             if angles.size == 0:
#                 continue

#             pos = dict_figure[ev]['pos']

#             ax = axs[pos]
#             color = dict_figure[ev]['color']
#             circular_plot_angles(angles, color=color, ax=ax, ratio_plot = ratio, with_title = with_title, with_arrow = True, with_rticks = True)
#             ts_label = timestamps_labels[ev]
#             if save_article:
#                 title = f'{event_types_titles[ev]} \n N : {angles.size}'
#             else:
#                 title = f'{ev} {ts_label} - {speed} \n' + ax.get_title()
#             ax.set_title(title, fontsize = 15, y = 1.1)
    
#     if save_article:
#         fig.savefig(save_folder / f'polar_plot_pooled_speed_{chan}{extension}', dpi = dpis, bbox_inches = 'tight')
#     else:
#         fig.savefig(save_folder / 'global' / f'polar_plot_pooled_speed_{chan}{extension}',  dpi = dpis, bbox_inches = 'tight')
        
#     plt.close()


# # POOLED WITH SPINDLE SPEED QUESTION and ALL CHANNEL QUESTION
# chan_loop = channels_events_select

# nrows = 3
# ncols = len(chan_loop)
# ev_looped = ['Slow spindles','Fast spindles','Slow-Waves']
# ev_load = {'Slow spindles':'spindles_SS','Fast spindles':'spindles_FS','Slow-Waves':'slowwaves'}
# colors = {'Slow spindles':None, 'Fast spindles':'skyblue', 'Slow-Waves':'forestgreen'}

# fig, axs = plt.subplots(nrows = nrows, ncols=ncols, figsize = (19,7), constrained_layout = True, subplot_kw=dict(projection = 'polar'))

# for c, chan in enumerate(chan_loop):
#     for r, ev_loop in enumerate(ev_looped):
#         ax = axs[r,c]

#         color = colors[ev_loop]

#         if not ev_loop == 'Slow-Waves':
#             load, speed = ev_load[ev_loop].split('_')
#         else:
#             load = ev_load[ev_loop]

#         ev_title = event_types_titles[ev]
        
#         ratio = get_respi_ratio(subject = '*', stage = stage, ratio_df = cycles_ratios)

#         angles = load_angles(subject = '*' , event = load, cooccuring = '*', speed = speed, chan = chan, half = '*')

#         circular_plot_angles(angles, color=color, ax=ax, ratio_plot = ratio, with_title = with_title, with_arrow = True, with_rticks = False, polar_ticks = 'light', lw = 6)

#         ts_label = timestamps_labels[ev]

#         if save_article:
#             if r == 0:
#                 title = f'{chan} \n {ev_loop} \n N : {angles.size}'
#             else:
#                 title = f'{ev_loop} \n N : {angles.size}'
#         else:
#             if r == 0:
#                 title = f'{chan} \n {ev_loop} \n' + ax.get_title()
#             else:
#                 title = ax.get_title()


#         ax.set_title(title, fontsize = 15, y = 1.1)  

    
#     if save_article:
#         fig.savefig(save_folder / f'polar_plot_pooled_speed_all_chans{extension}', dpi = dpis, bbox_inches = 'tight')
#     else:
#         fig.savefig(save_folder / 'global' / f'polar_plot_pooled_speed_all_chans{extension}',  dpi = dpis, bbox_inches = 'tight')
        
#     plt.close()


# # POOLED WITH SPINDLE SPEED and HALF NIGHT QUESTION

# pos = {'*_spindles_*_SS_firsthalf_Fz':[0,0],
#        '*_spindles_*_SS_secondhalf_Fz':[1,0],
#        '*_spindles_*_FS_firsthalf_Fz':[0,1],
#        '*_spindles_*_FS_secondhalf_Fz':[1,1],
#        '*_slowwaves_*_firsthalf_Fz':[0,2],
#        '*_slowwaves_*_secondhalf_Fz':[1,2]
#       }
                    
# color = {'*_spindles_*_SS_firsthalf_Fz':None,
#        '*_spindles_*_SS_secondhalf_Fz':'skyblue',
#        '*_spindles_*_FS_firsthalf_Fz':None,
#        '*_spindles_*_FS_secondhalf_Fz':'skyblue',
#        '*_slowwaves_*_firsthalf_Fz':'forestgreen',
#        '*_slowwaves_*_secondhalf_Fz':'limegreen'
#       }

# nrows = 2                     
# ncols = 3

# if save_article:
#     chan_loop = ['Fz']
# else:
#     chan_loop = channels_events_select
    
# for chan in chan_loop:
#     fig, axs = plt.subplots(nrows = nrows, ncols=ncols, figsize = (15,7), constrained_layout = True, subplot_kw=dict(projection = 'polar'))

#     for ev in ['spindles','slowwaves']:
#         ev_title = event_types_titles[ev]

#         ratio = get_respi_ratio(subject = '*', stage = stage, ratio_df = cycles_ratios)
        
#         for half, half_title in zip(['firsthalf','secondhalf'],['first half','second half']):

#             if ev == 'spindles':
#                 for speed, speed_title in zip(['SS','FS'],['Slow','Fast']):
#                     angles = load_angles(subject = '*' , event = ev, cooccuring = '*', speed = speed, chan = chan, half = half)

#                     if angles.size == 0:
#                         continue
                    
#                     key_plot = f'*_{ev}_*_{speed}_{half}_{chan}'
#                     color_rc = color[key_plot]
#                     pos_rc = pos[key_plot]
                    
#                     ax = axs[pos_rc[0], pos_rc[1]]
                    
#                     circular_plot_angles(angles, color=color_rc, ax=ax, ratio_plot = ratio, with_title = with_title, with_arrow = True, with_rticks = True)
#                     ts_label = timestamps_labels[ev]
#                     if save_article:
#                         title = f'{speed_title} {ev} ({half_title}) \n N : {angles.size}'
#                     else:
#                         title = f'{ev} {ts_label} - {speed} - {half_title} \n' + ax.get_title()
#                     ax.set_title(title, fontsize = 15, y = 1.1)  

#             elif ev == 'slowwaves':

#                 angles = load_angles(subject = '*' , event = ev,cooccuring = '*', speed = speed, chan = chan, half = half)

#                 if angles.size == 0:
#                     continue
                
#                 key_plot = f'*_{ev}_*_{half}_{chan}'
#                 color_rc = color[key_plot]
#                 pos_rc = pos[key_plot]

#                 ax = axs[pos_rc[0], pos_rc[1]]

#                 circular_plot_angles(angles, color=color_rc, ax=ax, ratio_plot = ratio, with_title = with_title, with_arrow = True, with_rticks = True)
#                 ts_label = timestamps_labels[ev]
#                 if save_article:
#                     title = f'{event_types_titles[ev]} ({half_title}) \n N : {angles.size}'
#                 else:
#                     title = f'{ev} {ts_label} - {speed} - {half_title} \n' + ax.get_title()
#                 ax.set_title(title, fontsize = 15, y = 1.1)
    
#     if save_article:
#         fig.savefig(save_folder / f'polar_plot_pooled_speed_halfnight_{chan}{extension}', dpi = dpis, bbox_inches = 'tight')
#     else:
#         fig.savefig(save_folder / 'global' / f'polar_plot_pooled_speed_halfnight_{chan}{extension}',  dpi = dpis, bbox_inches = 'tight')
        
#     plt.close()


    
    
    
# # SUBJECT
def polar_plots_individuals_chan_fig(chan, **p):

    cycles_ratios = get_cycles_ratios(run_keys) # get cycles ratio for the phase transition in polar plots
    save_article = p['save_article']
    univals = p['univals']
    with_stats = p['with_stats']
    bins = p['bins']
    seed = p['seed']
    stage = 'N2'

    if save_article:
        save_folder = article_folder 
        extension = '.tif'
        dpis = 300
        with_title = False
    else:
        save_folder = base_folder / 'results' / 'events_coupling_figures'
        extension = '.tif'
        dpis = 300
        with_title = True
            
    colors = {'spindles':None , 'slowwaves':'forestgreen'}

    for event_type in ['spindles','slowwaves']:
        color = colors[event_type]

        nrows = 4
        ncols = 5

        subjects_array = np.array(run_keys).reshape(nrows, ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize = (20,20), constrained_layout = True, subplot_kw=dict(projection = 'polar'))
        
        if not save_article:
            fig.suptitle(f'{event_type} polar distributions along respiration phase')

        for r in range(nrows):
            for c in range(ncols):
                ax = axs[r,c]
                subject = subjects_array[r,c]

                ratio = get_respi_ratio(subject = subject, stage = stage, ratio_df = cycles_ratios)

                angles = load_angles(subject = subject , event = event_type, cooccuring = '*', speed = '*', chan = chan, half = '*')

                if angles.size == 0:
                    continue 

                circular_plot_angles(angles, 
                                    color=color,
                                     ax=ax, 
                                     ratio_plot = ratio, 
                                     with_title = True, 
                                     with_arrow = True, 
                                     with_rticks = True,
                                    polar_ticks = 'light',
                                    lw = 6,
                                    univals=univals,
                                    with_stats=with_stats,
                                    seed=seed
                                     )

                title = ax.get_title() + f'\n {subject}'
                ax.set_title(title, fontsize = 9)  
                
        if not save_article:
            fig.savefig(save_folder / 'subjects' / f'polar_plot_individual_{chan}_{event_type}{extension}',  dpi = dpis, bbox_inches = 'tight')
        else:
            fig.savefig(save_folder / f'polar_plot_individual_{chan}_{event_type}{extension}',  dpi = dpis, bbox_inches = 'tight')
            
        plt.close()
    return xr.Dataset()

def test_polar_plots_individuals_chan_fig():
    ds = polar_plots_individuals_chan_fig('Fz', **events_coupling_figs_params)
    print(ds)

polar_plots_individuals_chan_fig_job = jobtools.Job(precomputedir, 
                                                              'polar_plots_individuals_chan_fig', 
                                                              events_coupling_figs_params, 
                                                              polar_plots_individuals_chan_fig)
jobtools.register_job(polar_plots_individuals_chan_fig_job)


# CHAN + SUBS + SPEED POOLED WITH CO-OCCUR vs NON CO-OCCTUR

def spindles_pool_chan_sub_speed_q_cooccur_fig(key, **p):
    cycles_ratios = get_cycles_ratios(run_keys) # get cycles ratio for the phase transition in polar plots
    ratio = get_respi_ratio(subject = '*', stage = 'N2', ratio_df = cycles_ratios)
    save_article = p['save_article']
    univals = p['univals']
    with_stats = p['with_stats']
    bins = p['bins']
    seed = p['seed']

    if save_article:
        save_folder = article_folder 
        extension = '.tif'
        dpis = 300
        with_title = False
    else:
        save_folder = base_folder / 'results' / 'events_coupling_figures'
        extension = '.tif'
        dpis = 300
        with_title = True
        
    ncols = 2
    loads = ['cooccur','notcooccur']
    colors = ['tab:blue','g']

    fig, axs = plt.subplots(ncols=ncols, figsize = (12,5), constrained_layout = True, subplot_kw=dict(projection = 'polar'))

    for c in range(ncols):
        ax = axs[c]

        color = colors[c]

        
        angles = load_angles(subject = '*' , event = 'spindles', cooccuring = loads[c], speed = '*', chan = '*', half = '*')
        N = angles.size

        circular_plot_angles(angles,
                             color=color,
                             ax=ax,
                             ratio_plot = ratio,
                             with_title = with_title,
                             with_arrow = True,
                             with_rticks = False, 
                             polar_ticks = 'light',
                             lw = 6,
                            univals=univals,
                            with_stats=with_stats,
                            seed=seed)

        if save_article:
            title = f'Spindles \n {loads[c]} \n N : {N}'
        else:
            title = f'Spindles \n {loads[c]} \n' + ax.get_title()



        ax.set_title(title, fontsize = 15, y = 1.1)  


    if save_article:
        fig.savefig(save_folder / f'polar_plot_pooled_speed_all_chans{extension}', dpi = dpis, bbox_inches = 'tight')
    else:
        fig.savefig(save_folder / 'global' / f'polar_plot_pooled_speed_chans_subs_{extension}',  dpi = dpis, bbox_inches = 'tight')

    plt.close()
    return xr.Dataset()

def test_spindles_pool_chan_sub_speed_q_cooccur_fig():
    ds = spindles_pool_chan_sub_speed_q_cooccur_fig('none', **events_coupling_figs_params)
    print(ds)

spindles_pool_chan_sub_speed_q_cooccur_fig_job = jobtools.Job(precomputedir, 
                                                              'spindles_pool_chan_sub_speed_q_cooccur_fig', 
                                                              events_coupling_figs_params, 
                                                              spindles_pool_chan_sub_speed_q_cooccur_fig)
jobtools.register_job(spindles_pool_chan_sub_speed_q_cooccur_fig_job)




# COMPUTE
def compute_all():
    # run_keys = [('none',)]
    
    # jobtools.compute_job_list(spindles_pool_chan_sub_speed_q_cooccur_fig_job, run_keys, force_recompute=True, engine='slurm',
    #                           slurm_params={'cpus-per-task':'2', 'mem':'5G', },
    #                           module_name='events_coupling_figs',
    #                           )

    run_keys = [(chan,) for chan in chans_events_detect]
    jobtools.compute_job_list(polar_plots_individuals_chan_fig_job, run_keys, force_recompute=True, engine='slurm',
                              slurm_params={'cpus-per-task':'2', 'mem':'5G', },
                              module_name='events_coupling_figs',
                              )

if __name__ == '__main__':
    # test_spindles_pool_chan_sub_speed_q_cooccur_fig()
    # test_polar_plots_individuals_chan_fig()
    compute_all()