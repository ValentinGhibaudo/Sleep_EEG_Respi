import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
from params import *
from configuration import *
import jobtools
from rsp_detection import resp_tag_job

"""
This scripts :
- compute stats and plot figs of respiration features according to sleep stages
- compute stats of number of cycle spindled or slowwaved by subject by stage
- compute mean cycles ratios by subject by stage for polar plots of events_coupling_figs.py
"""

# JOB STATS OF NUMBER OF RESP CYCLES WITH EVENTS INSIDE
def get_stats_resp_cycles_with_events(run_key, **p):
    features = resp_tag_job.get(run_key).to_dataframe()
    n_tot = features.shape[0] # total number of resp cycles considered (after cleaning)
    tot_spindled = features['Spindle_Tag'].sum() # total number of resp cycles with at least of spindle inside
    prop_spindled = tot_spindled / n_tot # proportion of resp cycles with at least one spindle inside
    tot_slowwaved = features['SlowWave_Tag'].sum() # total number of resp cycles with at least one slowwave inside
    prop_slowwaved = tot_slowwaved / n_tot # proportion of resp cycles with at least one slowwave inside
    data = [run_key, n_tot, tot_spindled, prop_spindled, tot_slowwaved, prop_slowwaved]
    columns = ['subject', 'n cycles', 'n spindled', 'proportion spindled', 'n slowwaved', 'proportion slowwaved']
    df_return = pd.Series(data=data, index = columns).to_frame().T
    return xr.Dataset(df_return)

stats_resp_event_job = jobtools.Job(precomputedir, 'stats_resp_cycles_with_event', resp_stats_params, get_stats_resp_cycles_with_events)
jobtools.register_job(stats_resp_event_job)

def test_get_stats_resp_cycles_with_events():
    run_key = 'S1'
    n_cycles_events = get_stats_resp_cycles_with_events(run_key, **resp_stats_params).to_dataframe()
    print(n_cycles_events)

# JOB STATS OF NUMBER OF RESP CYCLES WITH EVENTS INSIDE BY STAGE
def get_stats_resp_cycles_with_events_stage(run_key, **p):

    features = resp_tag_job.get(run_key).to_dataframe()
    n_tot = features.shape[0]
    rows_by_stage = []

    for stage in features['sleep_stage'].unique(): # loop on stages to get stats intra stage
        features_stage = features[features['sleep_stage'] == stage]  # mask on resp features of the stage
        n_in_stage = features_stage.shape[0]   # number of resp cycles of the subject considered in this stage
        prop_in_stage = n_in_stage / n_tot  # proportion of resp cycles of the subject considered in this stage
        n_spindled = features_stage['Spindle_Tag'].sum() # number of resp cycles with at least one spindle inside on this stage
        prop_spindled = n_spindled / n_in_stage # proportion of resp cycles with at least one spindle inside on this stage
        n_slowwaved = features_stage['SlowWave_Tag'].sum() # number of resp cycles with at least one slowwave inside on this stage
        prop_slowwaved = n_slowwaved / n_in_stage # proportion of resp cycles with at least one slowwave inside on this stage

        rows_by_stage.append([run_key, stage, n_in_stage , prop_in_stage, n_spindled, prop_spindled, n_slowwaved, prop_slowwaved]) # append previous stats to as a list to a list of rows
    
    columns = ['subject', 'stage', 'n in stage', 'proportion in stage', 'n spindled in stage', 'proportion spindled in stage', 'n slowwaved in stage', 'proportion slowwaved in stage']
    df = pd.Series(data=rows_by_stage, index = columns).to_frame().T # make df of resp cycles number and proportions by stage
    return xr.Dataset(df)

stats_resp_event_stage_job = jobtools.Job(precomputedir, 'stats_resp_cycles_with_event_stage', resp_stats_params, get_stats_resp_cycles_with_events_stage)
jobtools.register_job(stats_resp_event_stage_job)

def test_get_stats_resp_cycles_with_events_stage():
    run_key = 'S1'
    n_cycles_events_stage = get_stats_resp_cycles_with_events_stage(run_key, **resp_stats_params).to_dataframe()
    print(n_cycles_events_stage)





def save_cycle_ratios(run_keys):# COMPUTE MEAN RESP CYCLES RATIO BY SUBJECT BY STAGE (AND APPEND A MEAN VERSION ACROSS SUBJECTS)
    concat = [resp_tag_job.get(run_key).to_dataframe() for run_key in run_keys]
    pooled_features = pd.concat(concat)

    cycle_ratio_by_sub_by_stage = pooled_features.groupby(['subject','sleep_stage'])['cycle_ratio'].mean(numeric_only = True).reset_index()
    mean_cycle_ratio_by_stage = pooled_features.groupby('sleep_stage')['cycle_ratio'].mean(numeric_only = True).reset_index()
    mean_cycle_ratio_by_stage.insert(0, 'subject','mean')
    cycles_ratios = pd.concat([cycle_ratio_by_sub_by_stage, mean_cycle_ratio_by_stage])
    cycles_ratios.to_excel(base_folder / 'results' / 'resp_stats' / 'cycle_ratios.xlsx')
    return None

def save_resp_features(run_keys):
    useful_resp_features = ['subject','cycle_duration','cycle_ratio','inspi_duration','expi_duration']
    concat = []
    for run_key in run_keys:
        df = resp_tag_job.get(run_key).to_dataframe()
        df = df[df['sleep_stage'] == 'N2']
        df.insert(0, 'subject',run_key)
        concat.append(df[useful_resp_features])
    pooled_features_all = pd.concat(concat).groupby('subject').mean(numeric_only = True)
    pooled_features = pooled_features_all.copy().reindex(run_keys)
    pooled_features.loc['Mean',:] = pooled_features_all.mean(axis = 0).values
    pooled_features.loc['SD',:] = pooled_features_all.std(axis = 0).values
    pooled_features.round(2).reset_index().to_excel(base_folder / 'results' / 'resp_stats' / 'resp_features_N2.xlsx', index = False)
    return None



# FIGURES 

# FIG 1 : BARPLOT OF NUMBER OF RESPIRATION CYCLES TOTAL, SPINDLED, SLOWWAVED , ALL SUBJECTS POOLED
def fig_barplot_n_resp_event(run_keys):
    concat = [stats_resp_event_job.get(run_key).to_dataframe() for run_key in run_keys]
    global_stats = pd.concat(concat)
    
    fig, ax = plt.subplots()
    global_stats[['n cycles','n spindled','n slowwaved']].sum().plot.bar(ax=ax)
    ax.set_title('Pooled resp cycles from all subjects')
    for bar in ax.containers:
        ax.bar_label(bar)
        
    fig.savefig(base_folder / 'results' / 'resp_stats' / 'pooling_cycles_N.png', bbox_inches = 'tight')
    plt.close()

    return None

#  FIG 2 : VIOLINPLOTS SEARCHING A POSSIBLE DIFFERENCE OF THE FEATURES OF THE RESP CYCLES WITH AT LEAST ONE EVENT FOUND INSIDE
def fig_violinplot_eventing_effect_on_resp(run_keys):
    concat = [resp_tag_job.get(run_key).to_dataframe() for run_key in run_keys]
    pooled_features = pd.concat(concat)
    
    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (15,8), constrained_layout = True)
    fig.suptitle('Effect of event presence in resp cycles on their features')
    for r, predictor in enumerate(['Spindle_Tag', 'SlowWave_Tag']):
        for c, outcome in enumerate(['cycle_freq', 'inspi_volume', 'expi_volume']):
            ax = axs[r,c]
            sns.violinplot(data=pooled_features,  x= predictor, y = outcome, ax=ax)
    fig.savefig(base_folder / 'results' / 'resp_stats' / 'violin_eventing_effect.png', bbox_inches= 'tight')
    plt.close()
    return None


# FIG 3 :  VIOLINPLOTS THE FEATURES OF THE RESP CYCLES ACCORDING TO SLEEP STAGE
def fig_violinplot_stage_effect_on_resp(run_keys):
    concat = [resp_tag_job.get(run_key).to_dataframe() for run_key in run_keys]
    pooled_features = pd.concat(concat)

    nrows = 4
    ncols = 2
    outcomes = np.array(['cycle_duration','inspi_duration','expi_duration','cycle_ratio','cycle_volume','inspi_volume','expi_volume','second_volume']).reshape(nrows,ncols)
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, constrained_layout = True, figsize = (15,8))
    fig.suptitle('Effect of sleep staging on respiratory features')
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r,c]
            outcome = outcomes[r,c]
            sns.violinplot(data = pooled_features, y = outcome,  x = 'sleep_stage' ,ax=ax)
    fig.savefig(base_folder / 'results' / 'resp_stats' / 'plot_sleep_stage_effect', bbox_inches= 'tight')
    plt.close()
    return None

def compute_all_figs(run_keys):
    fig_barplot_n_resp_event(run_keys)
    fig_violinplot_eventing_effect_on_resp(run_keys)
    fig_violinplot_stage_effect_on_resp(run_keys)



def compute_all():
    # jobtools.compute_job_list(stats_resp_event_job, run_keys, force_recompute=False, engine='loop')
    jobtools.compute_job_list(stats_resp_event_stage_job, run_keys, force_recompute=False, engine='loop')


if __name__ == '__main__':
    # test_get_stats_resp_cycles_with_events() 
    # test_get_stats_resp_cycles_with_events_stage()

    # save_cycle_ratios(run_keys)
    save_resp_features(run_keys)

    # compute_all_figs(run_keys)

