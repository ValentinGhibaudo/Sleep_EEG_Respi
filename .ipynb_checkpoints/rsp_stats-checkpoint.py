import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from params import subjects, stages_labels

"""
This scripts :
- compute stats and plot figs of respiration features according to sleep stages
- compute stats of number of cycle spindled or slowwaved by subject by stage
- compute mean cycles ratios by subject by stage for polar plots of events_coupling_figs.py
"""

destination_folder = f'../resp_stats/' # folder of saving of the figs and tables

concat_rsp_features = [] # this list will concat all rsp features from subjects
rows_global = [] # this list will concat stats of presence of events in a N number of resp cycles
rows_by_stage = [] # this list will concat stats of presence of events in a N number of resp cycles by stage
for subject in subjects: # loop on subjects
    df_subject = pd.read_excel(f'../resp_features/{subject}_resp_features_tagged.xlsx', index_col = 0) # load rsp features tagged by staging and presence of events in resp cycles
    n_tot = df_subject.shape[0] # total number of resp cycles considered (after cleaning)
    tot_spindled = df_subject['Spindle_Tag'].sum() # total number of resp cycles with at least of spindle inside
    prop_spindled = tot_spindled / n_tot # proportion of resp cycles with at least one spindle inside
    tot_slowwaved = df_subject['SlowWave_Tag'].sum() # total number of resp cycles with at least one slowwave inside
    prop_slowwaved = tot_slowwaved / n_tot # proportion of resp cycles with at least one slowwave inside
    rows_global.append([subject, n_tot, tot_spindled, prop_spindled, tot_slowwaved, prop_slowwaved]) # append previous stats to as a list to a list of rows

    for stage in stages_labels: # loop on stages to get stats intra stage
        df_subject_stage = df_subject[df_subject['sleep_stage'] == stage]  # mask on resp features of the stage
        n_in_stage = df_subject_stage.shape[0]   # number of resp cycles of the subject considered in this stage
        prop_in_stage = n_in_stage / n_tot  # proportion of resp cycles of the subject considered in this stage
        n_spindled = df_subject_stage['Spindle_Tag'].sum() # number of resp cycles with at least one spindle inside on this stage
        prop_spindled = n_spindled / n_in_stage # proportion of resp cycles with at least one spindle inside on this stage
        n_slowwaved = df_subject_stage['SlowWave_Tag'].sum() # number of resp cycles with at least one slowwave inside on this stage
        prop_slowwaved = n_slowwaved / n_in_stage # proportion of resp cycles with at least one slowwave inside on this stage
        rows_by_stage.append([subject, stage, n_in_stage , prop_in_stage, n_spindled, prop_spindled, n_slowwaved, prop_slowwaved]) # append previous stats to as a list to a list of rows
        
    
    df_subject.insert(0, 'subject', subject) # insert subject id a first column
    concat_rsp_features.append(df_subject) # append rsp features of the subject to a list
        
global_stats = pd.DataFrame(rows_global, columns = ['subject', 'n cycles', 'n spindled', 'proportion spindled', 'n slowwaved', 'proportion slowwaved']) # make df of global resp cycles number and proportions 
staged_stats = pd.DataFrame(rows_by_stage, columns = ['subject', 'stage', 'n in stage', 'proportion in stage', 'n spindled in stage', 'proportion spindled in stage', 'n slowwaved in stage', 'proportion slowwaved in stage']) # make df of resp cycles number and proportions by stage
pooled_features = pd.concat(concat_rsp_features)

global_stats.to_excel(destination_folder + 'global_rsp_events_stats.xlsx')
staged_stats.to_excel(destination_folder + 'rsp_stage_events_stats.xlsx')


# FIG 1 : BARPLOT OF NUMBER OF RESPIRATION CYCLES TOTAL, SPINDLED, SLOWWAVED , ALL SUBJECTS POOLED
print('FIG 1')
fig, ax = plt.subplots()
global_stats[['n cycles','n spindled','n slowwaved']].sum().plot.bar(ax=ax)
ax.set_title('Pooled resp cycles from all subjects')
plt.savefig(destination_folder + 'pooling_cycles_N', bbox_inches = 'tight')
plt.close()


# FIG 2 : BOXPLOTS OF PROPORTION OF RESPI CYCLES WITH AT LEAST ONE EVENT INSIDE BY STAGE
print('FIG 2')
fig, axs = plt.subplots(ncols = 2, figsize =(15,5))
fig.suptitle('Proportion of events in respiratory cycles by stage')
ax = axs[0]
sns.boxplot(data = staged_stats, x = 'stage', y = 'proportion spindled in stage', ax=ax)
ax = axs[1]
sns.boxplot(data = staged_stats, x = 'stage', y = 'proportion slowwaved in stage', ax=ax)
plt.savefig(destination_folder + 'boxplot_proportion_event_by_stage', bbox_inches= 'tight')
plt.close()


# FIG 3 : VIOLINPLOTS SEARCHING A POSSIBLE DIFFERENCE OF THE FEATURES OF THE RESP CYCLES WITH AT LEAST ONE EVENT FOUND INSIDE
print('FIG 3')
fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (15,8), constrained_layout = True)
fig.suptitle('Effect of event presence in resp cycles on their features')
for r, predictor in enumerate(['Spindle_Tag', 'SlowWave_Tag']):
    for c, outcome in enumerate(['cycle_freq', 'inspi_volume', 'expi_volume']):
        ax = axs[r,c]
        sns.violinplot(data=pooled_features,  x= predictor, y = outcome, ax=ax)
plt.savefig(destination_folder + 'violin_eventing_effect', bbox_inches= 'tight')
plt.close()


# FIG 4 :  VIOLINPLOTS THE FEATURES OF THE RESP CYCLES ACCORDING TO SLEEP STAGE
print('FIG 4')
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
plt.savefig(destination_folder + 'plot_sleep_stage_effect', bbox_inches= 'tight')
plt.close()

for subject in subjects:
    print(subject)
    features_subject = pooled_features[pooled_features['subject'] == subject]
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, constrained_layout = True, figsize = (15,8))
    fig.suptitle(f'Effect of sleep staging on respiratory features in {subject}')
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r,c]
            outcome = outcomes[r,c]
            sns.violinplot(data = features_subject, y = outcome,  x = 'sleep_stage' ,ax=ax)
    plt.savefig(destination_folder + f'{subject}_plot_sleep_stage_effect', bbox_inches= 'tight')
    plt.close()


# COMPUTE MEAN RESP CYCLES RATIO BY SUBJECT BY STAGE (AND APPEND A MEAN VERSION ACROSS SUBJECTS)
cycle_ratio_by_sub_by_stage = pooled_features.groupby(['subject','sleep_stage'])['cycle_ratio'].mean(numeric_only = True).reset_index()
mean_cycle_ratio_by_stage = pooled_features.groupby('sleep_stage')['cycle_ratio'].mean(numeric_only = True).reset_index()
mean_cycle_ratio_by_stage.insert(0, 'subject','mean')
cycles_ratios = pd.concat([cycle_ratio_by_sub_by_stage, mean_cycle_ratio_by_stage])
cycles_ratios.to_excel('../resp_stats/cycle_ratios.xlsx')

