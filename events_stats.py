import pandas as pd
import seaborn as sns
from params import patients, dpis, interesting_variables
import matplotlib.pyplot as plt

save = True

event_types = ['sp','sw'] # types of event = spindles and slow waves
event_types_loads = {'sp':'spindles','sw':'slowwaves'} # label to load
event_types_titles = {'sp':'Spindles','sw':'Slow-Waves'} # clean label for titles
events_tables = {'sp':[],'sw':[]} # prepare lists for event types to pool events from all subjects

for patient in patients:
    for event_type in event_types:
        events = pd.read_excel(f'../event_detection/{patient}_{event_types_loads[event_type]}.xlsx',index_col = 0) # load events of the subject
        events.insert(0, 'subject', patient) # add subject label at col 0
        events_tables[event_type].append(events) # add the dataframe of the subject to a list

events_df = {event_type:pd.concat(events_tables[event_type]) for event_type in event_types} # pool all dataframes from subjects

chan_stats_concat = {} 
stage_stats_concat =  {}

for event_type in event_types:
    df = events_df[event_type]

    N_by_chan = df['Channel'].value_counts(sort = False).to_frame().T # number of events by channel
    N_by_chan.insert(0 , 'Normalization', 'no')
    N_by_chan.insert(0, 'event', event_types_titles[event_type])
    prop_by_chan = df['Channel'].value_counts(normalize=True, sort = False).to_frame().T # proportion of events by channel
    prop_by_chan.insert(0 , 'Normalization', 'yes')
    prop_by_chan.insert(0, 'event', event_types_titles[event_type])

    N_by_stage = df['Stage_Letter'].value_counts(sort = False).to_frame().T # number of events by stage 
    N_by_stage.insert(0 , 'Normalization', 'no')
    N_by_stage.insert(0, 'event', event_types_titles[event_type])
    prop_by_stage = df['Stage_Letter'].value_counts(normalize=True, sort = False).to_frame().T # proportion of events by stage
    prop_by_stage.insert(0 , 'Normalization', 'yes')
    prop_by_stage.insert(0, 'event', event_types_titles[event_type])

    chan_stats_concat[event_type] = pd.concat([N_by_chan, prop_by_chan]) # concat Number and Proportion by chan
    stage_stats_concat[event_type]  = pd.concat([N_by_stage, prop_by_stage]) # concat Number and Proportion by stage

chan_stats = pd.concat([chan_stats_concat['sp'], chan_stats_concat['sw']]).reset_index(drop = True) # concat both events stats for chans
stage_stats = pd.concat([stage_stats_concat['sp'], stage_stats_concat['sw']]).reset_index(drop = True) # concat both events stats for stages

if save:
    for df, name in zip([chan_stats, stage_stats],['event_by_chan','event_by_stage']):
        df.to_excel(f'../events_stats/{name}.xlsx')
    for event_type in event_types:
        events_df[event_type].describe()[interesting_variables[event_type]].to_excel(f'../events_stats/{event_type}_description.xlsx') # save estimators for the event type
        events_df[event_type].groupby('Channel').mean(numeric_only=True)[interesting_variables[event_type]].to_excel(f'../events_stats/{event_type}_gby_chan.xlsx') # save estimators for chan effet on the event
        events_df[event_type].groupby('Stage_Letter').mean(numeric_only=True)[interesting_variables[event_type]].to_excel(f'../events_stats/{event_type}_gby_stage.xlsx') # save estimators for stage effet on the event

fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (15,8), constrained_layout = True) # barplots of proportions of events by chan or by stage
for col, df in enumerate([chan_stats, stage_stats]):
    for row, event_type in enumerate(event_types):
        if col == 0:
            title = 'Events by channel'
        else:
            title = 'Events by stage'
        if row == 1:
            title = ''
        ax = axs[row, col]
        mask = (df['event'] == event_types_titles[event_type]) & (df['Normalization'] == 'yes')
        df_plot = df[mask]
        df_plot.plot.bar(ax=ax)
        ax.set_xlabel(event_types_titles[event_type])
        ax.set_xticks([])
        ax.set_ylabel('Proportion')
        ax.set_title(title)

if save:
    plt.savefig(f'../events_stats/events_stats.tif', format = 'tif', dpi = dpis, bbox_inches = 'tight')

plt.close()


for event_type in event_types:
    df_boxplot = events_df[event_type]
    fig, axs = plt.subplots(nrows = 2, ncols = len(interesting_variables[event_type]), figsize = (20,5), constrained_layout = True) # boxplot effects of stage or chan on events params
    fig.suptitle(f'{event_types_titles[event_type]} characteristics')
    for col, outcome in enumerate(interesting_variables[event_type]):
        for row, predictor in enumerate(['Channel','Stage_Letter']):
            ax = axs[row, col]
            sns.boxplot(data = df_boxplot, x = predictor, y = outcome, ax=ax)
            if predictor == 'Channel':
                ax.tick_params(axis='x', rotation=90)
    if save:
        plt.savefig(f'../events_stats/{event_type}_boxplot.tif', format = 'tif', dpi = dpis, bbox_inches = 'tight')
    plt.close()


    


    

