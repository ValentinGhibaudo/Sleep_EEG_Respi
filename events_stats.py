import pandas as pd
import seaborn as sns
from params import subjects, dpis, interesting_variables, encoder_events
import matplotlib.pyplot as plt


event_types = ['sp','sw'] # types of event = spindles and slow waves
event_types_loads = {'sp':'spindles','sw':'slowwaves'} # label to load
event_types_titles = {'sp':'Spindles','sw':'Slow-Waves'} # clean label for titles
events_tables = {'sp':[],'sw':[]} # prepare lists for event types to pool events from all subjects

for subject in subjects:
    for event_type in event_types:
        events = pd.read_excel(f'../event_detection/{subject}_{event_types_loads[event_type]}_reref_{encoder_events}.xlsx',index_col = 0) # load events of the subject
        events.insert(0, 'subject', subject) # add subject label at col 0
        events_tables[event_type].append(events) # add the dataframe of the subject to a list

events_df = {event_type:pd.concat(events_tables[event_type]) for event_type in event_types} # pool all dataframes from subjects

for event_type in event_types:
    events_df[event_type].describe()[interesting_variables[event_type]].to_excel(f'../events_stats/{event_type}_description.xlsx') # save estimators for the event type
    events_df[event_type].groupby('Channel').mean(numeric_only=True)[interesting_variables[event_type]].to_excel(f'../events_stats/{event_type}_gby_chan.xlsx') # save estimators for chan effet on the event
    events_df[event_type].groupby('Stage_Letter').mean(numeric_only=True)[interesting_variables[event_type]].to_excel(f'../events_stats/{event_type}_gby_stage.xlsx') # save estimators for stage effet on the event
    
    
    
# FIG : BARPLOT OF PROPORTION OF EVENTS IN STAGES or CHANNELS 
fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (15,8), constrained_layout = True)

for row, event_type in enumerate(event_types):
    for col, predictor in enumerate(['Channel','Stage_Letter']):
        ax = axs[row, col]

        df = events_df[event_type]
        prop = df[predictor].value_counts(normalize=True, sort = False).to_frame().reset_index().rename(columns={'index':predictor, predictor:'Proportion'}) # proportion of events 

        if col == 0:
            title = 'Events by channel'
        else:
            title = 'Events by stage'
        if row == 1:
            title = ''
        ax = axs[row, col]
        if predictor == 'Stage_Letter':
            sns.barplot(data=prop, x = predictor, y = 'Proportion', ax=ax, order=['W','R','N1','N2','N3'])
        else:
            sns.barplot(data=prop, x = predictor, y = 'Proportion', ax=ax)
        ax.set_ylabel(f'Proportion of {event_types_titles[event_type]}')
        ax.set_title(title)

plt.savefig('../events_stats/barplots_events.tif', format = 'tif', dpi = dpis, bbox_inches = 'tight')
plt.close()

# FIG x 2 : BOXPLOT OF PROPORTION OF EFFECT OF CHANNEL or STAGE ON THE EVENTS FEATURES 
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

    plt.savefig(f'../events_stats/{event_type}_boxplot.tif', format = 'tif', dpi = dpis, bbox_inches = 'tight')
    plt.close()


    


    

