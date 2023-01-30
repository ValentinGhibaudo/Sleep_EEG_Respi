import pandas as pd
import seaborn as sns
import numpy as np
from params import subjects, dpis, interesting_variables, encoder_events, stages_events_select, spindles_freq_threshold
import matplotlib.pyplot as plt


event_types = ['sp','sw'] # types of event = spindles and slow waves
event_types_loads = {'sp':'spindles','sw':'slowwaves'} # label to load
event_types_titles = {'sp':'Spindles','sw':'Slow-Waves'} # clean label for titles
events_tables = {'sp':[],'sw':[]} # prepare lists for event types to pool events from all subjects

for subject in subjects:
    for event_type in event_types:
        events = pd.read_excel(f'../event_detection/{subject}_{event_types_loads[event_type]}_cooccuring.xlsx',index_col = 0) # load events of the subject
        events.insert(0, 'subject', subject) # add subject label at col 0
        events_tables[event_type].append(events) # add the dataframe of the subject to a list

events_df = {event_type:pd.concat(events_tables[event_type]) for event_type in event_types} # pool all dataframes from subjects

for event_type in event_types:
    events_df[event_type].describe()[interesting_variables[event_type]].to_excel(f'../events_stats/{event_type}_description.xlsx') # save estimators for the event type
    events_df[event_type].groupby('Channel').mean(numeric_only=True)[interesting_variables[event_type]].to_excel(f'../events_stats/{event_type}_gby_chan.xlsx') # save estimators for chan effet on the event
    events_df[event_type].groupby('Stage_Letter').mean(numeric_only=True)[interesting_variables[event_type]].to_excel(f'../events_stats/{event_type}_gby_stage.xlsx') # save estimators for stage effet on the event
    
    
    
# FIG : BARPLOT OF PROPORTION OF EVENTS IN STAGES or CHANNELS 
print('FIG 1')
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

plt.savefig('../events_stats/barplots_events', bbox_inches = 'tight')
plt.close()

# FIG x 2 : BOXPLOT OF PROPORTION OF EFFECT OF CHANNEL or STAGE ON THE EVENTS FEATURES 
print('FIG 2')
event_type_cooccur_label = {'sp':'in_slowwave', 'sw':'sp_inside'}
for event_type in event_types:
    df_boxplot = events_df[event_type]
    fig, axs = plt.subplots(nrows = 3, ncols = len(interesting_variables[event_type]), figsize = (20,15), constrained_layout = True) # boxplot effects of stage or chan on events params
    fig.suptitle(f'{event_types_titles[event_type]} characteristics')
    for col, outcome in enumerate(interesting_variables[event_type]):
        for row, predictor in enumerate(['Channel','Stage_Letter',event_type_cooccur_label[event_type]]):
            ax = axs[row, col]
            sns.boxplot(data = df_boxplot, x = predictor, y = outcome, ax=ax)
            if predictor == 'Channel':
                ax.tick_params(axis='x', rotation=90)

    plt.savefig(f'../events_stats/{event_type}_boxplot', bbox_inches = 'tight')
    plt.close()
    
    
# FIG 3 : Density of spindles by chan by stage
print('FIG 3')
def get_stage_duration(sleep_stats, subject, stage):
    return sleep_stats.set_index('subject').loc[subject, stage]

sleep_stats = pd.read_excel(f'../subject_characteristics/global_sleep_stats_{encoder_events}.xlsx', index_col = 0)
rows = []
evs = ['spindles','slowwaves']
for ev in evs:
    for sub in subjects:
        events = pd.read_excel(f'../event_detection/{sub}_{ev}_reref_{encoder_events}.xlsx', index_col = 0)
        n_ev_total = events.shape[0]
        for stage in ['N2','N3']:
            events_stage = events[events['Stage_Letter'] == stage]
            n_ev_stage = events_stage.shape[0]
            stage_duration = get_stage_duration(sleep_stats, sub, stage)
            density_by_stage = n_ev_stage / stage_duration
            for chan in events_stage['Channel'].unique():
                events_stage_chan = events_stage[events_stage['Channel'] == chan]
                n_ev_stage_chan = events_stage_chan.shape[0]
                density_by_stage_by_chan = n_ev_stage_chan / stage_duration

                row = [ev ,sub, stage, chan, n_ev_total, stage_duration, n_ev_stage, density_by_stage, n_ev_stage_chan, density_by_stage_by_chan]
                rows.append(row)
                    
df_density = pd.DataFrame(rows, columns = ['event','subject','stage','chan','n_events_total','stage_duration',
                                           'n_events_stage','density_by_stage','n_events_stage_chan','density_by_stage_by_chan'])

df_density.to_excel('../events_stats/density.xlsx')

order = ['Fp2','Fp1','C3','C4','Fz','Cz','Pz','T4','T3','O1','O2']

fig, axs = plt.subplots(nrows =2, figsize = (15,10), constrained_layout = True)
for r, ev in enumerate(evs):
    df_plot = df_density[df_density['event'] == ev]
    ax = axs[r]
    sns.pointplot(data = df_plot , x = 'chan', y= 'density_by_stage_by_chan', hue = 'stage', ax=ax, order = order)
    ax.set_title(ev)
    # ax.set_ylim(0,4)
    ax.set_ylabel(f"Density of {ev} by minute / stage / chan")

plt.savefig('../events_stats/density_events')
plt.close()





# DISTRIBUTIONS
print('FIG 4')
resp_features_to_include = ['cycle_duration','inspi_duration','expi_duration','cycle_freq','cycle_ratio','inspi_amplitude','expi_amplitude','inspi_volume','expi_volume']
ev_features_to_include = {'sp':['Duration', 'Amplitude', 'RMS', 'AbsPower','RelPower', 'Frequency', 'Oscillations', 'Symmetry'],
'sw':['Duration','ValNegPeak', 'ValPosPeak', 'PTP', 'Slope', 'Frequency','PhaseAtSigmaPeak', 'ndPAC']}

for event_type in event_types:
    df_events = events_df[event_type]
    df_events_staged = df_events[df_events['Stage_Letter'].isin(stages_events_select)]
    
    for subject in subjects:
        print(subject)
        df_events_staged_subject = df_events_staged[df_events_staged['subject'] == subject]
        N = df_events_staged_subject.shape[0]

        nrows = 2
        ncols = 4
        ev_features_to_include_array = np.array(ev_features_to_include[event_type]).reshape(nrows,ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize = (20,10), constrained_layout = True)
        fig.suptitle(f'{subject} - N : {int(N)} {event_types_titles[event_type]}', fontsize = 20, y = 1.05)
        for r in range(nrows):
            for c in range(ncols):
                ax = axs[r,c]
                metric = ev_features_to_include_array[r,c]
                ax.hist(df_events_staged_subject[metric], bins = 100)
                ax.set_title(metric)
                if metric == 'Frequency' and event_type == 'sp':
                    ax.axvline(x = spindles_freq_threshold[subject], color = 'r')
        plt.savefig(f'../events_stats/{subject}_{event_type}_distributions', bbox_inches = 'tight')
        plt.close()
        
        for predictor in ['Stage_Letter','Channel']:
            fig, axs = plt.subplots(nrows, ncols, figsize = (20,10), constrained_layout = True)
            fig.suptitle(f'{subject} - N : {int(N)} {event_types_titles[event_type]}', fontsize = 20, y = 1.05)
            for r in range(nrows):
                for c in range(ncols):
                    ax = axs[r,c]
                    metric = ev_features_to_include_array[r,c]
                    sns.kdeplot(data = df_events_staged_subject, x = metric , hue = predictor, ax=ax)
                    if metric == 'Frequency' and event_type == 'sp':
                        ax.axvline(x = spindles_freq_threshold[subject], color = 'r')
                    ax.set_title(metric)
            plt.savefig(f'../events_stats/{subject}_{event_type}_kde_{predictor}', bbox_inches = 'tight')
            plt.close()



            

    

