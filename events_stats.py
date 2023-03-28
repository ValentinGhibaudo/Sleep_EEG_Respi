import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from params import *
from configuration import base_folder
from detect_sleep_events import spindles_tag_job, slowwaves_tag_job
from preproc_staging import sleep_stats_job

p = events_stats_params
stage = p['stage']
interesting_variables = p['interesting_variables']

events_df = {}
events_df['spindles'] = pd.concat([spindles_tag_job.get(run_key).to_dataframe() for run_key in run_keys])
events_df['slowwaves'] = pd.concat([slowwaves_tag_job.get(run_key).to_dataframe() for run_key in run_keys])

event_labels = ['spindles','slowwaves']



# FIG : BARPLOT OF PROPORTION OF EVENTS IN CHANNELS 
print('FIG : BARPLOT PROPORTION EVENTS IN CHANS')
fig, axs = plt.subplots(nrows = len(event_labels), figsize = (15,8), constrained_layout = True)
predictor = 'Channel'
for row, event_label in enumerate(event_labels):
    ax = axs[row]

    df = events_df[event_label]
    prop = df[predictor].value_counts(normalize=True, sort = False).to_frame().reset_index().rename(columns={'index':predictor, predictor:'Proportion'}) # proportion of events 

    sns.barplot(data=prop, x = predictor, y = 'Proportion', ax=ax)
    ax.set_ylabel(f'Proportion')
    ax.set_title(f'{event_label} by channel')

fig.savefig(base_folder / 'results' / 'events_stats' / 'barplots_events_by_channel.png', bbox_inches = 'tight')
plt.close()

# FIG : NUMBER OF COOCCURING EVENTS
print('FIG : N COOCCURING EVENTS')
nrows = len(event_labels)
ncols = events_df['spindles']['Channel'].unique().size

fig, axs = plt.subplots(nrows = nrows, ncols =ncols,  figsize = (20,7), constrained_layout = True)
fig.suptitle('Number of cooccuring events', fontsize = 20, y = 1.04)
for row, event_label in enumerate(event_labels):
    for col, chan in enumerate(events_df['spindles']['Channel'].unique()):
        ax = axs[row,col]

        df = events_df[event_label]
        df_plot = df[df['Channel'] == chan]
        df_plot['cooccuring'].value_counts(normalize=False).plot.bar(ax=ax)
        if col == 0:
            ax.set_ylabel(f'N')
        else:
            ax.set_ylabel(None)
        ax.set_title(f'{event_label} in {chan}')
        for bar in ax.containers:
            ax.bar_label(bar)

fig.savefig(base_folder / 'results' / 'events_stats' / 'cooccuring_events.png', bbox_inches = 'tight')
plt.close()


# FIG : NUMBER OF COOCCURING EVENTS
print('FIG : N SLOW VS FAST SPINDLES')
nrows = 4
ncols = 5
subjects_array = np.array(run_keys).reshape(nrows, ncols)
fig, axs = plt.subplots(nrows,ncols, figsize = (20,15), constrained_layout = True)
fig.suptitle('Number of fast vs slow spindles (all chans pooled)', y = 1.04, fontsize = 20)
for r in range(nrows):
    for c in range(ncols):
        ax = axs[r,c]
        sub = subjects_array[r,c]
        spindles_sub = events_df['spindles'][events_df['spindles']['subject'] == sub]
        spindles_sub['Sp_Speed'].value_counts().reindex(['SS','FS']).plot.bar(ax=ax)
        ax.set_ylabel(sub)
        for bar in ax.containers:
            ax.bar_label(bar)

fig.savefig(base_folder / 'results' / 'events_stats' / 'speed_spindles.png', bbox_inches = 'tight')  
plt.close()


nrows = 4
ncols = 5
chan_sel = p['chan']
subjects_array = np.array(run_keys).reshape(nrows, ncols)
fig, axs = plt.subplots(nrows,ncols, figsize = (20,15), constrained_layout = True)
fig.suptitle(f'Number of fast vs slow spindles in {chan_sel}', y = 1.04, fontsize = 20)
for r in range(nrows):
    for c in range(ncols):
        ax = axs[r,c]
        sub = subjects_array[r,c]
        spindles_sub = events_df['spindles'][(events_df['spindles']['subject'] == sub) & (events_df['spindles']['Channel'] == chan_sel)]
        spindles_sub['Sp_Speed'].value_counts().reindex(['SS','FS']).plot.bar(ax=ax)
        ax.set_ylabel(sub)
        for bar in ax.containers:
            ax.bar_label(bar)
fig.savefig(base_folder / 'results' / 'events_stats' / f'speed_spindles_{chan_sel}.png', bbox_inches = 'tight')  
plt.close()




chan_sel = p['chan']
fig, ax = plt.subplots(figsize = (15,5), constrained_layout = True)
fig.suptitle(f'Number of fast vs slow spindles in {chan_sel}', y = 1.04, fontsize = 20)
spindles_sub = events_df['spindles'][(events_df['spindles']['Channel'] == chan_sel)]
spindles_sub['Sp_Speed'].value_counts().reindex(['SS','FS']).plot.bar(ax=ax)
ax.set_ylabel('N')
for bar in ax.containers:
    ax.bar_label(bar)
fig.savefig(base_folder / 'results' / 'events_stats' / f'speed_spindles_{chan_sel}_all.png', bbox_inches = 'tight')  
plt.close()



# FIG : PLOT OF EFFECT OF CHANNEL ON THE EVENTS FEATURES 
print('FIG : CHAN EFFECT ON FEATURES')
for event_label in event_labels:
    df_boxplot = events_df[event_label]
    fig, axs = plt.subplots(nrows = len(interesting_variables[event_label]), figsize = (20,20), constrained_layout = True) # boxplot effects of stage or chan on events params
    fig.suptitle(f'{event_label} characteristics')
    for row, outcome in enumerate(interesting_variables[event_label]):
        ax = axs[row]
        sns.pointplot(data = df_boxplot, x = 'Channel', y = outcome, ax=ax)
        ax.tick_params(axis='x', rotation=90)

    fig.savefig(base_folder / 'results' / 'events_stats' / f'{event_label}_metrics_by_channel.png', bbox_inches = 'tight')
    plt.close()
    
    
# FIG : Density of events by chan
print('FIG : DENSITY')

sleep_stats = pd.concat([sleep_stats_job.get(run_key).to_dataframe() for run_key in run_keys])

def get_stage_duration(sleep_stats, subject, stage):
    return sleep_stats.set_index('subject').loc[subject, stage]

rows = []
for event_label in event_labels:
    events = events_df[event_label]
    for sub in run_keys:
        sub_stage_duration = get_stage_duration(sleep_stats, sub, stage)
        for chan in events['Channel'].unique():
            mask = (events['subject'] == sub) & (events['Stage_Letter'] == stage) & (events['Channel'] == chan)
            n_ev_sub_stage_chan = events[mask].shape[0]
            density_by_stage_by_chan = n_ev_sub_stage_chan / sub_stage_duration

            row = [event_label ,sub, stage, sub_stage_duration, chan , n_ev_sub_stage_chan, density_by_stage_by_chan]
            rows.append(row)
                    
df_density = pd.DataFrame(rows, columns = ['event','subject','stage','stage_duration','chan','N','Density'])
df_density.to_excel(base_folder / 'results' / 'events_stats' / 'density.xlsx')


nrows = len(run_keys)
fig, axs = plt.subplots(nrows, figsize = (20,30), constrained_layout = True)
for row, sub in enumerate(run_keys):
    ax = axs[row]
    density_sub = df_density[df_density['subject'] == sub]
    sns.pointplot(data = density_sub, x = 'chan', y = 'Density', hue = 'event', ax=ax)
    ax.set_ylabel(f'Density in {sub}')
fig.savefig(base_folder / 'results' / 'events_stats' / 'densities.png', bbox_inches = 'tight')  
plt.close()

fig, ax = plt.subplots(figsize = (15,5), constrained_layout = True)
sns.pointplot(data = df_density, x = 'chan', y = 'Density', hue = 'event', ax=ax)
fig.savefig(base_folder / 'results' / 'events_stats' / 'density_all.png', bbox_inches = 'tight')  
plt.close()

# DISTRIBUTIONS OF FREQ OF SPINDLES
print('FIG : DISTRIB SPINDLE FREQS')
nrows = 4
ncols = 5
subjects_array = np.array(run_keys).reshape(nrows, ncols)
fig, axs = plt.subplots(nrows,ncols, figsize = (20,15), constrained_layout = True)
fig.suptitle('Distributions of spindle frequencies', y = 1.04, fontsize = 20)
for r in range(nrows):
    for c in range(ncols):
        ax = axs[r,c]
        sub = subjects_array[r,c]
        spindles_sub = events_df['spindles'][events_df['spindles']['subject'] == sub]
        sns.kdeplot(data = spindles_sub, x = 'Frequency' , hue = 'Channel', ax=ax, bw_adjust = 0.4)
        ax.axvline(x = spindles_freq_threshold[sub], color = 'k')
        ax.set_ylabel(sub)
        ax.set_xlim(12,15)
        if r == (nrows - 1):
            ax.set_xlabel('Frequency [Hz]')
fig.savefig(base_folder / 'results' / 'events_stats' / 'frequencies_spindles_kde.png', bbox_inches = 'tight')  
plt.close()

nrows = 4
ncols = 5
subjects_array = np.array(run_keys).reshape(nrows, ncols)
fig, axs = plt.subplots(nrows,ncols, figsize = (20,15), constrained_layout = True)
fig.suptitle('Distributions of spindle frequencies', y = 1.04, fontsize = 20)
for r in range(nrows):
    for c in range(ncols):
        ax = axs[r,c]
        sub = subjects_array[r,c]
        spindles_sub = events_df['spindles'][events_df['spindles']['subject'] == sub]
        ax.hist(spindles_sub['Frequency'], bins = 100)
        ax.axvline(x = spindles_freq_threshold[sub], color = 'r')
        ax.set_ylabel(sub)
        ax.set_xlim(12,15)
        if r == (nrows - 1):
            ax.set_xlabel('Frequency [Hz]')
fig.savefig(base_folder / 'results' / 'events_stats' / 'frequencies_spindles.png', bbox_inches = 'tight')  
plt.close()


# DISTRIBUTIONS
print('FIG : ALL DISTRIBUTIONS')

ev_features_to_include = {
    'spindles':['Duration', 'Amplitude', 'RMS', 'AbsPower','RelPower', 'Frequency', 'Oscillations'],                        
    'slowwaves':['Duration','ValNegPeak', 'ValPosPeak', 'PTP', 'Slope', 'Frequency']
    }

for event_label in event_labels:
    events = events_df[event_label]
    for subject in run_keys:

        nrows = events['Channel'].unique().size
        ncols = len(ev_features_to_include[event_label])
        fig, axs = plt.subplots(nrows = nrows, 
                                ncols = ncols, 
                                figsize = (20,20),
                                constrained_layout = True)
        
        fig.suptitle(f'{subject} - {event_label}', fontsize = 20, y = 1.04)

        for r, chan in enumerate(events['Channel'].unique()):
            mask = (events['subject'] == sub) & (events['Stage_Letter'] == stage) & (events['Channel'] == chan)
            events_sub_stage_chan = events[mask]
            N = events_sub_stage_chan.shape[0]
            
            for c, metric in enumerate(ev_features_to_include[event_label]):
                ax = axs[r,c]
                ax.hist(events_sub_stage_chan[metric], bins = 50)

                if r == 0:
                    ax.set_title(metric)
                
                if c == 0:
                    ax.set_ylabel(chan)

                if metric == 'Frequency' and event_label == 'spindles':
                    ax.axvline(x = spindles_freq_threshold[subject], color = 'r')

        fig.savefig(base_folder / 'results' / 'events_stats' / f'{subject}_{event_label}_distributions.png', bbox_inches = 'tight')
        plt.close()
        


        # KDEPLOT
        fig, axs = plt.subplots(ncols = len(ev_features_to_include[event_label]), figsize = (20,5), constrained_layout = True)
        fig.suptitle(f'{subject} - {event_label}', fontsize = 20, y = 1.04)

        mask_kde_plot = (events['subject'] == sub) & (events['Stage_Letter'] == stage)
        data_kdeplot =  events[mask_kde_plot]


        for c , metric in enumerate(ev_features_to_include[event_label]):
            ax = axs[c]
            sns.kdeplot(data = data_kdeplot, x = metric , hue = 'Channel', ax=ax, bw_adjust = 0.5)
            if metric == 'Frequency' and event_label == 'spindles':
                ax.axvline(x = spindles_freq_threshold[subject], color = 'r')
            ax.set_title(metric)
        fig.savefig(base_folder / 'results' / 'events_stats' / f'kdeplot_{subject}_{event_label}.png', bbox_inches = 'tight')
        plt.close()


# KDEPLOT POOLED SPINDLES
print('FIG : KDEPLOT POOLED')
fig, ax = plt.subplots(figsize = (15,5), constrained_layout = True)
sns.kdeplot(data = events_df['spindles'], x = 'Frequency' , hue = 'Channel', ax=ax, bw_adjust = 0.5)
mean_freq_tresh = np.mean(np.array([spindles_freq_threshold[run_key] for run_key in run_keys]))
ax.axvline(x = mean_freq_tresh, color = 'k')
ax.set_title(f'Spindles frequency pooled (mean thresh : {round(mean_freq_tresh, 2)} Hz)')
fig.savefig(base_folder / 'results' / 'events_stats' / f'kdeplot_spindles_pooled.png', bbox_inches = 'tight')
plt.close()


            

    

