import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from params import eeg_chans, eeg_mono_chans, timestamps_labels


subject = 'S1'
start = 20000
stop = 21000


chans = {'sp':eeg_chans, 'sw':eeg_mono_chans}
event_types = ['sp','sw']
event_types_clean = {'sp':'spindles','sw':'slowwaves'}
da_type = {'sp':'bipol','sw':'reref'}
colors = {'sp':'r', 'sw':'k'}

events_df = {}
events_da = {}
for event_type in event_types:
    events_da[event_type] =  xr.open_dataarray(f'../preproc/{subject}_{da_type[event_type]}.nc').sel(chan = chans[event_type], time = slice(start,stop))
    events_df[event_type] = pd.read_excel(f'../event_detection/{subject}_{event_types_clean[event_type]}_reref_yasa.xlsx', index_col = 0)

time = events_da['sp'].coords['time'].values

shift = -100 # shifting in microvolts

fig, axs = plt.subplots(nrows = 2, figsize = (20,15), sharex = True, sharey = True)
fig.subplots_adjust(hspace = 0)

for subplot, event_type in enumerate(event_types):
    events = events_df[event_type]
    channels = chans[event_type]
    ax = axs[subplot]

    for i, channel in enumerate(channels):
        event_chan = events[events['Channel'] == channel]
        event_times = event_chan[timestamps_labels[event_type]].values
        event_sliced = event_times[(event_times >= start) & (event_times < stop)]

        sig_raw = events_da[event_type].sel(chan = channel).values
        sig = sig_raw + i * shift

        ax.plot(time, sig, linewidth = 0.8, label = channel)
        ax.plot(event_sliced, sig[np.where(np.isin(time, event_sliced))[0]], 'o', color = colors[event_type])
        ax.set_ylabel(da_type[event_type])

    ax.legend()
    
plt.show()