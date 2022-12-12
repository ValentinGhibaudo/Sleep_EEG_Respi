import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from params import eeg_mono_chans

subject = 'S1'

da = xr.open_dataarray(f'../preproc/{subject}_reref.nc')
spindles = pd.read_excel(f'../event_detection/{subject}_spindles.xlsx', index_col = 0)
slow_waves = pd.read_excel(f'../event_detection/{subject}_slowwaves.xlsx', index_col = 0)

da_eeg = da.sel(chan = eeg_mono_chans)
time = da_eeg.coords['time'].values

spindles_peaks = (spindles['Peak'].values * srate).astype(int)
slow_waves_peaks = (slow_waves['NegPeak'].values * srate).astype(int)

shift = -5 # shifting in microvolts

fig, ax = plt.subplots(figsize = (20,15))

for i, channel in enumerate(eeg_mono_chans):
    sig_raw = da.sel(chan = channel).values
    sig = sig_raw - i * shift
    ax.plot(time, sig, linewidth = 0.6, label = channel)
    ax.plot(time[spindles_peaks], sig[spindles_peaks], 'o', color = 'r', label = 'spindle')
    ax.plot(time[slow_waves_peaks], sig[slow_waves_peaks], 'o', color = 'g', label = 'slow-wave')
ax.set_xlim(0,30)
ax.legend()
    
plt.show()