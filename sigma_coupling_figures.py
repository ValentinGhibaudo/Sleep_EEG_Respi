import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from params import patients, dpis

for patient in patients:
    print(patient)
    phase_freqs = xr.open_dataarray(f'../sigma_coupling/{patient}_phase_freq_sigma.nc') # 5 D xarray : chan * zscoring * cycle * freq * time
    print(phase_freqs.coords['cycle'].size)
    phase_freq_mean_cycle = phase_freqs.mean('cycle')
    del phase_freqs
    phase_freq_mean_cycle.sel(zscoring = 'raw').plot.pcolormesh(x = 'point', y = 'freq', col = 'chan', col_wrap = 4, add_colorbar = True)
    plt.savefig(f'../sigma_coupling_figures/{patient}_phase_freq.tif', bbox_inches = 'tight', dpi = dpis)
    plt.close()


