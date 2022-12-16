import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from params import srate, respi_chan

patient = 'S1'

resp_signal = xr.open_dataarray(f'../preproc/{patient}_reref.nc').sel(chan = respi_chan).data
resp_features = pd.read_excel(f'../resp_features/{patient}_resp_features.xlsx', index_col = 0)

markers = {'start':'r','transition':'g'}

fig, ax = plt.subplots(figsize = (15,10))
ax.plot(resp_signal)
for marker in markers.keys():
    ax.plot(resp_features[marker], resp_signal[resp_features[marker]], 'o', color = markers[marker], label = f'{marker}')
ax.legend()
ax.set_title(patient)

plt.show()
