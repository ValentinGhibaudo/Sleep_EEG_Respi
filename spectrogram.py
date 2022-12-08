import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from params import srate, subjects
import yasa

for subject in subjects:
    print(subject)
    chan_sel = ['C4-T4','C3-T3'] # sigs from these chan will be averaged
    data = xr.open_dataarray(f'../preproc/{subject}.nc') # load preprocessed data
    sig = data.sel(chan = chan_sel).mean('chan').values # mean sig of chan sel
    hypno = pd.read_excel(f'../hypnos/hypno_{subject}.xlsx', index_col = 0) # load hypno of the run key
    hypno_upsampled = yasa.hypno_upsample_to_data(hypno = hypno['yasa hypnogram'].values, sf_hypno=1/30, data=sig, sf_data=srate) # upsample hypno

    fig = yasa.plot_spectrogram(sig, sf=srate, hypno=hypno_upsampled, cmap='Spectral_r')
    plt.savefig('../sandbox/spectrogram')
    plt.close()




        
    

