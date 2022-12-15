import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import ghibtools as gh
import pandas as pd
from params import *



for subject in subjects: # loop on run keys
    print(subject)
    
    da = xr.open_dataarray(f'../preproc/{subject}_reref.nc') # load lazy the whole night recordings preprocessed
    da_eeg = da.sel(chan = eeg_mono_chans) # select eeg derivations
    
    da_sigma_hilbert = None 
    for channel in eeg_mono_chans: # loop on eeg chans
        sig = da_eeg.sel(chan = channel).values # select sig of the deriation
        sig_sigma_filtered = gh.filter_sig(sig, da.attrs['srate'], f_start , f_stop) # filter sig between f_start and f_stop (see params) = sigma filtered
        sigma_env = gh.get_amp(sig_sigma_filtered) # extract envelope of the sigma filtered sig
        if da_sigma_hilbert is None:
            da_sigma_hilbert = gh.init_da({'type':['filtered','envelope'], 'chan':eeg_mono_chans, 'time':da.coords['time'].values}) # init xarray with right shapes
        da_sigma_hilbert.loc['filtered',channel,:] = sig_sigma_filtered # store filtered sigma sig
        da_sigma_hilbert.loc['envelope',channel,:] = sigma_env # store envelope
        

    da_sigma_hilbert.to_netcdf(f'../sigma/{subject}_hilbert_sigma.nc') # save