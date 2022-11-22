import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import ghibtools as gh
import pandas as pd
from params import *

save = True

for patient in patients: # loop on run keys
    print(patient)
    
    da = xr.open_dataarray(f'../preproc/{patient}.nc') # load lazy the whole night recordings preprocessed
    da_eeg = da.sel(chan = eeg_chans) # select eeg derivations
    
    da_sigma_hilbert = None 
    for derivation in eeg_chans: # loop on eeg chans
        sig = da_eeg.sel(chan = derivation).values # select sig of the deriation
        sig_sigma_filtered = gh.filter_sig(sig, da.attrs['srate'], f_start , f_stop) # filter sig between f_start and f_stop (see params) = sigma filtered
        sigma_env = gh.get_amp(sig_sigma_filtered) # extract envelope of the sigma filtered sig
        if da_sigma_hilbert is None:
            da_sigma_hilbert = gh.init_da({'type':['filtered','envelope'], 'chan':eeg_chans, 'time':da.coords['time'].values}) # init xarray with right shapes
        da_sigma_hilbert.loc['filtered',derivation,:] = sig_sigma_filtered # store filtered sigma sig
        da_sigma_hilbert.loc['envelope',derivation,:] = sigma_env # store envelope
        
    if save:
        da_sigma_hilbert.to_netcdf(f'../sigma/{patient}_hilbert_sigma.nc') # save