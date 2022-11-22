import numpy as np
import ghibtools as gh
import xarray as xr
from params import patients, eeg_chans, srate, f_start, f_stop, n_step, cycle_start, cycle_stop, sigma_power_chans

save = True

for patient in patients:
    print(patient)
    data = xr.load_dataarray(f'../preproc/{patient}.nc') # load preprocessed data
    data_eeg = data.sel(chan = eeg_chans) # keep only eeg data
    del data # delete original xarray from memory
    
    sigma_power = None 
    for computed_chan in sigma_power_chans: # loop on parametred chans
        print(computed_chan)
        sig = data_eeg.sel(chan = computed_chan).values # select sig from chan
        whole_tf = gh.tf(sig, srate, f_start, f_stop, n_step, cycle_start, cycle_stop) # compute tf with set params
        if sigma_power is None:
            sigma_power = gh.init_da({'chan':sigma_power_chans, 'freq':whole_tf.coords['freqs'].values, 'time':whole_tf.coords['time'].values}) # init dataarray with desired shapes
        sigma_power.loc[computed_chan,:,:] = whole_tf.values # fill xarray at the chan with the whole night tf map of the chan
    
    if save:
        sigma_power.to_netcdf(f'../sigma/{patient}_morlet_sigma.nc') # save
        
    

