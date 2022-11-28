import numpy as np
import xarray as xr
import pandas as pd
import ghibtools as gh
from deform_tools import deform_to_cycle_template
from params import patients, nb_point_by_cycle, transition_ratio, stages_sigma_select, channels_sigma_select

save = True

for patient in patients:
    print(patient)
    morlet_sigma = xr.open_dataarray(f'../sigma/{patient}_morlet_sigma.nc') # lazy load time_frequency sigma map
    rsp_features = pd.read_excel(f'../resp_features/{patient}_rsp_features_staged.xlsx', index_col = 0) # load respi times
    mask_stages = rsp_features['sleep_stage'].isin(stages_sigma_select) # select only sigma extracted during the stages from list "stages_sigma_select"
    rsp_features = rsp_features[mask_stages]

    phase_freq_sigma = None

    for channel in channels_sigma_select: # select only sigma extracted in the channel list "channels_sigma_select"
        print(channel)
        morlet_sigma_chan = morlet_sigma.sel(chan = channel) # sel time freq map of the channel
        # stretch and slice phase-freq maps
        clipped_times, times_to_cycles, cycles, cycle_points, deformed_data = deform_to_cycle_template(data = morlet_sigma_chan.values.T, # 0 axis = time, 1 axis = freq
                                                                                                    times = morlet_sigma.coords['time'].values , 
                                                                                                    cycle_times=rsp_features[['start_time','transition_time']].values, 
                                                                                                    nb_point_by_cycle=nb_point_by_cycle,
                                                                                                    inspi_ratio = transition_ratio)

        deformed = deformed_data.T # transpose to have axis 0 = freq and axis 1 = phase points

        for cycle in cycles: # loop on cycles
            data_of_the_cycle = deformed[:,cycle*nb_point_by_cycle:(cycle+1)*nb_point_by_cycle] # slice phase-frequency map of the cycle
            if phase_freq_sigma is None:
                phase_freq_sigma = gh.init_da({'chan':channels_sigma_select,'cycle' : cycles, 'freq': morlet_sigma.coords['freq'].values , 'point':np.arange(0,nb_point_by_cycle,1)}) # init xarray 
            phase_freq_sigma.loc[channel,cycle, : , :] = data_of_the_cycle # phase-freq map is stored in xarray
        
        phase_freq_sigma = phase_freq_sigma.assign_attrs({'clipped_times':clipped_times,'times_to_cycles':times_to_cycles, 'cycle_points':cycle_points})

    if save:
        phase_freq_sigma.to_netcdf(f'../sigma_coupling/{patient}_phase_freq_sigma.nc') # save this 4 D xarray : chan * cycle * freq * time



