import numpy as np
import xarray as xr
import pandas as pd
import ghibtools as gh
from deform_tools import deform_to_cycle_template
from params import patients, nb_point_by_cycle, transition_ratio, stages_sigma_select, channels_sigma_select

save = True

for patient in patients:
    print(patient)
    morlet_sigma = xr.open_dataarray(f'../sigma/{patient}_morlet_sigma.nc') # lazy load time_frequency sigma map (chan * freq * time)
    rsp_features = pd.read_excel(f'../resp_features/{patient}_resp_features_tagged.xlsx', index_col = 0) # load respi times
    mask_stages = rsp_features['sleep_stage'].isin(stages_sigma_select) # select only sigma extracted during the stages from list "stages_sigma_select"
    rsp_features = rsp_features[mask_stages].reset_index(drop=True)

    phase_freq_sigma_concat = [] # process is divided in two steps that will be concatenated in this list

    for chan_list in [channels_sigma_select[:4],channels_sigma_select[4:]]: # cut chan list in two part to not overload memory
        print(chan_list)
        morlet_sigma_half = morlet_sigma.sel(chan = chan_list)
        data = morlet_sigma_half.data.T # from chan * freq * time to time * freq * chan
        # stretch and slice phase-freq maps
        clipped_times, times_to_cycles, cycles, cycle_points, deformed_data = deform_to_cycle_template(data = data, # 0 axis = time, 1 axis = freq, 2 axis = chan
                                                                                                    times = morlet_sigma.coords['time'].values , 
                                                                                                    cycle_times=rsp_features[['start_time','transition_time']].values, 
                                                                                                    nb_point_by_cycle=nb_point_by_cycle,
                                                                                                    inspi_ratio = transition_ratio)

        deformed = deformed_data.T # transpose to have axis 0 = chan, 1 = freq, 2 = phase points

        phase_freq_sigma = None
        for cycle in cycles: # loop on cycles
            data_of_the_cycle = deformed[:,:,cycle*nb_point_by_cycle:(cycle+1)*nb_point_by_cycle] # slice phase-frequency map of the cycle
            if phase_freq_sigma is None:
                phase_freq_sigma = gh.init_da({'chan':chan_list,'cycle' : cycles, 'freq': morlet_sigma.coords['freq'].values , 'point':np.arange(0,nb_point_by_cycle,1)}) # init xarray 
            phase_freq_sigma.loc[:, cycle, : , :] = data_of_the_cycle # phase-freq map is stored in xarray
        
        new_rsp_features = rsp_features[rsp_features.index.isin(list(cycles))] # one or two last cycles could be removed by deform tool
        new_event_tagging = list(new_rsp_features.loc[:,'Spindle_Tag'].values) # keep tagging
        
        phase_freq_sigma_concat.append(phase_freq_sigma)

    phase_freq_sigma_concat = xr.concat(phase_freq_sigma_concat, dim = 'chan') # concat the two products
    phase_freq_sigma_concat = phase_freq_sigma_concat.assign_attrs({'cycle_tagging':new_event_tagging}) # store cycle tagging by events (1 if resp cycle have spindle(s) inside or 0 if not)

    if save:
        phase_freq_sigma_concat.to_netcdf(f'../sigma_coupling/{patient}_phase_freq_sigma.nc') # save this 4 D xarray : chan * cycle * freq * time
        new_rsp_features.to_excel(f'../resp_features/{patient}_resp_features_tagged_stretch.xlsx') # save resp features of the keep cycles of these phase frequency maps
