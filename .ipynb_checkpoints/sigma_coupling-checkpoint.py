import numpy as np
import xarray as xr
import pandas as pd
import ghibtools as gh
from deform_tools import deform_to_cycle_template
from params import subjects, nb_point_by_cycle, transition_ratio, stages_sigma_select, channels_sigma_select

def add_nan_where_cleaned_cycle(rsp): # Function that add nan where cycles have been cleaned to not be computed by deform_tools
    rsp_nan = rsp.copy() 
    for c, row in rsp.iterrows(): # loop on cycles features
        if not c == rsp.index[-1]: # do not do thet next steps in last cycle
            if row['stop'] != rsp.loc[c+1,'start']: # if start index of the next cycle is not same as stop index of the present cycle, means that there are lacking cycles between..
                rsp_nan.loc[c+0.5,:] = np.nan # .. so between these two cycles is added Nan values that will not be computed by deform_to_cycle_template at a middle loc
    return rsp_nan.sort_index().reset_index(drop=True) # sort by index to reorder nan rows at the right loc


for subject in subjects:
    print(subject)
    morlet_sigma = xr.open_dataarray(f'../sigma/{subject}_morlet_sigma_down.nc') # lazy load time_frequency sigma map (chan * freq * time)
    rsp_features = pd.read_excel(f'../resp_features/{subject}_resp_features_tagged.xlsx', index_col = 0).reset_index(drop=True) # load respi times
    mask_stages = rsp_features['sleep_stage'].isin(stages_sigma_select) # select only sigma extracted during the stages from list "stages_sigma_select"
    rsp_features = rsp_features[mask_stages].reset_index(drop=True)
    rsp_features_nan = add_nan_where_cleaned_cycle(rsp_features) # add nan where cycles should not be deformed because too long because of lacking because of cleaning
    
    phase_freq_sigma = None

    for channel in morlet_sigma.coords['chan'].values: # cut chan list in two part to not overload memory
        print(channel)
        data = morlet_sigma.sel(chan = channel).data.T # from chan * freq * time to time * freq * chan
        # stretch and slice phase-freq maps
        clipped_times, times_to_cycles, cycles, cycle_points, deformed_data = deform_to_cycle_template(data = data, # 0 axis = time, 1 axis = freq
                                                                                                    times = morlet_sigma.coords['time'].values , 
                                                                                                    cycle_times=rsp_features_nan[['start_time','transition_time']].values, 
                                                                                                    nb_point_by_cycle=nb_point_by_cycle,
                                                                                                    inspi_ratio = transition_ratio)
        
        
        # deformed_data : axis 0 = point, 1 = freq
        deformed_data_sliced_by_cycle = deformed_data.reshape(cycles.size, nb_point_by_cycle, deformed_data.shape[1]) # reshape to have axis 0 = cycles, 1 = points, axis 2 = freq
        
        if phase_freq_sigma is None:
            phase_freq_sigma = gh.init_da({'chan':morlet_sigma.coords['chan'].values,'cycle' : cycles, 
                                           'point': np.linspace(0,1,nb_point_by_cycle) , 'freq':morlet_sigma.coords['freq'].values })# init xarray
            
        phase_freq_sigma.loc[channel, :, : , :] = deformed_data_sliced_by_cycle # phase-freq map is stored in xarray
        
    new_rsp_features = rsp_features_nan.iloc[cycles,:] # resp features are masked to the true computed cycles in deform_to_cycle_template

    phase_freq_sigma.to_netcdf(f'../sigma_coupling/{subject}_phase_freq_sigma.nc') # save this 4 D xarray : chan * cycle * phase * freq
    new_rsp_features.to_excel(f'../resp_features/{subject}_resp_features_tagged_stretch.xlsx') # save resp features of the kept cycles of these phase frequency maps
