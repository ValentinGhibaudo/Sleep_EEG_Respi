import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from params import patients, respi_chan, srate
import ghibtools as gh

save = True

for patient in patients:
    print(patient)
    da = xr.open_dataarray(f'../preproc/{patient}.nc').sel(chan = respi_chan) # select respi chan = DEBIT
    rsp = da.values
    rsp_features = gh.get_resp_features(rsp, srate, manual_baseline_correction = -5)
    
    if save:
        rsp_features.to_excel(f'../resp_features/{patient}_rsp_features.xlsx')


    # cycles = rsp_cycle_detection(rsp, srate)
    # inspis = cycles[:,0]
    # expis = cycles[:,1]
    
    # duration_min=1.5
    # duration_max = 20
    # inspi_min=0.5
    # inspi_max=10
    
    # resp_features = get_all_respiration_features(resp_sig=rsp, sampling_rate=srate, cycles=cycles, t_start = 0.)
    # resp_features.insert(0, 'participant',patient)
    # initial_n_cycles = resp_features.shape[0]
    # clean_cycles = []
    # for i , cycle in resp_features.iterrows():
    #     if cycle['cycle_duration'] > duration_min and cycle['cycle_duration'] < duration_max and cycle['insp_duration'] > inspi_min and cycle['insp_duration'] < inspi_max and cycle['exp_duration'] > inspi_min and cycle['exp_duration'] < inspi_max:
    #         clean_cycles.append(cycle)

    # df_return = pd.concat(clean_cycles, axis = 1).T

    # print(f'{initial_n_cycles - df_return.shape[0]} cycles removed')
    # print(f'{df_return.shape[0]} cycles kept')

    # df_return.to_excel(f'../df_analyse/resp_features_{patient}.xlsx')
