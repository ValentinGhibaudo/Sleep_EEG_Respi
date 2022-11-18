import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import xarray as xr
from cycle_detection import detect_respiration_cycles
from respiration_features import get_all_respiration_features
from params import patients, respi_chan, srate , rsp_detect_sign
import ghibtools as gh

def rsp_cycle_detection(resp_sig, srate):

    cycles = detect_respiration_cycles(resp_sig, sampling_rate=srate, t_start = 0., output = 'index',

                                    # preprocessing
                                    inspiration_sign = '+',
                                    high_pass_filter = None,
                                    constrain_frequency = None,
                                    median_windows_filter = None,

                                    # baseline
                                    baseline_with_average = True,
                                    manual_baseline = 0.,

                                    # clean
                                    eliminate_time_shortest_ratio = 2,
                                    eliminate_amplitude_shortest_ratio = 10,
                                    eliminate_mode = 'OR', # 'AND'

                                    )

    return cycles

for patient in patients:
    print(patient)
    da = xr.load_dataarray(f'../dataarray/da_N2N3_{patient}.nc').sel(chan = respi_chan)
    rsp = da.values
    t = da.coords['time'].values
    cycles = rsp_cycle_detection(rsp, srate)
    inspis = cycles[:,0]
    expis = cycles[:,1]
    
    duration_min=1.5
    duration_max = 20
    inspi_min=0.5
    inspi_max=10
    
    resp_features = get_all_respiration_features(resp_sig=rsp, sampling_rate=srate, cycles=cycles, t_start = 0.)
    resp_features.insert(0, 'participant',patient)
    initial_n_cycles = resp_features.shape[0]
    clean_cycles = []
    for i , cycle in resp_features.iterrows():
        if cycle['cycle_duration'] > duration_min and cycle['cycle_duration'] < duration_max and cycle['insp_duration'] > inspi_min and cycle['insp_duration'] < inspi_max and cycle['exp_duration'] > inspi_min and cycle['exp_duration'] < inspi_max:
            clean_cycles.append(cycle)

    df_return = pd.concat(clean_cycles, axis = 1).T

    print(f'{initial_n_cycles - df_return.shape[0]} cycles removed')
    print(f'{df_return.shape[0]} cycles kept')

    df_return.to_excel(f'../df_analyse/resp_features_{patient}.xlsx')
