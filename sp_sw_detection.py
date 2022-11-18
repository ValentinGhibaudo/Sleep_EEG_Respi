import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import yasa
from params import eeg_chans, patients, srate

save = True

def detect_spindles_and_slow_waves(patient, srate=srate, chan = eeg_chans, save=False):
    input_file = f'../dataarray/da_N2N3_{patient}.nc'
    da = xr.load_dataarray(input_file)
    data = da.sel(chan = chan).dropna(dim='time').values

    sp = yasa.spindles_detect(data=data, sf=srate, ch_names=chan, multi_only=False, remove_outliers=True)
    spindles = sp.summary()  
    spindles.insert(0 , 'patient', patient)

    sw = yasa.sw_detect(data=data, sf=srate, ch_names=chan, freq_sw=[0.2,1.5], remove_outliers=True, coupling = True) # O.2 lowcut to include resp ?
    slow_waves = sw.summary()
    slow_waves.insert(0 , 'patient', patient)
        
    if save: 
        spindles.to_excel(f'../df_analyse/spindles_{patient}.xlsx')
        slow_waves.to_excel(f'../df_analyse/sw_{patient}.xlsx')
    return spindles, slow_waves

# sp_concat = []
# sw_concat = []
for patient in patients:
    print(patient)
    sp, sw = detect_spindles_and_slow_waves(patient, save)
    print(f'n spindles detected = {sp.shape[0]}')
    print(f'n slow-waves detected = {sw.shape[0]}')
    
#     sp_concat.append(sp)
#     sw_concat.append(sw)
    
# sp_all = pd.concat(sp_concat)
# sw_all = pd.concat(sw_concat)
# if save:
#     sp_all.to_excel('../df_analyse/spindles_all_patients.xlsx')
#     sw_all.to_excel('../df_analyse/slow_waves_all_patients.xlsx')

