# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

"""
Script to extract features from respiratory cycles:
  * duration
  * amplitudes
  * volume
  *...


"""

import numpy as np
import pandas as pd

from cycle_detection import detect_respiration_cycles


def get_all_respiration_features(resp_sig, sampling_rate, cycles, t_start = 0.):
    
    times = np.arange(resp_sig.size)/sampling_rate + t_start
    sr = sampling_rate
    
    assert cycles.dtype.kind == 'i'
    
    n = cycles.shape[0]-1
    
    index = np.arange(n, dtype = 'int64')
    df = pd.DataFrame(index = index)
    
    insp = times[cycles[:-1,0]]
    expi = times[cycles[:-1,1]]
    insp_next = times[cycles[1:,0]]

    ix1 = cycles[:-1,0]
    ix2 = cycles[:-1,1]
    ix3 = cycles[1:,0]
    
    df['cycle_num'] = pd.Series(range(n) , index = index, dtype = int)
    df['inspi_index'] = pd.Series(ix1 , index = index, dtype = 'int64')
    df['expi_index'] = pd.Series(ix2 , index = index, dtype = 'int64')
    df['inspi_time'] = pd.Series(insp , index = index)
    df['expi_time'] = pd.Series(expi , index = index)
    df['cycle_duration'] = pd.Series(insp_next - insp  , index = index)
    df['insp_duration'] = pd.Series(expi - insp  , index = index)
    df['exp_duration'] = pd.Series(insp_next - expi  , index = index)
    df['cycle_freq'] = 1./df['cycle_duration']
    for k in ('insp_volume', 'exp_volume', 'total_amplitude', 'insp_amplitude', 'exp_amplitude'):
        df[k] = pd.Series(index = index)
    
    #missing cycle
    mask = ix2==-1
    df.loc[mask, ['expi_time', 'cycle_duration', 'insp_duration', 'exp_duration', 'cycle_freq']] = np.nan
    
    for c in range(n):
        i1, i2, i3 = ix1[c], ix2[c], ix3[c]
        if i2 == -1:
            #this is a missing cycle in the middle
            continue
        #~ print(i1, i2, i3)
        df.loc[c, 'insp_volume'] = np.sum(resp_sig[i1:i2])/sr
        df.loc[c, 'exp_volume'] = np.sum(resp_sig[i2:i3])/sr
        df.loc[c, 'insp_amplitude'] = np.max(np.abs(resp_sig[i1:i2]))
        df.loc[c, 'exp_amplitude'] = np.max(np.abs(resp_sig[i2:i3]))
    
    df['total_amplitude'] = df['insp_amplitude']+df['exp_amplitude']
        
    return df


"""
def get_respiration_features_by_trial(resp_features, trial_times, cycle_range = range(-2,2)):
    n = trial_times.size
    index = np.arange(n, dtyep = 'int64')
    columns = pd.MultiIndex.from_product([resp_features.columns, cycle_range], names = ['feature', 'trial_cycle'])
    df = DataFrame(index = index, columns = columns)
    
    
    for feature in resp_features.features:
        for trial_time in 
    
    return df

"""
