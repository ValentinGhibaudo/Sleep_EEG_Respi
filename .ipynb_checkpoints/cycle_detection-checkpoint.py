# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

"""
Script for cycle detection on a respiratory signals.

This work on numpy+sampling_rate or neo.AnalogSignal


Vieux code historique qui méterait un jour d'être revu....

"""

import numpy as np
import scipy.signal


def fft_filter(sig, f_low =None, f_high=None, sampling_rate=1., axis=0):
    n = sig.shape[axis]
    n = n//2*2+1 #trick to avoid shape problem between mask and f_sig
    freqs = np.fft.fftfreq(n, d = 1./sampling_rate)
    freqs = freqs[freqs>=0]
    
    mask = np.zeros((freqs.size), dtype = bool)
    if f_low is not None:
        mask[np.abs(freqs) <= f_low] = True
    
    if f_high is not None:
        mask[np.abs(freqs) >= f_high] = True
    
    sl = [ slice(None) for i in range(sig.ndim) ]
    sl[axis] = mask
    f_sig = np.fft.rfft(sig,n=n, axis = axis)
    f_sig[sl] = 0.
    filtered_sig = np.fft.irfft(f_sig, axis = axis)
    return filtered_sig



def detect_respiration_cycles(resp_sig, sampling_rate, t_start = 0., output = 'index',

                                    # preprocessing
                                    inspiration_sign = '-',
                                    high_pass_filter = None,
                                    constrain_frequency = None,
                                    median_windows_filter = None,
                                    
                                    # baseline
                                    baseline_with_average = False,
                                    manual_baseline = 0.,
                                    
                                    # clean
                                    eliminate_time_shortest_ratio = 10,
                                    eliminate_amplitude_shortest_ratio = 10,
                                    eliminate_mode = 'OR', # 'AND'
                                    
                                    ):


    sig = resp_sig.copy()
    sr = sampling_rate

    # STEP 1 : preprocessing
    sig = sig  - manual_baseline

    if inspiration_sign =='-' :
        sig = -sig
    
    if median_windows_filter is not None:
        k = int(np.round(median_windows_filter*sr/2.)*2+1)
        sig = scipy.signal.medfilt(sig, kernel_size = k)
    
    original_sig = resp_sig.copy()
    
    #baseline center
    if baseline_with_average:
        centered_sig = sig - sig.mean()
    else:
        centered_sig = sig

    if high_pass_filter is not None:
        sig =  fft_filter(sig, f_low =high_pass_filter, f_high=None, sampling_rate=sr)
    
    # hard filter to constrain frequency
    if constrain_frequency is not None:
        filtered_sig =  fft_filter(centered_sig, f_low =None, f_high=constrain_frequency, sampling_rate=sr)
    else :
        filtered_sig = centered_sig
    
    # STEP 2 : crossing zeros on filtered_sig
    ind1, = np.where( (filtered_sig[:-1] <=0) & (filtered_sig[1:] >0))
    ind2, = np.where( (filtered_sig[:-1] >=0) & (filtered_sig[1:] <0))
    if ind1.size==0 or ind2.size==0:
        return np.zeros((0,2), dtype='int64')
    ind2 = ind2[ (ind2>ind1[0]) & (ind2<ind1[-1]) ]
    
    # STEP 3 : crossing zeros on centered_sig
    ind_inspi_possible, = np.where( (centered_sig[:-1]<=0 ) &  (centered_sig[1:]>0 ) )
    list_inspi = [ ]
    for i in range(len(ind1)) :
        ind = np.argmin( np.abs(ind1[i] - ind_inspi_possible) )
        list_inspi.append( ind_inspi_possible[ind] )
    list_inspi = np.unique(list_inspi)

    ind_expi_possible, = np.where( (centered_sig[:-1]>0 ) &  (centered_sig[1:]<=0 ) )
    list_expi = [ ]
    for i in range(len(list_inspi)-1) :
        ind_possible = ind_expi_possible[ (ind_expi_possible>list_inspi[i]) & (ind_expi_possible<list_inspi[i+1]) ]
        
        ind_possible2 = ind2[ (ind2>list_inspi[i]) & (ind2<list_inspi[i+1]) ]
        ind_possible2.sort()
        if ind_possible2.size ==1 :
            ind = np.argmin( abs(ind_possible2 - ind_possible ) )
            list_expi.append( ind_possible[ind] )
        elif ind_possible2.size >=1 :
            ind = np.argmin( np.abs(ind_possible2[-1] - ind_possible ) )
            list_expi.append( ind_possible[ind]  )
        else :
            list_expi.append( max(ind_possible)  )
    
    list_inspi,list_expi =  np.array(list_inspi,dtype = 'int64')+1, np.array(list_expi,dtype = 'int64')+1
    
    
    # STEP 4 :  cleaning for small amplitude and duration
    nb_clean_loop = 20
    if eliminate_mode == 'OR':
        # eliminate cycle with too small duration or too small amplitude
    
        if eliminate_amplitude_shortest_ratio is not None :
            for b in range(nb_clean_loop) :
                max_inspi = np.zeros((list_expi.size))
                for i in range(list_expi.size) :
                    max_inspi[i] = np.max( np.abs(centered_sig[list_inspi[i]:list_expi[i]]) )
                ind, = np.where( max_inspi < np.median(max_inspi)/eliminate_amplitude_shortest_ratio)
                list_inspi[ind] = -1
                list_expi[ind] = -1
                list_inspi = list_inspi[list_inspi != -1]
                list_expi = list_expi[list_expi != -1]
                
                max_expi = np.zeros((list_expi.size))
                for i in range(list_expi.size) :
                    max_expi[i] = np.max( abs(centered_sig[list_expi[i]:list_inspi[i+1]]) )
                ind, = np.where( max_expi < np.median(max_expi)/eliminate_amplitude_shortest_ratio)
                list_inspi[ind+1] = -1
                list_expi[ind] = -1
                list_inspi = list_inspi[list_inspi != -1]
                list_expi = list_expi[list_expi != -1]
            
        if eliminate_time_shortest_ratio is not None :
            for i in range(nb_clean_loop) :
                l = list_expi - list_inspi[:-1]
                ind, = np.where(l< np.median(l)/eliminate_time_shortest_ratio )
                list_inspi[ind] = -1
                list_expi[ind] = -1
                list_inspi = list_inspi[list_inspi != -1]
                list_expi = list_expi[list_expi != -1]
                
                l = list_inspi[1:] - list_expi
                ind, = np.where(l< np.median(l)/eliminate_time_shortest_ratio )
                list_inspi[ind+1] = -1
                list_expi[ind] = -1
                list_inspi = list_inspi[list_inspi != -1]
                list_expi = list_expi[list_expi != -1]
    
    
    elif eliminate_mode == 'AND':
        # eliminate cycle with both too small duration and too small amplitude
        max_inspi = np.zeros((list_expi.size))
        for b in range(nb_clean_loop) :
            
            max_inspi = np.zeros((list_expi.size))
            for i in range(list_expi.size) :
                max_inspi[i] = np.max( np.abs(centered_sig[list_inspi[i]:list_expi[i]]) )
            l = list_expi - list_inspi[:-1]
            cond = ( max_inspi < np.median(max_inspi)/eliminate_amplitude_shortest_ratio ) & (l< np.median(l)/eliminate_time_shortest_ratio)
            ind,  = np.where(cond)
            list_inspi[ind] = -1
            list_expi[ind] = -1
            list_inspi = list_inspi[list_inspi != -1]
            list_expi = list_expi[list_expi != -1]
            
            max_expi = np.zeros((list_expi.size))
            for i in range(list_expi.size) :
                max_expi[i] = np.max( abs(centered_sig[list_expi[i]:list_inspi[i+1]]) )
            l = list_inspi[1:] - list_expi
            cond = ( max_expi < np.median(max_expi)/eliminate_amplitude_shortest_ratio) & (l< np.median(l)/eliminate_time_shortest_ratio )
            ind,  = np.where(cond)
            list_inspi[ind+1] = -1
            list_expi[ind] = -1
            list_inspi = list_inspi[list_inspi != -1]
            list_expi = list_expi[list_expi != -1]
    
    
    # STEP 5 : take crossing zeros on original_sig, last one before min for inspiration
    ind_inspi_possible, = np.where( (original_sig[:-1]<=0 ) &  (original_sig[1:]>0 ) )
    for i in range(len(list_inspi)-1) :
        ind_max = np.argmax(centered_sig[list_inspi[i]:list_expi[i]])
        ind = ind_inspi_possible[ (ind_inspi_possible>=list_inspi[i]) & (ind_inspi_possible<=list_inspi[i]+ind_max) ]
        if ind.size!=0:
            list_inspi[i] = ind.max()
    
    if output == 'index':
        cycles = -np.ones( (list_inspi.size, 2),  dtype = 'int64')
        cycles[:,0] = list_inspi
        cycles[:-1,1] = list_expi
    elif output == 'times':
        cycles = zeros( (list_inspi.size, 2),  dtype = 'float64')*np.nan
        times = np.arange(sig.size, dtype = float)/sr + t_start
        cycles[:,0] = times[list_inspi]
        cycles[:-1,1] = times[list_expi]
    
    return cycles




def neo_detect_respiration_cycles(neo_resp_sig, **kargs):
    resp_sig = neo_resp_sig.magnitude
    sampling_rate = neo_resp_sig.sampling_rate.rescale('Hz').magnitude
    t_start = neo_resp_sig.t_start.rescale('s').magnitude
    return detect_respiration_cycles(resp_sig, sampling_rate, t_start = t_start, **kargs)
