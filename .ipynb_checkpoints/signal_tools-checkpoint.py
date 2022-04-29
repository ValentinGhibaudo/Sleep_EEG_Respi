import numpy as np
import pandas as pd
from scipy import signal
import scipy.interpolate

def discrete_FT_homemade(sig, srate):
    t = np.arange(0, sig.size/srate, 1/srate)
    f = np.linspace(0, int(srate/2), int(t.size / 2)) 
    # "number of unique freqs that can be extracted from a time series is 1/2 nb of points of the time series plus the zero freq, because of nyquist Th"
    # nyquist th = you need at least two points per cycle to measure a sine wave, and thus half the number of points in the data to the fastest frequency that can be extracted from t"
    fourier = np.zeros(t.size, dtype = 'complex_') # initialize fourier coeff with sig.size == time.size, dtype = complexe to be able to receive complex values
    for i in range(t.size): # loop on bins of time
        freq = i # freq of the kernel = the sine wave = 0, 1, 2 ... N
        kernel = np.exp(-1j * 2 * np.pi * freq * t ) # kernel = sine wave of freq i , and size = time = sig.size, but imaginary defined sine wave !
        dot_product = np.sum(sig * kernel) # dot product = sum of sig * kernel simply, because sig.size == kernel.size, so sliding/zero-padding is not needed
        fourier[i] = dot_product # fourier coefficient in bin i == dot_product of bin i
    fourier_full = fourier 
    fourier = fourier[0:int(t.size/2)] # see below (keep only positive freqs by selecting first half of fourier) 
    f = f * duration
    power = np.abs(fourier) ** 2
    phase = np.angle(fourier, deg = False) # Return the angle of the complex argument. In radians or degrees according to deg param
    return f, fourier, fourier_full, power, phase # fourier coef = concatenation of dot products of sig vs kernel with kernel = imaginary sine waves of freq = idx of iteration across time bins

def discrete_FT_homemade_short(sig, srate):
    t = np.arange(0, sig.size/srate, 1/srate)
    f = np.linspace(0, int(srate/2), int(t.size / 2)) 
    # "number of unique freqs that can be extracted from a time series is 1/2 nb of points of the time series plus the zero freq, because of nyquist Th"
    # nyquist th = you need at least two points per cycle to measure a sine wave, and thus half the number of points in the data to the fastest frequency that can be extracted from t"
    fourier = np.zeros(t.size, dtype = 'complex_') # initialize fourier coeff with sig.size == time.size, dtype = complexe to be able to receive complex values
    for i in range(t.size): # loop on bins of time
        freq = i # freq of the kernel = the sine wave = 0, 1, 2 ... N
        kernel = np.exp(-1j * 2 * np.pi * freq * t ) # kernel = sine wave of freq i , and size = time = sig.size, but imaginary defined sine wave !
        dot_product = np.sum(sig * kernel) # dot product = sum of sig * kernel simply, because sig.size == kernel.size, so sliding/zero-padding is not needed
        fourier[i] = dot_product # fourier coefficient in bin i == dot_product of bin i
    fourier_full = fourier 
    fourier = fourier[0:int(t.size/2)] # see below (keep only positive freqs by selecting first half of fourier) 
    f = f * duration
    power = np.abs(fourier) ** 2
    phase = np.angle(fourier, deg = False) # Return the angle of the complex argument. In radians or degrees according to deg param
    return f, power, fourier_full # fourier coef = concatenation of dot products of sig vs kernel with kernel = imaginary sine waves of freq = idx of iteration across time bins

def inverse_FT_homemade(N, fourier_full):
    sine_waves = np.zeros((N,N))
    for fi in range(N):
        sine_wave = fourier_full[fi] * np.exp(-1j * 2 * np.pi * fi * time)
        sine_wave = np.real(sine_wave)
        sine_waves[fi,:] = sine_wave
    sig_reconstructed = - np.sum(sine_waves, axis = 0) / (N * duration)
    return sig_reconstructed

def inverse_FT_homemade_modif(N, fourier_full):
    sine_waves = np.zeros((N,N))
    for fi in range(N):
        sine_wave = fourier_full[fi] * np.exp(-1j * 2 * np.pi * fi * time)
        sine_wave = np.real(sine_wave)
        sine_waves[fi,:] = sine_wave
    sig_reconstructed = - np.sum(sine_waves, axis = 0) / (N * duration)
    return sig_reconstructed


def convolution_time_domain(sig, kernel):
    kernel_flip = kernel[::-1] # flip the kernel
    zero_padding = np.zeros(kernel.size - 1) # prepare the amount of zeros to add to the beginning and end of the sig to begin and end the convolution and extremes of sig
    sig_zero_padded = np.concatenate((zero_padding, sig, zero_padding)) # add zero_padding to beginning and end of sig
    conv = np.zeros(sig_zero_padded.size - kernel.size + 1) # prepare shape of convolution to be filled with time bin results

    for i in range(sig_zero_padded.size - kernel.size + 1): # loop on each bin of sig (nb iterations = conv.size)
        bin_product = []
        for j in range(kernel.size): # loop on kernel bins
            bin_product.append(sig_zero_padded[i+j]*kernel_flip[j]) # append values (= bin_sig * bin_kernel) to be summed to get dot product
        dot_product = np.sum(bin_product) # sum values to get the dot product
        conv[i] = dot_product # add dot_product in a time series
        
    convolution = conv[int(kernel.size/2):conv.size - int(kernel.size/2) + 1 ] # cut result with one-half size of kernel at the beginning and one-half + 1 size of the kernel at the end to get same size of sig
    
    return convolution

def convolution_freq_domain(sig, kernel):
    f_sig, power_sig, fourier_full_sig  = discrete_FT_homemade_short(sig, srate)
    f_kernel, power_kernel, fourier_full_kernel = discrete_FT_homemade_short(kernel, srate)
    
    multiple_fourier_full = fourier_full_sig * fourier_full_kernel
    
    convo_by_freq = inverse_FT_homemade(N = sig.size, fourier_full = multiple_fourier_full)
    
    return convo_by_freq, power_sig, power_kernel, f_sig, multiple_fourier_full

def gaussian_win(a , time, m , n , freq):
    
    # a = 2 # amplitude
    # time = time # time
    # m  = time[-1] / 2 # offset (not relevant for eeg analyses and can be always set to zero)
    # n = 10 # refers to the number of wavelet cycles , defines the trade-off between temporal precision and frequency precision
    # freq = 10 # change the width of the gaussian and therefore of the wavelet
    # s = n / (2 * np.pi * freq) # std of the width of the gaussian, freq = in Hz
    
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2))
    
    return GaussWin

def morlet_wavelet(a , time, m , n , freq):
    s = n / (2 * np.pi * freq)
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2))
    SinWin = np.sin ( 2 * np.pi * freq * time )
    MorletWavelet = GaussWin * SinWin
    return MorletWavelet

def complex_mw(a , time, n , freq, m = 0):
    s = n / (2 * np.pi * freq)
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2))
    complex_sinewave = np.exp(1j * 2 *np.pi * freq * time)
    cmw = GaussWin * complex_sinewave
    return cmw

def extract_features_from_cmw_family(sig, time_sig, cmw_family_params, return_cmw_family=False):
    
    shape = (2, cmw_family_params['range'].size, cmw_family_params['time'].size)
    cmw_family = np.zeros(shape)

    dims = ['axis','freq','time']
    coords = {'axis':['real','imag'],'freq':cmw_family_params['range'], 'time':cmw_family_params['time']}
    da_cmw_family = xr.DataArray(data = cmw_family, dims = dims, coords = coords, name = 'cmw_family')

    shape = (cmw_family_params['range'].size , time_sig.size)
    
    reals = np.zeros(shape)
    imags = np.zeros(shape)
    modules = np.zeros(shape)
    angles = np.zeros(shape)

    features = ['filtered','i_filtered','phase','power']
    data = np.zeros(shape = (len(features),reals.shape[0] , reals.shape[1]))
    dims = ['feature','freqs','time']
    coords = {'feature':features, 'freqs':cmw_family_params['range'], 'time':time_sig}
    da_features = xr.DataArray(data = data, dims = dims, coords = coords, name = 'features')

    cmw_family_freq_range = cmw_family_params['range']
    cmw_family_n_range = cmw_family_params['n_cycles']
    idx = np.arange(0,cmw_family_freq_range.size,1)
    
    for i, fi, ni in zip(idx, cmw_family_freq_range, cmw_family_n_range):
     
        a = cmw_family_params['amp']
        time = cmw_family_params['time']
        m = cmw_family_params['m']
        
        cmw_f = complex_mw(a=a, time=time,n=ni, freq=fi, m = m)

        da_cmw_family.loc['real',fi,:] = np.real(cmw_f)
        da_cmw_family.loc['imag',fi,:] = np.imag(cmw_f)

        complex_conv = signal.convolve(sig, cmw_f, mode = 'same')

        real = np.real(complex_conv)
        imag = np.imag(complex_conv)
        angle = np.angle(complex_conv)
        module = np.zeros((time_sig.size))
        for i_bin in range(time_sig.size):
            module_i_bin = np.sqrt((real[i_bin])**2 + (imag[i_bin])**2)
            module[i_bin] = module_i_bin

        reals[i,:] = real
        imags[i,:] = imag
        modules[i,:] = module
        angles[i,:] = angle

        da_features.loc['filtered',:,:] = reals
        da_features.loc['i_filtered',:,:] = imags
        da_features.loc['phase',:,:] = angle
        da_features.loc['power',:,:] = modules
    if return_cmw_family:
        return da_features, da_cmw_family
    else:
        return da_features.loc['power',:,:]
    
    
    
def stretch_data(resp_features, nb_point_by_cycle, data, srate):

    # params
    cycle_times = resp_features[['inspi_time', 'expi_time']].values
    mean_cycle_duration = np.mean(resp_features[['insp_duration', 'exp_duration']].values, axis=0)
    mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
    times = np.arange(0,np.size(data))/srate

    clipped_times, times_to_cycles, cycles, cycle_points, data_stretch_linear = deform_to_cycle_template(
            data, times, cycle_times, nb_point_by_cycle=nb_point_by_cycle, inspi_ratio=0.4)

    nb_cycle = data_stretch_linear.shape[0]//nb_point_by_cycle
    phase = np.arange(nb_point_by_cycle)/nb_point_by_cycle
    data_stretch = data_stretch_linear.reshape(int(nb_cycle), int(nb_point_by_cycle))

    return data_stretch


def deform_to_cycle_template(data, times, cycle_times, nb_point_by_cycle=40, inspi_ratio = 0.4):
    """
    Input:
    data: ND array time axis must always be 0
    times: real timestamps associated to data
    cycle_times: N*2 array columns are inspi and expi times. If expi is "nan", corresponding cycle is skipped
    nb_point_by_cycle: number of respi phase per cycle
    inspi_ratio: relative length of the inspi in a full cycle (between 0 and 1)
    
    Output:
    clipped_times: real times used (when both respi and signal exist)
    times_to_cycles: conversion of clipped_times in respi cycle phase
    cycles: array of cycle indices (rows of cycle_times) used
    cycle_points: respi cycle phases where deformed_data is computed
    deformed_data: data rescaled to have cycle_points as "time" reference
    """
    
    #~ nb_point_inspi = int(nb_point_by_cycle*inspi_ratio)
    #~ nb_point_expi = nb_point_by_cycle - nb_point_inspi
    #~ one_cycle = np.linspace(0,1,nb_point_by_cycle)
    #~ two_cycle = np.linspace(0,2,nb_point_by_cycle*2)
    
    #~ print('cycle_times.shape', cycle_times.shape)
    
    #clip cycles if data/times smaller than cycles
    keep_cycle = (cycle_times[:, 0]>=times[0]) & (cycle_times[:, 1]<times[-1])
    first_cycle = np.where(keep_cycle)[0].min()
    last_cycle = np.where(keep_cycle)[0].max()+1
    if last_cycle==cycle_times.shape[0]:
        #~ print('yep')
        last_cycle -= 1
    #~ print('first_cycle', first_cycle, 'last_cycle', last_cycle)

    #clip times/data if cycle_times smaller than times
    keep = (times>=cycle_times[first_cycle,0]) & (times<cycle_times[last_cycle,0])
    #~ print(keep)
    clipped_times = times[keep]
    clipped_data = data[keep]
    #~ print('clipped_times', clipped_times.shape, clipped_times[0], clipped_times[-1])
    
    # construct cycle_step
    times_to_cycles = np.zeros(clipped_times.shape)*np.nan
    cycles = np.arange(first_cycle, last_cycle)
    t_start = clipped_times[0]
    sr = np.median(np.diff(clipped_times))
    #~ print('t_start', t_start, 'sr', sr)
    for c in cycles:
        #2 segments : inspi + expi
        
        if not np.isnan(cycle_times[c, 1]):
            #no missing cycles
            mask_inspi_times=(clipped_times>=cycle_times[c, 0])&(clipped_times<cycle_times[c, 1])
            mask_expi_times=(clipped_times>=cycle_times[c, 1])&(clipped_times<cycle_times[c+1, 0])
            times_to_cycles[mask_inspi_times]=(clipped_times[mask_inspi_times]-cycle_times[c, 0])/(cycle_times[c, 1]-cycle_times[c, 0])*inspi_ratio+c
            times_to_cycles[mask_expi_times]=(clipped_times[mask_expi_times]-cycle_times[c, 1])/(cycle_times[c+1, 0]-cycle_times[c, 1])*(1-inspi_ratio)+c+inspi_ratio
                    
        else:
            #there is a missing cycle
            mask_cycle_times=(clipped_times>=cycle_times[c, 0])&(clipped_times<cycle_times[c+1, 0])
            times_to_cycles[mask_cycle_times]=(clipped_times[mask_cycle_times]-cycle_times[c, 0])/(cycle_times[c+1, 0]-cycle_times[c, 0])+c
    
    # new clip with cycle
    keep = ~np.isnan(times_to_cycles)
    times_to_cycles = times_to_cycles[keep]
    clipped_times = clipped_times[keep]
    clipped_data = clipped_data[keep]
    
    
    interp = scipy.interpolate.interp1d(times_to_cycles, clipped_data, kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')
    cycle_points = np.arange(first_cycle, last_cycle, 1./nb_point_by_cycle)

    if cycle_points[-1]>times_to_cycles[-1]:
        # it could append that the last bins of the last cycle is out last cycle
        # due to rounding effect so:
        last_cycle = last_cycle-1
        cycles = np.arange(first_cycle, last_cycle)
        cycle_points = np.arange(first_cycle, last_cycle, 1./nb_point_by_cycle)
    
    deformed_data = interp(cycle_points)
    
    #put NaN for missing cycles
    missing_ind,  = np.nonzero(np.isnan(cycle_times[:, 1]))
    #~ print('missing_ind', missing_ind)
    for c in missing_ind:
        #mask = (cycle_points>=c) & (cycle_points<(c+1))
        #due to rounding problem add esp
        esp = 1./nb_point_by_cycle/10.
        mask = (cycle_points>=(c-esp)) & (cycle_points<(c+1-esp))
        deformed_data[mask] = np.nan
    
    
    return clipped_times, times_to_cycles, cycles, cycle_points, deformed_data
    
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

def ecg_to_hrv(ecg, srate=srate, show = False):
    ecg = -ecg
    if show:
        plt.figure(figsize=(15,10))
        ecg_signals, info_ecg = nk.ecg_process(ecg, sampling_rate=srate, method='neurokit')
        nk.ecg_plot(ecg_signals, rpeaks=info_ecg, sampling_rate=srate, show_type='default')
    
    clean = nk.ecg_clean(ecg, sampling_rate=srate, method='neurokit')
    peaks, info_ecg = nk.ecg_peaks(clean, sampling_rate=srate,method='neurokit', correct_artifacts=True)
    
    R_peaks = info_ecg['ECG_R_Peaks'] # get R time points
    diff_R_peaks = np.diff(R_peaks) 
    x = vector_time
    xp = R_peaks[1::]/srate
    fp = diff_R_peaks
    interpolated_hrv = np.interp(x, xp, fp, left=None, right=None, period=None) / srate
    fci = 60 / interpolated_hrv
    return clean, fci