import numpy as np
import pandas as pd
from scipy import signal

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

def extract_features_from_cmw_family(sig, time_sig, cmw_family_params):
    
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
        
    return da_features, da_cmw_family