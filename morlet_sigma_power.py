import numpy as np
import ghibtools as gh
import xarray as xr
from scipy import signal
from params import subjects, eeg_mono_chans, srate, f_start, f_stop, n_step, n_cycles, sigma_power_chans, decimate_factor, srate_down, amplitude_exponent

# FUNCTIONS 
def complex_mw(time, n_cycles , freq, a= 1, m = 0): 
    """
    Create a complex morlet wavelet by multiplying a gaussian window to a complex sinewave of a given frequency
    
    ------------------------------
    a = amplitude of the wavelet
    time = time vector of the wavelet
    n_cycles = number of cycles in the wavelet
    freq = frequency of the wavelet
    m = 
    """
    s = n_cycles / (2 * np.pi * freq)
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2)) # real gaussian window
    complex_sinewave = np.exp(1j * 2 *np.pi * freq * time) # complex sinusoidal signal
    cmw = GaussWin * complex_sinewave
    return cmw

def morlet_family(srate, f_start, f_stop, n_step, n_cycles):
    """
    Create a family of morlet wavelets
    
    ------------------------------
    srate : sampling rate
    f_start : lowest frequency of the wavelet family
    f_stop : highest frequency of the wavelet family
    n_step : number of frequencies from f_start to f_stop
    n_cycles : number of waves in the wavelet
    """
    tmw = np.arange(-5,5,1/srate)
    freqs = np.linspace(f_start,f_stop,n_step) 
    mw_family = np.zeros((freqs.size, tmw.size), dtype = 'complex')
    for i, fi in enumerate(freqs):
        mw_family[i,:] = complex_mw(tmw, n_cycles = n_cycles, freq = fi)
    return freqs, mw_family

def compute_tf(sig, srate, f_start, f_stop, n_step, n_cycles, amplitude_exponent=1):
    """
    Compute time-frequency matrix by convoluting wavelets on a signal
    
    ------------------------------
    sig : the signal 
    srate : sampling rate
    f_start : lowest frequency of the wavelet family
    f_stop : highest frequency of the wavelet family
    n_step : number of frequencies from f_start to f_stop
    n_cycles : number of waves in the wavelet
    amplitude_exponent : amplitude values extracted from the length of the complex vector will be raised to this exponent factor
    """
    freqs, family = morlet_family(srate, f_start = f_start, f_stop = f_stop, n_step = n_step, n_cycles = n_cycles)
    sigs = np.tile(sig, (n_step,1))
    tf = signal.fftconvolve(sigs, family, mode = 'same', axes = 1)
    power = np.abs(tf) ** amplitude_exponent
    return freqs , power


# RUN 
for subject in subjects:
    print(subject)
    data = xr.open_dataarray(f'../preproc/{subject}_reref.nc') # load preprocessed data (lazy load)
    data_eeg = data.sel(chan = eeg_mono_chans) # keep only eeg data
    
    sigma_power = None 
    for computed_chan in sigma_power_chans: # loop on set chans
        print(computed_chan)
        sig = data_eeg.sel(chan = computed_chan).values # select sig from chan
        sig_down = signal.decimate(sig, q=decimate_factor) # down sample sig before tf computation for faster computing and less memory used
        freqs , whole_tf = compute_tf(sig_down, srate_down, f_start, f_stop, n_step, n_cycles, amplitude_exponent) # compute tf with set params
        if sigma_power is None:
            sigma_power = gh.init_da({'chan':sigma_power_chans, 'freq':freqs, 'time':np.arange(0, sig_down.size/srate_down, 1/srate_down)}) # init dataarray with desired shapes
        sigma_power.loc[computed_chan,:,:] = whole_tf # fill xarray at the chan with the whole night tf map of the chan
    
    sigma_power.to_netcdf(f'../sigma/{subject}_morlet_sigma_down.nc') # save
        
    

