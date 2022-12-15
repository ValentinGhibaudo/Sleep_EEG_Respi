import numpy as np
import ghibtools as gh
import xarray as xr
from scipy import signal
from params import subjects, eeg_mono_chans, srate, f_start, f_stop, n_step, n_cycles, sigma_power_chans

# FUNCTIONS 
def complex_mw(a , time, n_cycles , freq, m = 0): 
    """
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

def compute_tf(sig, srate, f_start, f_stop, n_step, n_cycles, wavelet_duration = 2, squaring=True, increase = 'linear', extracted_feature = 'power'):

    a = 1 # amplitude of the cmw
    m = 0 # max time point of the cmw
    time_cmw = np.arange(-wavelet_duration,wavelet_duration,1/srate) # time vector of the cmw

    range_freqs = np.linspace(f_start,f_stop,n_step) 

    time_sig = np.arange(0, sig.size / srate , 1 / srate)

    shape = (range_freqs.size , time_sig.size)
    data = np.zeros(shape)
    dims = ['freqs','time']
    coords = {'freqs':range_freqs, 'time':time_sig}
    tf = xr.DataArray(data = data, dims = dims, coords = coords)
    
    for i, fi in enumerate(range_freqs):
        cmw_f = complex_mw(a=a, time=time_cmw, n_cycles=n_cycles, freq=fi, m = m) # make the complex mw
        complex_conv = signal.convolve(sig, cmw_f, mode = 'same')

        if squaring:
            module = np.abs(complex_conv) ** 2
        else:
            module = np.abs(complex_conv) # abs method without squaring (more "real")
            
        tf.loc[fi,:] = module

    return tf


# RUN 
for subject in subjects:
    print(subject)
    data = xr.open_dataarray(f'../preproc/{subject}_reref.nc') # load preprocessed data (lazy load)
    data_eeg = data.sel(chan = eeg_mono_chans) # keep only eeg data
    
    sigma_power = None 
    for computed_chan in sigma_power_chans: # loop on parametred chans
        print(computed_chan)
        sig = data_eeg.sel(chan = computed_chan).values # select sig from chan
        whole_tf = compute_tf(sig, srate, f_start, f_stop, n_step, n_cycles) # compute tf with set params
        if sigma_power is None:
            sigma_power = gh.init_da({'chan':sigma_power_chans, 'freq':whole_tf.coords['freqs'].values, 'time':whole_tf.coords['time'].values}) # init dataarray with desired shapes
        sigma_power.loc[computed_chan,:,:] = whole_tf.values # fill xarray at the chan with the whole night tf map of the chan
    
    sigma_power.to_netcdf(f'../sigma/{subject}_morlet_sigma.nc') # save
        
    

