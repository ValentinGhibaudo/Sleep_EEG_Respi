import numpy as np
from bibliotheque import init_da
import xarray as xr
from scipy import signal
import pandas as pd
from params import *
from preproc_staging import preproc_job
from rsp_detection import resp_tag_job
import jobtools
from configuration import *
from deform_tools import deform_to_cycle_template

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


# JOB SIGMA POWER 
def compute_sigma_power(run_key, **p):
    data = preproc_job.get(run_key)['preproc'] # load preprocessed data (lazy load)
    srate = data.attrs['srate']
    srate_down = srate / p['decimate_factor']
    
    sigma_power = None 
    for computed_chan in p['chans']: # loop on set chans
        sig = data.sel(chan = computed_chan).values # select sig from chan
        sig_down = signal.decimate(sig, q=p['decimate_factor']) # down sample sig before tf computation for faster computing and less memory used
        freqs , whole_tf = compute_tf(
            sig = sig_down, # compute tf with set params
            srate = srate_down,
            f_start = p['f_start'], 
            f_stop = p['f_stop'], 
            n_step = p['n_step'], 
            n_cycles = p['n_cycles'],
            amplitude_exponent = p['amplitude_exponent']) 
        
        if sigma_power is None:
            sigma_power = init_da({'chan':p['chans'], 'freq':freqs, 'time':np.arange(0, sig_down.size/srate_down, 1/srate_down)}) # init dataarray with desired shapes
        sigma_power.loc[computed_chan,:,:] = whole_tf # fill xarray at the chan with the whole night tf map of the chan
    
    ds = xr.Dataset()
    ds['sigma_power'] = sigma_power
    return ds

sigma_power_job = jobtools.Job(precomputedir, 'sigma_power', sigma_power_params, compute_sigma_power)
jobtools.register_job(sigma_power_job)

def test_compute_sigma_power():
    run_key = 'S1'
    sigma_power = compute_sigma_power(run_key, **sigma_power_params)['sigma_power']
    print(sigma_power)



# JOB SIGMA COUPLING WITH RESPI
def add_nan_where_cleaned_cycle(rsp): # Function that add nan where cycles have been cleaned to not be computed by deform_tools
    rsp_nan = rsp.copy() 
    for c, row in rsp.iterrows(): # loop on cycles features
        if not c == rsp.index[-1]: # do not do thet next steps in last cycle
            if row['stop'] != rsp.loc[c+1,'start']: # if start index of the next cycle is not same as stop index of the present cycle, means that there are lacking cycles between..
                rsp_nan.loc[c+0.5,:] = np.nan # .. so between these two cycles is added Nan values that will not be computed by deform_to_cycle_template at a middle loc
    return rsp_nan.sort_index().reset_index(drop=True) # sort by index to reorder nan rows at the right loc


def compute_sigma_coupling(run_key, **p):
    morlet_sigma = sigma_power_job.get(run_key)['sigma_power'] # lazy load time_frequency sigma map (chan * freq * time)
    rsp_features = resp_tag_job.get(run_key).to_dataframe() # load respi times

    mask_stages = rsp_features['sleep_stage'].isin(p['stage']) # select only sigma extracted during the stages from list "stages_sigma_select"
    rsp_features = rsp_features[mask_stages].reset_index(drop=True)
    rsp_features_nan = add_nan_where_cleaned_cycle(rsp_features) # add nan where cycles should not be deformed because too long because of lacking because of cleaning
    
    phase_freq_sigma = None

    chan_loop = p['chans']

    for channel in chan_loop: # loop over channels
        print(channel)
        data = morlet_sigma.sel(chan = channel).data.T # from chan * freq * time to time * freq * chan
        # stretch and slice phase-freq maps
        clipped_times, times_to_cycles, cycles, cycle_points, deformed_data = deform_to_cycle_template(data = data, # 0 axis = time, 1 axis = freq
                                                                                                    times = morlet_sigma.coords['time'].values , 
                                                                                                    cycle_times=rsp_features_nan[['start_time','transition_time']].values, 
                                                                                                    nb_point_by_cycle=p['nb_point_by_cycle'],
                                                                                                    inspi_ratio = p['transition_ratio'])
        
        
        # deformed_data : axis 0 = point, 1 = freq
        deformed_data_sliced_by_cycle = deformed_data.reshape(cycles.size, p['nb_point_by_cycle'], deformed_data.shape[1]) # reshape to have axis 0 = cycles, 1 = points, axis 2 = freq
        
        if phase_freq_sigma is None:
            phase_freq_sigma = init_da({'chan':morlet_sigma.coords['chan'].values,
                                           'cycle' : cycles, 
                                           'point': np.linspace(0,1,p['nb_point_by_cycle']) , 
                                           'freq':morlet_sigma.coords['freq'].values })# init xarray
            
        phase_freq_sigma.loc[channel, :, : , :] = deformed_data_sliced_by_cycle # phase-freq map is stored in xarray
        
    resp_features_stretched = rsp_features_nan.iloc[cycles,:] # resp features are masked to the true computed cycles in deform_to_cycle_template

    c_spindled = resp_features_stretched[(resp_features_stretched['Spindle_Tag'] == 1) & (resp_features_stretched['sleep_stage'].isin(p['stage']))].index.to_numpy()
    c_unspindled = resp_features_stretched[(resp_features_stretched['Spindle_Tag'] == 0) & (resp_features_stretched['sleep_stage'].isin(p['stage']))].index.to_numpy()
    c_N2 = resp_features_stretched[resp_features_stretched['sleep_stage'] == 'N2'].index.to_numpy()
    c_N3 = resp_features_stretched[resp_features_stretched['sleep_stage'] == 'N3'].index.to_numpy()
    c_all = resp_features_stretched.dropna().index.to_numpy()
    

    N_cycle_averaged = []

    c_labels = ['all','spindled','unspindled','N2','N3','diff']

    da_mean = None

    for c_label, c_type in zip(c_labels,[c_all, c_spindled, c_unspindled, c_N2,c_N3,'diff']):
        if c_label == 'diff':
            N_cycle_averaged.append(0)
            phase_freq_mean = phase_freq_sigma.sel(cycle = c_spindled).mean('cycle') - phase_freq_sigma.sel(cycle = c_unspindled).mean('cycle')
            
        else:
            N_cycle_averaged.append(c_type.size)
            phase_freq_mean = phase_freq_sigma.sel(cycle = c_type).mean('cycle')

        if da_mean is None: 
            da_mean = init_da({'cycle_type':c_labels,
                                'chan':phase_freq_sigma.coords['chan'],
                                'point':phase_freq_sigma.coords['point'],
                                  'freq':phase_freq_sigma.coords['freq'], 
                                  })
        
        da_mean.loc[c_label,:, : ,:] = phase_freq_mean.values

    da_mean.attrs['data_n_cycle_averaged'] = N_cycle_averaged
    # da_mean.attrs['label_n_cycle_averaged'] = c_labels

    ds = xr.Dataset()
    ds['sigma_coupling'] = da_mean
    return ds

sigma_coupling_job = jobtools.Job(precomputedir, 'sigma_coupling', sigma_coupling_params, compute_sigma_coupling)
jobtools.register_job(sigma_coupling_job)

def test_compute_sigma_coupling():
    run_key = 'S2'
    sigma_coupling_ds = compute_sigma_coupling(run_key, **sigma_coupling_params)
    # sigma_coupling_ds.to_netcdf(base_folder / 'precompute' / 'test.nc')
    print(sigma_coupling_ds['sigma_coupling'])



def compute_all():
    # jobtools.compute_job_list(sigma_power_job, run_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(sigma_power_job, run_keys, force_recompute=True, engine='joblib', n_jobs = 2)

    # jobtools.compute_job_list(sigma_power_job, [(sub,) for sub in run_keys], force_recompute=True, engine='slurm',
    #                           slurm_params={'cpus-per-task':'4', 'mem':'20G', },
    #                           module_name='sigma_coupling',
    #                           )

    # jobtools.compute_job_list(sigma_coupling_job, run_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(sigma_coupling_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 2)

    jobtools.compute_job_list(sigma_coupling_job, [(sub,) for sub in run_keys], force_recompute=True, engine='slurm',
                              slurm_params={'cpus-per-task':'4', 'mem':'20G', },
                              module_name='sigma_coupling',
                              )

if __name__ == '__main__':
    # test_compute_sigma_power()
    # test_compute_sigma_coupling()
    
    compute_all()


