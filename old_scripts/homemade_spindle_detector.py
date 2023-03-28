import numpy as np
import ghibtools as gh

### LOW-LEVEL
def get_sigma_env(sig, srate, lowcut = 11, highcut = 16):
    sigma = gh.iirfilt(sig, srate, lowcut, highcut)
    sigma_env = gh.get_amp(sigma)
    return sigma_env

def detect_sigma_bursts(env, n_mads):
    rises, = np.where((env[:-1] <=n_mads) & (env[1:] >n_mads)) # detect where crossing n_mads but rising
    decays, = np.where((env[:-1] >=n_mads) & (env[1:] <n_mads)) # detect where crossing n_mads but decaying
    if rises[0] > decays[0]: # first point detected has to be a rise
        decays = decays[1:] # so remove the first decay if is before first rise
    if rises[-1] > decays[-1]: # last point detected has to be a decay
        rises = rises[:-1] # so remove the last rise if is after last decay

    return pd.DataFrame.from_dict({'rises':rises, 'decays':decays}, orient = 'index').T

def compute_sigma_bursts_features(sigma_env, sigma_bursts, srate):
    features = []
    for i , row in sigma_bursts.iterrows(): # last cycle is probably not complete so it is removed in any case
        if i != sigma_bursts.index[-1]:
            
            start_ind = int(row['rises'])
            stop_ind = int(row['decays'])
            
            start_t = start_ind / srate
            stop_t = stop_ind / srate
            
            burst_duration = stop_t - start_t
            peak_amp = np.max(sigma_env[start_ind:stop_ind])
            peak_ind, = np.nonzero(sigma_env == peak_value)[0]
            peak_t = peak_ind / srate
            
            burst_integral = np.trapz(sigma_env[start_ind:stop_ind])
            
            features.append([start_ind, stop_ind, start_t, stop_t, burst_duration,peak_ind, peak_t, peak_amp, burst_integral])
    df_features = pd.DataFrame(features, columns = ['start_ind','stop_ind','start_t','stop_t','burst_duration','peak_ind','peak_t','peak_amplitude','burst_integral'])
    return df_features   

def clean_bursts(burst_features, min_duration=0.5, max_duration=3):
    return burst_features[(burst_features['burst_duration'] > min_duration) & (burst_features['burst_duration'] < max_duration)].reset_index(drop = True)

### HIGH-LEVEL

def detect_spindles(sig, srate, n_mads=5, sigma_low=11, sigma_high=16, min_duration=0.5, max_duration = 3):
    sigma_env = get_sigma_env(sig, srate, sigma_low, sigma_high)
    sigma_bursts = detect_sigma_bursts(sigma_env, n_mads)
    sigma_bursts_features = compute_sigma_bursts_features(sigma_env, sigma_bursts, srate)
    clean_sigma_bursts_features = clean_bursts(sigma_bursts_features, min_duration=min_duration, max_duration=max_duration)
    return clean_sigma_bursts_features

def eeg_da_spindles(da, srate, n_mads=5, sigma_low=11, sigma_high=16, min_duration=0.5, max_duration = 3, computing_chans = None, verbose=False):
    chan_spindles_concat = []
    if computing_chans is None:
        chans = da.coords['chan'].values
    else:
        chans = computing_chans
    for chan in chans:
        if verbose:
            print(chan)
        sig = da.loc[chan,:].data
        spindles = detect_spindles(sig, srate, n_mads, sigma_low, sigma_high, min_duration, max_duration)
        spindles['channel'] = chan
        chan_spindles_concat.append(spindles)
    return pd.concat(chan_spindles_concat)

