import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from params import *
from configuration import *
import jobtools
from preproc_staging import preproc_job, upsample_hypno_job
from detect_sleep_events import spindles_detect_job, slowwaves_detect_job


def iirfilt(sig, srate, lowcut=None, highcut=None, order = 4, ftype = 'butter'): # IIR-FILTER

    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)
    filtered_sig = signal.sosfiltfilt(sos, sig, axis=0)

    return filtered_sig

def detect_zerox(sig):
    rises, = np.where((sig[:-1] <=0) & (sig[1:] >0)) # detect where sign inversion from - to +
    decays, = np.where((sig[:-1] >=0) & (sig[1:] <0)) # detect where sign inversion from + to -
    if rises[0] > decays[0]: # first point detected has to be a rise
        decays = decays[1:] # so remove the first decay if is before first rise
    if rises[-1] > decays[-1]: # last point detected has to be a decay
        rises = rises[:-1] # so remove the last rise if is after last decay
    return pd.DataFrame.from_dict({'rises':rises, 'decays':decays}, orient = 'index').T

def get_cycle_features(zerox, sig, srate):
    features = []
    for i , row in zerox.iterrows(): # last cycle is probably not complete so it is removed in any case
        if i != zerox.index[-1]:
            start = int(row['rises'])
            transition = int(row['decays'])
            stop = int(zerox.loc[i+1, 'rises'])
            start_t = start / srate
            transition_t = transition / srate
            stop_t = stop / srate
            cycle_duration = stop_t - start_t
            inspi_duration = transition_t - start_t
            expi_duration = stop_t - transition_t
            cycle_freq = 1 / cycle_duration
            cycle_ratio = inspi_duration / cycle_duration
            inspi_amplitude = np.max(np.abs(sig[start:transition]))
            expi_amplitude = np.max(np.abs(sig[transition:stop]))
            cycle_amplitude = inspi_amplitude + expi_amplitude
            inspi_volume = np.trapz(np.abs(sig[start:transition]))
            expi_volume = np.trapz(np.abs(sig[transition:stop]))
            cycle_volume = inspi_volume + expi_volume
            second_volume = cycle_freq * cycle_volume
            features.append([start, transition , stop, start_t, transition_t, stop_t, cycle_duration,
                             inspi_duration, expi_duration, cycle_freq, cycle_ratio, inspi_amplitude,
                             expi_amplitude,cycle_amplitude, inspi_volume, expi_volume, cycle_volume, second_volume])
    df_features = pd.DataFrame(features, columns = ['start','transition','stop','start_time','transition_time',
                                                    'stop_time','cycle_duration','inspi_duration','expi_duration','cycle_freq','cycle_ratio',
                                                    'inspi_amplitude','expi_amplitude','cycle_amplitude','inspi_volume','expi_volume','cycle_volume','second_volume'])
    return df_features

def robust_zscore(sig): # center by median and reduce by std
    return (sig - np.median(sig)) / np.std(sig)

def get_resp_features(rsp, srate):
    zerox = detect_zerox(rsp)
    features = get_cycle_features(zerox, rsp, srate)
    return features

def get_dispersion(vector, n_sd=2, n_q=0.05):
    m = np.mean(vector)
    sd = np.std(vector)
    sd_inf = m - n_sd * sd
    sd_sup = m + n_sd * sd
    quantile_inf = np.quantile(vector, n_q)
    quantile_sup = np.quantile(vector, 1 - n_q)
    return {'sd_inf':sd_inf, 'sd_sup':sd_sup, 'quantile_inf':quantile_inf, 'quantile_sup':quantile_sup}


# JOB RESP FEATURES
def compute_resp(run_key, **p):
    data = preproc_job.get(run_key)['preproc']
    srate = data.attrs['srate']
    resp_signal = data.sel(chan = p['respi_chan']).values
    resp_zscored = robust_zscore(resp_signal) # signal is centered and reduced to set on the same scale for the subjects
    resp_filtered = iirfilt(sig=resp_zscored, srate=srate, lowcut=p['lowcut'], highcut=p['highcut'], order=p['order'], ftype = p['ftype']) # filtering signal
    resp_shifted = resp_filtered + p['resp_shifting'] # Shift respi baseline a little bit to detect zero-crossings above baseline noise
    resp_features = get_resp_features(resp_shifted, srate) # compute resp features 
    mask_duration = (resp_features['cycle_duration'] > p['cycle_duration_vlims']['min']) & (resp_features['cycle_duration'] < p['cycle_duration_vlims']['max']) # keep cycles according to their duration
    mask_cycle_ratio = resp_features['cycle_ratio'] < p['max_cycle_ratio'] # inspi / expi duration time ratio has to be lower than set value
    mask_second_volume = resp_features['second_volume'] > p['min_second_volume'] # zscored air volume mobilized by second by resp cycle has to be higher than set value
    keep = mask_duration & mask_cycle_ratio & mask_second_volume
    resp_features_clean = resp_features[keep] # apply masking
    resp_features_clean = resp_features_clean.reset_index(drop=True)

    N_before_cleaning = resp_features.shape[0]
    N_after_cleaning = resp_features_clean.shape[0]

    ### PLOT DISTRIBUTIONS OF RESPIRATION METRICS BEFORE CLEANING
    metrics = np.array(['cycle_duration','inspi_duration','expi_duration','cycle_freq','cycle_ratio','inspi_amplitude','expi_amplitude','cycle_amplitude','inspi_volume','expi_volume','cycle_volume','second_volume']).reshape(4,3)
    nrows = metrics.shape[0]
    ncols = metrics.shape[1]

    fig, axs = plt.subplots(nrows = nrows, ncols = ncols , figsize = (20,10), constrained_layout = True)
    fig.suptitle(f'{run_key} - N : {N_before_cleaning} cycles')
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r, c]
            metric = metrics[r,c]
            vector = resp_features[metric].values
            ax.hist(vector, bins = 200)
            ax.set_title(f'Mean : {np.mean(vector).round(3)} - Median : {np.median(vector).round(3)}')
            ax.set_xlabel(metric)
            ax.set_ylabel('N')
            dispersion = get_dispersion(vector, 2, 0.05)
            for i, estimator in enumerate(dispersion.keys()):
                if i < 2:
                    color = 'r'
                else:
                    color = 'g'
                ax.axvline(x = dispersion[estimator], linestyle = '--', linewidth = 0.4, color = color, label = estimator)
            if metric == 'second_volume':
                ax.axvline(x = p['min_second_volume'], color = 'r', label = 'remove threshold')
            ax.legend()
        fig.savefig(base_folder / 'results' / 'view_respi_detection' / f'{run_key}_resp_detection_distribution', bbox_inches = 'tight')
    plt.close()

    ### PLOT DISTRIBUTIONS OF RESPIRATION METRICS AFTER CLEANING
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols , figsize = (20,10), constrained_layout = True)
    fig.suptitle(f'{run_key} - N : {N_after_cleaning} cycles')
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r, c]
            metric = metrics[r,c]
            vector = resp_features_clean[metric].values
            ax.hist(vector, bins = 200)
            ax.set_title(f'Mean : {np.mean(vector).round(3)} - Median : {np.median(vector).round(3)}')
            ax.set_xlabel(metric)
            ax.set_ylabel('N')

    fig.savefig(base_folder / 'results' / 'view_respi_detection' / f'{run_key}_resp_detection_distribution_clean', bbox_inches = 'tight')
    plt.close()

    ### PLOT DETECTED CYCLES ON THE RAW SIGNAL

    markers = {'start':'r','transition':'g'}
  
    fig, ax = plt.subplots(figsize = (15,10))
    ax.plot(resp_zscored)
    ax.plot(resp_filtered)
    for marker in markers.keys():
        ax.plot(resp_features_clean[marker], resp_zscored[resp_features_clean[marker]], 'o', color = markers[marker], label = f'{marker}')
    ax.legend()
    ax.set_title(run_key)

    fig.savefig(base_folder / 'results' / 'view_respi_detection' / f'{run_key}_resp_detection')

    plt.close()

    return xr.Dataset(resp_features_clean)

resp_features_job = jobtools.Job(precomputedir, 'resp_features', resp_params, compute_resp)
jobtools.register_job(resp_features_job)

def test_compute_resp():
    run_key = 'S1'
    resp_features = compute_resp(run_key, **resp_params).to_dataframe()
    print(resp_features)


# JOB TAGGING RESP WITH SLEEP
def resp_tag(run_key, **p):

    hypno_upsampled = upsample_hypno_job.get(run_key).to_dataframe() # load upsampled hypnogram of the subject
    rsp_features = resp_features_job.get(run_key).to_dataframe() # load resp features

    idx_start = rsp_features['start'].values # select start indexes of the cycles
    stage_of_the_start_idxs = hypno_upsampled['str'][idx_start] # keep stages (strings) corresponding to start resp cycle indexes
    rsp_features_staged = rsp_features.copy()
    rsp_features_staged['sleep_stage'] = stage_of_the_start_idxs.values # append sleep stage column to resp features
    
    event_in_resp = {'spindles':[],'slowwaves':[]} # list to encode if an event is present is in the resp cycle or not

    events_load = {'spindles':spindles_detect_job.get(run_key).to_dataframe(), 'slowwaves':slowwaves_detect_job.get(run_key).to_dataframe()} # load dataframe of detected events

    for event_label in ['spindles','slowwaves']: # loop on both types of events (slow waves and spindles)
        events = events_load[event_label] 
        event_times = events[p['timestamps_labels'][event_label]].values # get np array of events timings that summarize the whole events (set in params)
        
        for c, row in rsp_features_staged.iterrows(): # loop on rsp cycles
            start = row['start_time'] # get start time of the cycle
            stop = row['stop_time'] # get stop time of the cycle
            
            events_of_the_cycle = event_times[(event_times >= start) & (event_times < stop)] # keep event times present in the cycle
            
            if events_of_the_cycle.size != 0: # if at least one event is present in the cycle...
                event_in_resp[event_label].append(1) # ... append a 1
            else:
                event_in_resp[event_label].append(0)  # ... else append a 0
            
    rsp_features_tagged = rsp_features_staged.copy()
    rsp_features_tagged['Spindle_Tag'] = event_in_resp['spindles'] # append spindle tagging column to resp features
    rsp_features_tagged['SlowWave_Tag'] = event_in_resp['slowwaves'] # append slowwave tagging column to resp features
    
    return xr.Dataset(rsp_features_tagged)

resp_tag_job = jobtools.Job(precomputedir, 'resp_tag', resp_tag_params, resp_tag)
jobtools.register_job(resp_tag_job)

def test_resp_tag():
    run_key = 'S1'
    resp_tags = resp_tag(run_key, **resp_tag_params).to_dataframe()
    print(resp_tags)


def compute_all():
    # jobtools.compute_job_list(resp_features_job, run_keys, force_recompute=False, engine='loop')
    jobtools.compute_job_list(resp_tag_job, run_keys, force_recompute=False, engine='joblib', n_jobs = 10)


if __name__ == '__main__':
    # test_compute_resp()
    # test_resp_tag()
    

    compute_all()