import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from params import srate, respi_chan, filter_resp, resp_shifting, clean_resp_features, subjects
from mne.filter import filter_data

"""
This script allows detection of respiratory cycles and compute features for each cycle, for all subjects
"""

def iirfilt(sig, srate, lowcut=None, highcut=None, order = 4, ftype = 'butter', verbose = False, show = False): # IIR-FILTER

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

    if verbose:
        print(f'{ftype} iirfilter of {order}th-order')
        print(f'btype : {btype}')


    if show:
        w, h = signal.sosfreqz(sos,fs=srate)
        fig, ax = plt.subplots()
        ax.plot(w, np.abs(h))
        ax.set_title('Frequency response')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        plt.show()

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

save = True

for subject in subjects:
    print(subject)

    resp_signal = xr.open_dataarray(f'../preproc/{subject}_reref.nc').sel(chan = respi_chan).data
    resp_zscored = robust_zscore(resp_signal) # signal is centered and reduced to set on the same scale for the subjects
    resp_filtered = iirfilt(sig=resp_zscored, srate=srate, lowcut=filter_resp['lowcut'], highcut=filter_resp['highcut'], order=filter_resp['order'], ftype = filter_resp['ftype']) # filtering signal
    resp_shifted = resp_filtered + resp_shifting # Shift respi baseline a little bit to detect zero-crossings above baseline noise
    resp_features = get_resp_features(resp_shifted, srate) # compute resp features 
    mask_duration = (resp_features['cycle_duration'] > clean_resp_features['cycle_duration']['min']) & (resp_features['cycle_duration'] < clean_resp_features['cycle_duration']['max']) # keep cycles according to their duration
    mask_cycle_ratio = resp_features['cycle_ratio'] < clean_resp_features['cycle_ratio'] # inspi / expi duration time ratio has to be lower than set value
    mask_second_volume = resp_features['second_volume'] > clean_resp_features['second_volume'] # zscored air volume mobilized by second by resp cycle has to be higher than set value
    keep = mask_duration & mask_cycle_ratio & mask_second_volume
    resp_features_clean = resp_features[keep] # apply masking
    if save:
        resp_features_clean.reset_index(drop=True).to_excel(f'../resp_features/{subject}_resp_features.xlsx')


    N_before_cleaning = resp_features.shape[0]
    N_after_cleaning = resp_features_clean.shape[0]
    N_removed_by_cleaning = N_before_cleaning - N_after_cleaning
    print('N cycles removed :', N_removed_by_cleaning)

    ### PLOT DISTRIBUTIONS OF RESPIRATION METRICS BEFORE CLEANING
    metrics = np.array(['cycle_duration','inspi_duration','expi_duration','cycle_freq','cycle_ratio','inspi_amplitude','expi_amplitude','cycle_amplitude','inspi_volume','expi_volume','cycle_volume','second_volume']).reshape(4,3)
    nrows = metrics.shape[0]
    ncols = metrics.shape[1]

    fig, axs = plt.subplots(nrows = nrows, ncols = ncols , figsize = (20,10), constrained_layout = True)
    fig.suptitle(f'{subject} - N : {N_before_cleaning} cycles')
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
                ax.axvline(x = clean_resp_features['second_volume'], color = 'r', label = 'remove threshold')
            ax.legend()
    if save:
        plt.savefig(f'../view_respi_detection/{subject}_resp_detection_distribution', bbox_inches = 'tight')
    plt.show()
    plt.close()

    ### PLOT DISTRIBUTIONS OF RESPIRATION METRICS AFTER CLEANING
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols , figsize = (20,10), constrained_layout = True)
    fig.suptitle(f'{subject} - N : {N_after_cleaning} cycles')
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r, c]
            metric = metrics[r,c]
            vector = resp_features_clean[metric].values
            ax.hist(vector, bins = 200)
            ax.set_title(f'Mean : {np.mean(vector).round(3)} - Median : {np.median(vector).round(3)}')
            ax.set_xlabel(metric)
            ax.set_ylabel('N')

    if save:
        plt.savefig(f'../view_respi_detection/{subject}_resp_detection_distribution_clean', bbox_inches = 'tight')
    plt.show()
    plt.close()

    ### PLOT DETECTED CYCLES ON THE RAW SIGNAL

    markers = {'start':'r','transition':'g'}
  
    fig, ax = plt.subplots(figsize = (15,10))
    ax.plot(resp_zscored)
    ax.plot(resp_filtered)
    for marker in markers.keys():
        ax.plot(resp_features_clean[marker], resp_zscored[resp_features_clean[marker]], 'o', color = markers[marker], label = f'{marker}')
    ax.legend()
    ax.set_title(subject)

    if save:
        plt.savefig(f'../view_respi_detection/{subject}_resp_detection')

    plt.show()
    plt.close()