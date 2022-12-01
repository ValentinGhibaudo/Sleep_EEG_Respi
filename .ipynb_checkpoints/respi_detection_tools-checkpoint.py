import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from signal_tools import filter_sig

def detect_zerox(sig, show = False):

    rises, = np.where((sig[:-1] <=0) & (sig[1:] >0))
    decays, = np.where((sig[:-1] >=0) & (sig[1:] <0))

    if show:
        fig, ax = plt.subplots(figsize = (15,5))
        ax.plot(sig)
        ax.plot(rises, sig[rises], 'o', color = 'r', label = 'rise')
        ax.plot(decays, sig[decays], 'o', color = 'g', label = 'decay')
        ax.set_title('Zero-crossing')
        ax.legend()
        plt.show()

    return pd.DataFrame.from_dict({'rises':rises, 'decays':decays}, orient = 'index').T

def get_cycle_features(zerox, sig, srate, show = False):
    features = []
    for i , row in zerox.iterrows():
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
            inspi_volume = np.trapz(np.abs(sig[start:transition]))
            expi_volume = np.trapz(np.abs(sig[transition:stop]))
            features.append([start, transition , stop, start_t, transition_t, stop_t, cycle_duration, inspi_duration, expi_duration, cycle_freq, cycle_ratio, inspi_amplitude, expi_amplitude, inspi_volume, expi_volume])
    df_features = pd.DataFrame(features, columns = ['start','transition','stop','start_time','transition_time','stop_time', 'cycle_duration','inspi_duration','expi_duration','cycle_freq','cycle_ratio','inspi_amplitude','expi_amplitude','inspi_volume','expi_volume'])

    if show:
        fig, ax = plt.subplots()
        ax.hist(df_features['cycle_freq'], bins = 100)
        ax.set_ylabel('n_cycles')
        ax.set_xlabel('Freq [Hz]')
        median_cycle = df_features['cycle_freq'].median()
        ax.axvline(median_cycle, linestyle = '--', color='m')
        ax.set_title(f'Median freq : {round(median_cycle, 2)}')
        plt.show()
    return df_features

def clean_features(features, criteria= 'cycle_duration', n_std = 2, verbose = True):
    sd = features[criteria].std()
    m = features[criteria].mean()
    borne_inf = m - n_std * sd
    borne_sup = m + n_std * sd
    keep_above_borne_inf = features[criteria] > borne_inf
    keep_below_borne_sup = features[criteria] < borne_sup
    keep = keep_above_borne_inf & keep_below_borne_sup
    cleaned_features = features[keep]
    if verbose:
        print(f'{features[~keep].shape[0]} cycles removed :')
        print(features[~keep]['start'])
    return cleaned_features.reset_index(drop=True)

def get_resp_features(rsp, srate, manual_baseline_correction = 0, low = 0.05, high=0.8, cleaning = True, n_std = 2, verbose = True, show = False):
    sig_centered = rsp - np.mean(rsp)
    sig_filtered = filter_sig(sig_centered, srate, low, high) + manual_baseline_correction

    if show:
        fig, ax = plt.subplots()
        ax.plot(sig_centered, label = 'raw')
        ax.plot(sig_filtered, label = 'filtered')
        ax.set_title('Filtering')
        ax.legend()
        plt.show()
    
    zerox = detect_zerox(sig_filtered, show)
    features = get_cycle_features(zerox, rsp, srate, show)

    if cleaning:
        features_return =  clean_features(features=features, n_std=n_std, verbose=verbose)
        if show:
            rises = features_return['start']
            decays = features_return['transition']
            fig, ax = plt.subplots(figsize = (15,5))
            ax.plot(sig_filtered)
            ax.plot(rises, sig_filtered[rises], 'o', color = 'r', label = 'rise')
            ax.plot(decays, sig_filtered[decays], 'o', color = 'g', label = 'decay')
            ax.set_title('Zero-crossing - corrected')
            ax.legend()
            plt.show()

            fig, ax = plt.subplots()
            ax.hist(features_return['cycle_freq'], bins = 100)
            ax.set_ylabel('n_cycles')
            ax.set_xlabel('Freq [Hz]')
            median_cycle = features_return['cycle_freq'].median()
            ax.axvline(median_cycle, linestyle = '--', color='m')
            ax.set_title(f'Median freq : {round(median_cycle, 2)}')
            plt.show()
    else:
        features_return = features

    return features_return