import numpy as np
from scipy import signal
import mne

def filter_sig(sig,fs, low, high):
    return mne.filter.filter_data(sig, sfreq=fs, l_freq = low, h_freq = high, verbose = False)