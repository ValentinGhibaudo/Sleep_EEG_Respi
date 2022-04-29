import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack
import xarray as xr
import joblib
import pandas as pd
import mne

def notch(sig, fs):
    sig_notched = mne.filter.notch_filter(sig, Fs=fs, freqs=np.arange(50,101,50),  verbose=False)
    return sig_notched
    
def get_wsize(srate, lowest_freq , n_cycles=5):
    nperseg = ( n_cycles / lowest_freq) * srate
    return int(nperseg)

def get_memory(numpy_array):
    memory = numpy_array.nbytes * 1e-9
    print(f'{memory} Go')
    
def filter_sig(sig,fs, low, high , order=1, btype = 'mne', show = False):
    if btype == 'bandpass':
        # Paramètres de notre filtre :
        fe = fs
        f_lowcut = low
        f_hicut = high
        nyq = 0.5 * fe
        N = order                # Ordre du filtre
        Wn = [f_lowcut/nyq,f_hicut/nyq]  # Nyquist frequency fraction

        # Création du filtre :
        b, a = signal.butter(N, Wn, btype)

        # Calcul de la reponse en fréquence du filtre
        w, h = signal.freqz(b, a)

        if show:
            # Tracé de la réponse en fréquence du filtre
            fig, ax = plt.subplots(figsize=(8,5)) 
            ax.plot(0.5*fe*w/np.pi, np.abs(h), 'b')
            ax.set_xlabel('frequency [Hz]')
            ax.set_ylabel('Amplitude [dB]')
            ax.grid(which='both', axis='both')
            plt.show()

        # Applique le filtre au signal :
        filtered_sig = signal.filtfilt(b, a, sig)
        
    elif btype == 'lowpass':
        
        # Paramètres de notre filtre :
        fe = fs
        f_hicut = high
        nyq = 0.5 * fe
        N = order                  # Ordre du filtre
        Wn = f_hicut/nyq  # Nyquist frequency fraction

        # Création du filtre :
        b, a = signal.butter(N, Wn, btype)

        # Calcul de la reponse en fréquence du filtre
        w, h = signal.freqz(b, a)

        if show:
            # Tracé de la réponse en fréquence du filtre
            fig, ax = plt.subplots(figsize=(8,5)) 
            ax.plot(0.5*fe*w/np.pi, np.abs(h), 'b')
            ax.set_xlabel('frequency [Hz]')
            ax.set_ylabel('Amplitude [dB]')
            ax.grid(which='both', axis='both')
            plt.show()

        # Applique le filtre au signal :
        filtered_sig = signal.filtfilt(b, a, sig)
        
    elif btype == 'highpass':
        
        # Paramètres de notre filtre :
        fe = fs
        f_lowcut = low
        nyq = 0.5 * fe
        N = order                  # Ordre du filtre
        Wn = f_lowcut/nyq  # Nyquist frequency fraction

        # Création du filtre :
        b, a = signal.butter(N, Wn, btype)

        # Calcul de la reponse en fréquence du filtre
        w, h = signal.freqz(b, a)

        if show:
            # Tracé de la réponse en fréquence du filtre
            fig, ax = plt.subplots(figsize=(8,5)) 
            ax.plot(0.5*fe*w/np.pi, np.abs(h), 'b')
            ax.set_xlabel('frequency [Hz]')
            ax.set_ylabel('Amplitude [dB]')
            ax.grid(which='both', axis='both')
            plt.show()

        # Applique le filtre au signal :
        filtered_sig = signal.filtfilt(b, a, sig)
        
    elif btype == 'mne':
        filtered_sig = mne.filter.filter_data(sig, sfreq=fs, l_freq = low, h_freq = high, verbose = False)
        
    return filtered_sig

def norm(data):
    data = (data - np.mean(data)) / np.std(data)
    return data

def detrend(sig):
    dentrended = signal.detrend(sig)
    return dentrended

def center(sig):
    sig_centered = sig - np.mean(sig)
    return sig_centered

def time_vector(sig, srate):
    time = np.arange(0, sig.size / srate , 1 / srate)
    return time

def down_sample(sig, factor): 
    sig_down = signal.decimate(sig, q=factor, n=None, ftype='iir', axis=- 1, zero_phase=True)
    return sig_down

def spectre(sig, srate, wsize):
    nperseg = int(wsize * srate)
    nfft = nperseg * 2
    f, Pxx = signal.welch(sig, fs=srate, nperseg = nperseg , nfft = nfft, scaling='spectrum')
    # print(f.size)
    return f, Pxx

def coherence(sig1,sig2, srate, wsize):
    nperseg = int(wsize * srate)
    nfft = nperseg * 2
    f, Cxy = signal.coherence(sig1,sig2, fs=srate, nperseg = nperseg , nfft = nfft )
    # print(f.size)
    return f, Cxy

def init_da(coords, name = None):
    dims = list(coords.keys())
    coords = coords

    def size_of(element):
        element = np.array(element)
        size = element.size
        return size

    shape = tuple([size_of(element) for element in list(coords.values())])
    data = np.zeros(shape)
    da = xr.DataArray(data=data, dims=dims, coords=coords, name = name)
    return da

def parallelize(iterator, function, n_jobs):
    result = joblib.Parallel(n_jobs = n_jobs, prefer = 'threads')(joblib.delayed(function)(i) for i in iterator)
    return result



def shuffle_sig_one_inversion(sig):
    half_size = sig.shape[0]//2
    ind = np.random.randint(low=0, high=half_size)
    sig2 = sig.copy()
    
    sig2[ind:ind+half_size] *= -1
    if np.random.rand() >=0.5:
        sig2 *= -1

    return sig2

def shuffle_sig_one_break(sig):
    ind = np.random.randint(low=0, high=sig.shape[0])
    sig2 = np.hstack([sig[ind:], sig[:ind]])
    return sig2

def make_coherence_table(dict_Cxy):
    rows = []
    for ieeg_chan in list_chan_names:    
        for bloc in list_blocs:
            Cxy = dict_Cxy[ieeg_chan][bloc]
            argmax = np.argmax(Cxy[idx_min:idx_max])
            if argmax == 0:
                max_freq = np.nan
                value_at_max = np.nan
            else:
                max_freq = keep_f[argmax]
                value_at_max = Cxy[idx_min:idx_max][argmax]
            row = [ieeg_chan, bloc, max_freq, value_at_max]
            rows.append(row)
    df = pd.DataFrame(rows, columns=['chan', 'bloc', 'freq_at_max', 'value_at_max'])
    return df

def make_cxy_shuffle_sig(sig, sig_target, srate, nperseg, noverlap, nfft, iterations, percentile):
    
    shape = ( iterations , nperseg + 1 )
    shuffle_array = np.zeros(shape)
    
    for i in range(iterations):
        
        half_size = sig.shape[0]//2
        ind = np.random.randint(low=0, high=half_size)
        shuffle = sig.copy()

        shuffle[ind:ind+half_size] = - shuffle[ind:ind+half_size] 
        if np.random.rand() >=0.5:
            shuffle = -shuffle
        
        f, Cxy = signal.coherence(shuffle, sig_target, fs=srate, window='hann', nperseg=nperseg, noverlap=None, nfft=nfft)
        Cxy = Cxy.reshape(1, Cxy.shape[0])
        
        shuffle_array[i,:] = Cxy
    
    Cxy_shuffle = np.percentile(shuffle_array, percentile , axis = 0)

    return f, Cxy_shuffle


def make_cxy_shuffle_sig_parallel(sig, sig_target, srate, nperseg, nfft, iterations, percentile, n_jobs):
    
	def make_Cxy_shuffle(i):

		half_size = sig.shape[0]//2
		ind = np.random.randint(low=0, high=half_size)
		shuffle = sig.copy()

		shuffle[ind:ind+half_size] = - shuffle[ind:ind+half_size] 
		if np.random.rand() >=0.5:
			shuffle = -shuffle

		f, Cxy = signal.coherence(shuffle, sig_target, fs=srate, window='hann', nperseg=nperseg, noverlap=None, nfft=nfft)
		Cxy = Cxy.reshape(1, Cxy.shape[0])
			
		return Cxy
	
	shuffle_array = joblib.Parallel(n_jobs = n_jobs, prefer = 'threads')(joblib.delayed(make_Cxy_shuffle)(i) for i in range(iterations))
	shuffle_array = np.concatenate(shuffle_array, axis = 0)
    
	Cxy_shuffle = np.percentile(shuffle_array, percentile , axis = 0)

	return Cxy_shuffle

