import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pyedflib import highlevel
import glob
import ghibtools as gh
import mne

def raw_to_da(patient):
    input_file = glob.glob(f'../data/{patient}/*.edf')[0]
    signals, signal_headers, header = highlevel.read_edf(input_file)
    srate = signal_headers[0]['sample_rate']
    chans = [chan_dict['label'] for chan_dict in signal_headers]
    eeg_chans = [ chan for chan in chans if 'EEG' in chan]
    eeg_chans_clean = [ chan.split(' ')[1] for chan in chans if 'EEG' in chan]
    physio_chans = ['ECG','Menton','EOGDt','EOGG','DEBIT','THERM'] 
    unit = signal_headers[0]['dimension']
    time = np.arange(0 , signals.shape[1] / srate , 1 / srate)
    dérivations = ['Fp2-C4' , 'C4-T4', 'T4-O2' , 'Fz-Cz' , 'Cz-Pz' , 'Fp1-C3', 'C3-T3', 'T3-O1', 'EOGDt-A1', 'EOGG-A2']
    duration = round(time[-1], 2)
    da = xr.DataArray(data = signals, dims = ['chan','time'] , coords = {'chan':chans , 'time':time}).assign_attrs({'srate':srate})
    return da

def eeg_mono_to_bipol(da, dérivations=['Fp2-C4' , 'C4-T4', 'T4-O2' , 'Fz-Cz' , 'Cz-Pz' , 'Fp1-C3', 'C3-T3', 'T3-O1', 'EOGDt-A1', 'EOGG-A2']):
    da_bipol = []
    for bipol in dérivations : 
        pole1, pole2 = bipol.split('-')[0] , bipol.split('-')[1]
        if pole1 in ['EOGDt','EOGG']:
            chan1 = pole1
            chan2 = f'EEG {pole2}'
        else:
            chan1 = f'EEG {pole1}'
            chan2 = f'EEG {pole2}'
        sig1 = da.loc[chan1,:]
        sig2 = da.loc[chan2,:]

        bipol_sig = sig1 - sig2
        da_bipol.append(bipol_sig)
    da_bipolaire = xr.concat(da_bipol, dim = 'chan')
    da_bipolaire = da_bipolaire.assign_coords({'chan':dérivations}).assign_attrs({'srate':da.attrs['srate']})
    return da_bipolaire

def da_to_mne_object(da):
    ch_names = list(da.coords['chan'].values)
    sfreq = da.attrs['srate']
    info = mne.create_info(ch_names, sfreq, ch_types='misc', verbose=None)
    raw = mne.io.RawArray(data = da.values, info=info, first_samp=0, copy='auto', verbose=None)
    return raw

def mne_viewer(patient):
    da_mono = raw_to_da(patient)
    da_bipol = eeg_mono_to_bipol(da_mono)
    mne_obj = da_to_mne_object(da_bipol)
    mne_obj.plot()
    plt.show()

def xr_viewer(patient, chan, start, stop):
    da_mono = raw_to_da(patient)
    da_bipol = eeg_mono_to_bipol(da_mono)
    srate = da_bipol.attrs['srate']
    sig = da_bipol.loc[chan,start:stop].values
    tf_sig = gh.tf(sig, srate=srate, f_start = 10, f_stop = 20, n_steps = 40, cycle_start = 4, cycle_stop = 10)
    print(tf_sig.coords['time'].values.shape)

    t = np.arange(0, sig.size/srate, 1/srate)
    f = tf_sig.coords['freqs'].values

    fig, axs = plt.subplots(nrows = 2)

    ax = axs[0]
    ax.plot(t, sig)

    ax = axs[1]
    ax.pcolormesh(t, f, tf_sig.values)

    plt.show()

if __name__ == "__main__":
    mne_viewer('P1')
    # xr_viewer('P1','Fp1-C3', 119,125)