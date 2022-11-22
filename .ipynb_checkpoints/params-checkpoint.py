# RUN KEYS

# patients = ['P1','P2','P3','P4','P5','P6','P7','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20'] #  P8 exclude because of truncated signal and bad respi signal
# patients = ['P11','P12','P13','P14','P15','P16','P18','P19','P20']
# patients = ['P14','P18']
# patients = ['P11','P12','P13','P15','P16','P17','P18','P19','P20']
patients = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20']

patient = 'P17' 


# USEFUL LISTS AND VARIABLES
labelling_method = 'ia' # set ia or human hypnogram chosen to label signals epochs
eeg_chans = ['Fp2-C4' , 'Fz-Cz', 'Fp1-C3', 'C4-T4','C3-T3', 'Cz-Pz','T4-O2','T3-O1']
dérivations = ['Fp2-C4' , 'Fz-Cz', 'Fp1-C3', 'C4-T4','C3-T3', 'Cz-Pz','T4-O2','T3-O1','EOGDt-A1', 'EOGG-A2'] 
respi_chan = 'DEBIT'
ecg_chan = 'ECG'
eog_chans = ['EOGDt-A2','EOGG-A1']
sel_chans = ['Fp2-C4','C4-T4','T4-O2','Fz-Cz','Cz-Pz','Fp1-C3','C3-T3','T3-O1','DEBIT','THERM','ECG']
stages_labels = ['W','R','N1','N2','N3']
srate = 256
HP = 0.17
LP = 100


# SELECT RESPI CHAN TO DETECT RESPIRATION CYCLES
rsp_chan = {
    'P1':'DEBIT',
    'P2':'DEBIT',
    'P3':'DEBIT',
    'P4':'DEBIT',
    'P5':'DEBIT',
    'P6':'DEBIT',
    'P7':'DEBIT',
    'P8':'DEBIT',
    'P9':'DEBIT',
    'P10':'DEBIT',
    'P11':'DEBIT',
    'P12':'DEBIT',
    'P13':'DEBIT',
    'P14':'DEBIT',
    'P15':'DEBIT',
    'P16':'DEBIT',
    'P17':'DEBIT',
    'P18':'DEBIT',
    'P19':'DEBIT',
    'P20':'DEBIT'
}


# REVERSE RESP SIGNAL OR NOT
rsp_detect_sign = {
    'P1':'+',
    'P2':'+',
    'P3':'+',
    'P4':'+',
    'P5':'+',
    'P6':'+',
    'P7':'+',
    'P8':'+',
    'P9':'+',
    'P10':'+',
    'P11':'+',
    'P12':'+',
    'P13':'+',
    'P14':'+',
    'P15':'+',
    'P16':'+',
    'P17':'+',
    'P18':'+',
    'P20':'+'
}


# SPINDLES DETECTION PARAMS
sp_duration = (0.5, 2) # The minimum and maximum duration of the spindles. In secs
sp_min_distance = 500 # If two spindles are closer than min_distance (in ms), they are merged into a single spindles. in milliseconds

# 'rel_pow': Relative power (= power ratio freq_sp / freq_broad).
# 'corr': Moving correlation between original signal and sigma-filtered signal.
# 'rms': Number of standard deviations above the mean of a moving root mean square of sigma-filtered signal.
sp_thresh = {'corr': 0.65, 'rel_pow': 0.3, 'rms': 1.5} # default = 0.65 , 0.2 , 1.5


# SLOW WAVES DETECTION PARAMS
freq_sw = (0.3, 1.5) # Slow wave frequency range, in Hz
sw_dur_neg = (0.3, 1.5) # The minimum and maximum duration of the negative deflection of the slow wave, in secs
sw_dur_pos = (0.1, 1) # The minimum and maximum duration of the positive deflection of the slow wave, in secs
sw_dur_pos = (0.1, 1) # The minimum and maximum duration of the positive deflection of the slow wave, in secs
sw_amp_neg = (40,200) # Absolute minimum and maximum negative trough amplitude of the slow-wave. In µV
sw_amp_pos = (10,150) # Absolute minimum and maximum positive peak amplitude of the slow-wave. In µV
sw_amp_ptp = (75,350) # Minimum and maximum peak-to-peak amplitude of the slow-wave. In µV


# MORLET WAVELETS PARAMS FOR EXTRACTION OF SIGMA POWER 
sigma_power_chans = ['Fp2-C4' , 'C4-T4', 'T4-O2' , 'Fz-Cz' , 'Cz-Pz' , 'Fp1-C3', 'C3-T3', 'T3-O1']
f_start = 10 # start frequency of computing, in Hz
f_stop = 16 # stop frequency of computing, in Hz
n_step = 30 # number of frequencies computed between f_start and f_stop
cycle_start = 10 # number of morlet wavelet cycles of the f_start
cycle_stop = 10 # number of morlet wavelet cycles of the f_stop

