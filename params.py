# RUN KEYS
patients = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20']
# patients = ['P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20']
# patients = ['P1']
subjects = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20']


# USEFUL LISTS AND VARIABLES
eeg_chans = ['Fp2-C4' , 'Fz-Cz', 'Fp1-C3', 'C4-T4','C3-T3', 'Cz-Pz','T4-O2','T3-O1']
eeg_mono_chans = ['Fp2','Fp1','Fz','C4','C3','Cz','T4','T3','Pz','O1','O2']
dérivations = ['Fp2-C4' , 'Fz-Cz', 'Fp1-C3', 'C4-T4','C3-T3', 'Cz-Pz','T4-O2','T3-O1','EOGDt-A1', 'EOGG-A2'] 
ecg_chan = 'ECG'
eog_chans = ['EOGDt-A2','EOGG-A1']
sel_chans = ['Fp2-C4','C4-T4','T4-O2','Fz-Cz','Cz-Pz','Fp1-C3','C3-T3','T3-O1','DEBIT','THERM','ECG']
stages_labels = ['W','R','N1','N2','N3']
srate = 256
HP = 0.17
LP = 100
get_num = lambda x : x.split('P')[1]
subjects_from_patient = {patient:f'S{get_num(patient)}' for patient in patients}

# RESPI DETECTION PARAMS
respi_chan = 'DEBIT' # Define which respi channel is used for respiration cycle detection
clean_resp_features = {
    'cycle_duration':{'min':1,'max':15},
    'expi_amplitude':{
        'P1':0.2,'P2':0.2,'P3':0.5,'P4':0.4,'P5':0.3,'P6':0.1,'P7':0.3,'P8':0.5,'P9':0.3,'P10':0.5,
        'P11':0.25,'P12':0.15,'P13':0.2,'P14':0.2,'P15':0.25,'P16':0.3,'P17':0.3,'P18':0.1,'P19':0.2,'P20':0.5
        }
} # Define absolute criteria of filtering of respiration cycles
filter_resp = {'lowcut':None, 'highcut':2} # Define how to filter respiration signal before zero crossing detection
resp_shifting = -0.05 # Shift respi baseline a little bit to detect zero-crossings above baseline noise

# SLEEP STAGING
chan_sleep_staging = 'C4-T4' # sleep staging computed on this chan


# SPINDLES DETECTION PARAMS
freq_sp = (11, 16) # Spindles frequency range. YASA uses a FIR filter (implemented in MNE) with a 1.5Hz transition band, which means that for freq_sp = (12, 15 Hz), the -6 dB points are located at 11.25 and 15.75 Hz.
sp_duration = (0.5, 2) # The minimum and maximum duration of the spindles. In secs
sp_min_distance = 500 # If two spindles are closer than min_distance (in ms), they are merged into a single spindles. in milliseconds

# 'rel_pow': Relative power (= power ratio freq_sp / freq_broad).
# 'corr': Moving correlation between original signal and sigma-filtered signal.
# 'rms': Number of standard deviations above the mean of a moving root mean square of sigma-filtered signal.
sp_thresh = {'corr': 0.69, 'rel_pow': 0.3, 'rms': 1.6} # default = 0.65 , 0.2 , 1.5 

"""A sleep spindle detection algorithm that emulates human expert spindle scoring, Karine Lacours et al., 2019
- rel_pow = relative sigma power to ensure the increase of power is specific to the sigma band
- corr = moving correlation to identify a high correlation between sigma vs broadband. A high sigma correlation will indicate that the changes in sigma
result in the change in broadband
"""


# SLOW WAVES DETECTION PARAMS
freq_sw = (0.3, 1.5) # Slow wave frequency range, in Hz
sw_dur_neg = (0.3, 1.5) # The minimum and maximum duration of the negative deflection of the slow wave, in secs
sw_dur_pos = (0.1, 1) # The minimum and maximum duration of the positive deflection of the slow wave, in secs
sw_amp_neg = (50,200) # Absolute minimum and maximum negative trough amplitude of the slow-wave. In µV
sw_amp_pos = (10,150) # Absolute minimum and maximum positive peak amplitude of the slow-wave. In µV
sw_amp_ptp = (75,350) # Minimum and maximum peak-to-peak amplitude of the slow-wave. In µV

"""The Sleep Slow Oscillation as a Traveling Wave, Massimini et al, 2004 : The criteria for the detection of the slow oscillation were applied independently to each local average (bandpass, 0.1–4 Hz) and were
as follows (Fig. 1B, inset): (1) a negative zero crossing and a subsequent
positive zero crossing separated by 0.3–1.0 sec, (2) a negative peak between the two zero crossings with voltage less than 80 µV, and (3) a
negative-to-positive peak-to-peak amplitude 140 µV. Criteria 1 and 2 were similar to those used in Molle et al. (2002)."""


# MORLET WAVELETS PARAMS FOR EXTRACTION OF SIGMA POWER 
sigma_power_chans = ['Fp2-C4' , 'Fz-Cz', 'Fp1-C3', 'C4-T4','C3-T3', 'Cz-Pz','T4-O2','T3-O1']
f_start = 10 # start frequency of computing, in Hz
f_stop = 16 # stop frequency of computing, in Hz
n_step = 30 # number of frequencies computed between f_start and f_stop
cycle_start = 10 # number of morlet wavelet cycles of the f_start
cycle_stop = 10 # number of morlet wavelet cycles of the f_stop


### EVENT COUPLING ###
# EVENTS TIMESTAMPS LABELS TO SUMMARIZE AN EVENT (SPINDLE or SLOWWAVE) TO ON TIMING
timestamps_labels = {'sp':'Peak','sw':'NegPeak'} # labels = colnames of the yasa detection output
# CHANNELS EVENTS TO KEEP
# channels_events_select =  ['Fp2-C4' , 'Fz-Cz', 'Fp1-C3', 'C4-T4','C3-T3', 'Cz-Pz','T4-O2','T3-O1'] # only events detected in these channels are kept
channels_events_select =  ['Fp2-C4' , 'Fz-Cz', 'Fp1-C3', 'C4-T4','C3-T3', 'Cz-Pz'] # only events detected in these channels are kept
# STAGE EVENTS TO KEEP
stages_events_select =  ['R','N1','N2','N3'] # only events detected during these sleep stages are kept

### EVENT STATS ###
interesting_variables = {
    'sp':['Duration','Amplitude','RMS','AbsPower','RelPower','Frequency','Oscillations','Symmetry'],
    'sw':['Duration','ValNegPeak','ValPosPeak','PTP','Slope','Frequency']
}


### SIGMA COUPLING ###
transition_ratio = 0.4 # phase point (in %) corresponding to inspi to expi transition on phase-frequency matrix
nb_point_by_cycle = 100 # number of phase bins for phase-frequency matrix
channels_sigma_select =  ['Fp2-C4' , 'Fz-Cz', 'Fp1-C3', 'C4-T4', 'C3-T3', 'Cz-Pz','T4-O2','T3-O1'] # only sigma extracted in these channels is kept
stages_sigma_select =  ['N2','N3'] # only sigma extracted during these sleep stages is kept


### FIGURES
dpis = 100 # dots per inch = resolution of figs
sigma_coupling_chan_means = ['C4-T4','C3-T3'] # global sigma coupling fig with just one matrix by patient will average phase-freq from these chans
fig_global_cycle_type = 'spindled' # global sigma coupling fig with just one matrix by patient will select this type of tagging of resp cycles

