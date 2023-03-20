# RUN KEYS
subjects = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20']
# subjects = ['S12','S13','S14','S15','S16','S17','S18','S19','S20']
# subjects = ['S4']


# USEFUL LISTS AND VARIABLES
eeg_chans = ['Fp2-C4' , 'Fz-Cz', 'Fp1-C3', 'C4-T4','C3-T3', 'Cz-Pz','T4-O2','T3-O1']
eeg_mono_chans = ['Fp2','Fp1','Fz','C4','C3','Cz','T4','T3','Pz','O1','O2']
dérivations = ['Fp2-C4' , 'Fz-Cz', 'Fp1-C3', 'C4-T4','C3-T3', 'Cz-Pz','T4-O2','T3-O1','EOGDt-A1', 'EOGG-A2'] 
ecg_chan = 'ECG'
eog_chans = ['EOGDt-A1','EOGG-A2']
sel_chans = ['Fp2-C4','C4-T4','T4-O2','Fz-Cz','Cz-Pz','Fp1-C3','C3-T3','T3-O1','DEBIT','THERM','ECG']
stages_labels = ['W','R','N1','N2','N3']
srate = 256

# SLEEP STAGING
chan_sleep_staging = 'C4' # sleep staging computed on this chan
mapper_human_stages_to_yasa_stages = {'W  ':'W','N1 ':'N1','N2 ':'N2','N3 ':'N3','REM':'R'} # stages from human hypnogram to yasa corresponding labels
mapper_yasa_encoding = {'W':0,'N1':1,'N2':2,'N3':3,'R':4} # encoding of str stages to int stages codes

# RESPI DETECTION PARAMS
respi_chan = 'DEBIT' # Define which respi channel is used for respiration cycle detection
filter_resp = {'lowcut':None, 'highcut':1.5, 'order':4, 'ftype':'butter'} # Define how to filter respiration signal before zero crossing detection
resp_shifting = -0.02 # Shift respi baseline a little bit to detect zero-crossings above baseline noise
# Define absolute criteria of filtering of respiration cycles
clean_resp_features = {
    'cycle_duration':{'min':1,'max':15}, # secs
    'cycle_ratio':0.6, # inspi / expi duration time ratio has to be lower than this value
    'second_volume':20 # zscored air volume mobilized by second by resp cycle has to be higher than this value
} 


# SPINDLES DETECTION PARAMS
freq_sp = (12, 15) # Spindles frequency range. YASA uses a FIR filter (implemented in MNE) with a 1.5Hz transition band, which means that for freq_sp = (12, 15 Hz), the -6 dB points are located at 11.25 and 15.75 Hz.
sp_duration = (0.5, 2) # The minimum and maximum duration of the spindles. In secs 
sp_min_distance = 500 # If two spindles are closer than min_distance (in ms), they are merged into a single spindles. in milliseconds

# 'rel_pow': Relative power (= power ratio freq_sp / freq_broad).
# 'corr': Moving correlation between original signal and sigma-filtered signal.
# 'rms': Number of standard deviations above the mean of a moving root mean square of sigma-filtered signal.
sp_thresh = {'corr': 0.65, 'rel_pow': 0.2, 'rms': 1.5} # default = 0.65 , 0.2 , 1.5 
remove_outliers_sp = False # remove or not outliers by YASA using sklearn.ensemble.IsolationForest. 
include_sp_stages = (2,3) # 



"""A sleep spindle detection algorithm that emulates human expert spindle scoring, Karine Lacours et al., 2019
- rel_pow = relative sigma power to ensure the increase of power is specific to the sigma band
- corr = moving correlation to identify a high correlation between sigma vs broadband. A high sigma correlation will indicate that the changes in sigma
result in the change in broadband
"""

# SLOW VS FAST SPINDLES SUBJECTS & THRESHOLDS
spindles_freq_threshold = {'S1':13.7,'S2':12.9,'S3':13.3,'S4':12.7,'S5':13.6,'S6':12.8,'S7':12.8,'S8':12.2,'S9':12.7,'S10':13.1,
                           'S11':13.4,'S12':13.1,'S13':13,'S14':13,'S15':13.4,'S16':13.3,'S17':13,'S18':12.9,'S19':12.8,'S20':13.1}


# SLOW WAVES DETECTION PARAMS
remove_outliers_sw = False # remove or not outliers by YASA using sklearn.ensemble.IsolationForest. 
include_sw_stages = (2,3)

# freq_sw = (0.3, 1.5) # Slow wave frequency range, in Hz
# sw_dur_neg = (0.3, 1.5) # The minimum and maximum duration of the negative deflection of the slow wave, in secs
# sw_dur_pos = (0.1, 1) # The minimum and maximum duration of the positive deflection of the slow wave, in secs
# sw_amp_neg = (30,250) # Absolute minimum and maximum negative trough amplitude of the slow-wave. In µV 
# sw_amp_pos = (10,150) # Absolute minimum and maximum positive peak amplitude of the slow-wave. In µV
# sw_amp_ptp = (40,350) # Minimum and maximum peak-to-peak amplitude of the slow-wave. In µV

freq_sw = (0.3, 1.5) # Slow wave frequency range, in Hz
sw_dur_neg = (0.3, 1.5) # The minimum and maximum duration of the negative deflection of the slow wave, in secs
sw_dur_pos = (0.1, 1) # The minimum and maximum duration of the positive deflection of the slow wave, in secs
sw_amp_neg = (40,200) # Absolute minimum and maximum negative trough amplitude of the slow-wave. In µV 
sw_amp_pos = (10,150) # Absolute minimum and maximum positive peak amplitude of the slow-wave. In µV
sw_amp_ptp = (75,350) # Minimum and maximum peak-to-peak amplitude of the slow-wave. In µV

"""The Sleep Slow Oscillation as a Traveling Wave, Massimini et al, 2004 : The criteria for the detection of the slow oscillation were applied independently to each local average (bandpass, 0.1–4 Hz) and were
as follows (Fig. 1B, inset): (1) a negative zero crossing and a subsequent
positive zero crossing separated by 0.3–1.0 sec, (2) a negative peak between the two zero crossings with voltage less than 80 µV, and (3) a
negative-to-positive peak-to-peak amplitude 140 µV. Criteria 1 and 2 were similar to those used in Molle et al. (2002)."""


# MORLET WAVELETS PARAMS FOR EXTRACTION OF SIGMA POWER 
decimate_factor = 5 # down sample sig before tf computation for faster computing and less memory used
srate_down = srate / decimate_factor 
sigma_power_chans = ['Fp2','Fp1','Fz','C4','C3','Cz']
f_start = 10 # start frequency of computing, in Hz
f_stop = 16 # stop frequency of computing, in Hz
n_step = 40 # number of frequencies computed between f_start and f_stop
n_cycles = 20 # number of morlet wavelet cycles
amplitude_exponent = 2 # raise factor to amplify values 


### EVENT COUPLING ###
# ENCODER OF HYPNOGRAM THAT LABELIZED STAGING OF EVENTS (HUMAN or YASA)
encoder_events = 'yasa'
# EVENTS TIMESTAMPS LABELS TO SUMMARIZE AN EVENT (SPINDLE or SLOWWAVE) TO ON TIMING
timestamps_labels = {'sp':'Start','sw':'NegPeak'} # labels = colnames of the yasa detection output
# CHANNELS EVENTS TO KEEP
channels_events_select =  ['Fp2','Fp1','Fz','C4','C3','Cz','T4','T3','Pz','O1','O2'] # only events detected in these channels are kept
# STAGE EVENTS TO KEEP
stages_events_select =  ['N2','N3'] # only events detected during these sleep stages are kept

### EVENT STATS ###
interesting_variables = {
    'sp':['Duration','Amplitude','RMS','AbsPower','RelPower','Frequency','Oscillations','Symmetry'],
    'sw':['Duration','ValNegPeak','ValPosPeak','PTP','Slope','Frequency']
}


### SIGMA COUPLING ###
transition_ratio = 0.4 # phase point (in %) corresponding to inspi to expi transition on phase-frequency matrix
nb_point_by_cycle = 40 # number of phase bins for phase-frequency matrix
channels_sigma_select =  ['Fp2','Fp1','Fz','C4','C3','Cz'] # only sigma extracted in these channels is kept
stages_sigma_select =  ['N2','N3'] # only sigma extracted during these sleep stages is kept


### FIGURES
dpis = 100 # dots per inch = resolution of figs
sigma_coupling_chan = 'Fz' # global sigma coupling fig with just one matrix by patient will average phase-freq from these chans
fig_global_cycle_type = 'spindled' # global sigma coupling fig with just one matrix by patient will select this type of tagging of resp cycles (choices = ['all','spindled','unspindled','N2','N3','diff'])

