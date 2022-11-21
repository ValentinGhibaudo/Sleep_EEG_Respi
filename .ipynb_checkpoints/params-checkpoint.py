# patients = ['P1','P2','P3','P4','P5','P6','P7','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20'] #  P8 exclude because of truncated signal and bad respi signal
# patients = ['P11','P12','P13','P14','P15','P16','P18','P19','P20']
# patients = ['P14','P18']
# patients = ['P11','P12','P13','P15','P16','P17','P18','P19','P20']
patients = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20'] 
# patients = ['P14']

# patient = 'P1' 
# patient = 'P2' 
# patient = 'P3'
# patient = 'P4' 
# patient = 'P5' 
# patient = 'P6' 
# patient = 'P7' 
# patient = 'P8' 
# patient = 'P9' 
# patient = 'P10' 
# patient = 'P11' 
# patient = 'P12' 
# patient = 'P13' 
# patient = 'P14' 
# patient = 'P15' 
# patient = 'P16' 
patient = 'P17' 
# patient = 'P18' 
# patient = 'P19' 
# patient = 'P20' 

labelling_method = 'ia' # set ia or human hypnogram chosen to label signals epochs

eeg_chans = ['Fp2-C4','C4-T4','T4-O2','Fz-Cz','Cz-Pz','Fp1-C3','C3-T3','T3-O1']
dérivations = ['Fp2-C4' , 'C4-T4', 'T4-O2' , 'Fz-Cz' , 'Cz-Pz' , 'Fp1-C3', 'C3-T3', 'T3-O1', 'EOGDt-A1', 'EOGG-A2'] 
respi_chan = 'DEBIT'
ecg_chan = 'ECG'
eog_chans = ['EOGDt-A2','EOGG-A1']
sel_chans = ['Fp2-C4','C4-T4','T4-O2','Fz-Cz','Cz-Pz','Fp1-C3','C3-T3','T3-O1','DEBIT','THERM','ECG']
stages_labels = ['W','R','N1','N2','N3']
srate = 256
HP = 0.17
LP = 100

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
