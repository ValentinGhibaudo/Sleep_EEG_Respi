# RUN KEYS
run_keys = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20']


# MAPPERS STAGES / INT (NEEDED FOR YASA)
mapper_stage_to_int = {'W':0,'N1':1,'N2':2,'N3':3,'R':4} # encoding of str stages to int stages codes
mapper_int_to_stage = {0:'W',1:'N1',2:'N2',3:'N3',4:'R'} # encoding of int stages codes to str stages 


# EVENTS DETECTION : COMMON PARAMS
chans_events_detect = ['Fp2','Fp1','Fz','C4','C3','Cz','T4','T3','Pz','O1','O2'] # where spindles and slowwaves are detected by yasa
include_stages = (2,3) # stages when spindles and slowwaves are detected
remove_outliers = False  # remove or not outliers of detection by YASA using sklearn.ensemble.IsolationForest.
compute_stage = ['N2','N3']
compute_chan = 'Fz'
timestamps_labels = {'spindles':'Start','slowwaves':'NegPeak'} # labels = colnames of the yasa detection output
channels_events_select =  ['Fp2','Fp1','Fz','C4','C3','Cz','T4','T3','Pz','O1','O2'] # only events detected in these channels are used for computing

# SLOW VS FAST SPINDLES SUBJECTS THRESHOLDS
spindles_freq_threshold = {'S1':13.6,
                           'S2':13.05,
                           'S3':13.55,
                           'S4':12.9,
                           'S5':13.7,
                           'S6':12.85,
                           'S7':12.8,
                           'S8':12.3,
                           'S9':13,
                           'S10':13.1,
                           'S11':13.4,
                           'S12':13.2,
                           'S13':12.95,
                           'S14':13.25,
                           'S15':13.4,
                           'S16':13.4,
                           'S17':13.2,
                           'S18':13,
                           'S19':13.05,
                           'S20':13.2}


### JOB PARAMS
metadata_params = {
}
    
preproc_params = {
    'physio_chans':['ECG','Menton','EOGDt','EOGG','DEBIT','THERM']
}

sleep_staging_params = {
    'preproc_params':preproc_params,
    'chan_sleep_staging':'C4', # sleep staging computed on this chan
    'eog_name':'EOGG-A2',
    'emg_name':'Menton',
}

upsample_hypno_params = {
    'sleep_staging_params':sleep_staging_params
}

spectrogram_params = {
    'preproc_params':preproc_params,
    'sleep_staging_params':sleep_staging_params
}

sleep_stats_params = {
    'sleep_staging_params':sleep_staging_params
}


spindles_detect_params = {
    'preproc_params':preproc_params,
    'upsample_hypno_params':upsample_hypno_params,
    'mapper':mapper_int_to_stage, # mapper from int stage to str stage (same than YASA)
    'chans':chans_events_detect , # chans where spindles are detected
    'freq_sp':(12, 15), # Spindles frequency range. YASA uses a FIR filter (implemented in MNE) with a 1.5Hz transition band, which means that for freq_sp = (12, 15 Hz), the -6 dB points are located at 11.25 and 15.75 Hz.
    'duration':(0.5, 2), # The minimum and maximum duration of the spindles. In secs 
    'min_distance': 500, # If two spindles are closer than min_distance (in ms), they are merged into a single spindles. in milliseconds
    'thresh':{'corr': 0.65, 'rel_pow': 0.2, 'rms': 1.5}, # default = 0.65 , 0.2 , 1.5
    'remove_outliers': remove_outliers, # remove or not outliers by YASA using sklearn.ensemble.IsolationForest. 
    'include': include_stages # stages when spindles are detected              
}

slowwaves_detect_params = {
    'preproc_params':preproc_params,
    'upsample_hypno_params':upsample_hypno_params,
    'mapper':mapper_int_to_stage, # mapper from int stage to str stage (same than YASA)
    'chans':chans_events_detect , # chans where spindles are detected
    'freq_sw':(0.3, 1.5), # Slow wave frequency range, in Hz
    'dur_neg':(0.3, 1.5), # The minimum and maximum duration of the negative deflection of the slow wave, in secs
    'dur_pos':(0.1, 1), # The minimum and maximum duration of the positive deflection of the slow wave, in secs
    'amp_neg':(40,200), # Absolute minimum and maximum negative trough amplitude of the slow-wave. In µV 
    'amp_pos':(10,150), # Absolute minimum and maximum positive peak amplitude of the slow-wave. In µV
    'amp_ptp':(75,350), # Minimum and maximum peak-to-peak amplitude of the slow-wave. In µV
    'remove_outliers':remove_outliers, # remove or not outliers by YASA using sklearn.ensemble.IsolationForest. 
    'include':include_stages # stages when slowwaves are detected              
}

spindles_tagging_params = {
    'spindles_detect_params':spindles_detect_params,
    'slowwaves_detect_params':slowwaves_detect_params,
    'spindles_freq_threshold':spindles_freq_threshold,
    'win_margin':0.2
}

slowwaves_tagging_params = {
    'spindles_detect_params':spindles_detect_params,
    'slowwaves_detect_params':slowwaves_detect_params,
    'win_margin':spindles_tagging_params['win_margin']
}

events_stats_params = {
    'spindles_tagging_params':spindles_tagging_params,
    'slowwaves_tagging_params':slowwaves_tagging_params,
    'stage':'N2',
    'chan':compute_chan,
    'interesting_variables':{'spindles':['Duration','Amplitude','RMS','AbsPower','RelPower','Frequency','Oscillations','Symmetry'],
                             'slowwaves':['Duration','ValNegPeak','ValPosPeak','PTP','Slope','Frequency']},
    'save_article':False
}

resp_params = {
    'preproc_params':preproc_params,
    'respi_chan':'DEBIT', # Define which respi channel is used for respiration cycle detection
    'lowcut':None,
    'highcut':1.5,
    'order':4,
    'ftype':'butter', # Define how to filter respiration signal before zero crossing detection
    'resp_shifting':-0.02, # Shift respi baseline a little bit to detect zero-crossings above baseline noise
    'cycle_duration_vlims':{'min':1,'max':15}, # secs
    'max_cycle_ratio':0.6, # inspi / expi duration time ratio has to be lower than this value
    'min_second_volume':20 # zscored air volume mobilized by second by resp cycle has to be higher than this value
} 

resp_tag_params = {
    'resp_params':resp_params,
    'upsample_hypno_params':upsample_hypno_params,
    'spindles_detect_params':spindles_detect_params,
    'slowwaves_detect_params':slowwaves_detect_params,
    'timestamps_labels':timestamps_labels # timestamp label of the event that will be considered for tagging of respi cycles
} 

resp_stats_params = {
    'resp_tag_params':resp_tag_params,
} 

events_coupling_params = {
    'spindles_tagging_params':spindles_tagging_params,
    'slowwaves_tagging_params':slowwaves_tagging_params,
    'resp_tag_params':resp_tag_params,
    'chans':chans_events_detect,
    'stage':compute_stage,
    'timestamps_labels':{'Spindles':timestamps_labels['spindles'] , 'SlowWaves':timestamps_labels['slowwaves']} # timestamp label of the event that will be considered for tagging of respi cycles
}

concat_events_coupling_params = {
    'events_coupling_params':events_coupling_params
}

events_coupling_stats_params = {
    'events_coupling_params':events_coupling_params,
    'chan':compute_chan,
    'univals':1000
}

events_coupling_figs_params = {
    'events_coupling_params':events_coupling_params,
    'stage':'N2',
    'save_article':False, # will save outputs in article folder if True
    'univals':1000,
    'with_stats':True,
    'bins':18,
    'seed':None,
}

cross_correlogram_params = {
    'resp_tag_params':resp_tag_params,
    'spindles_tagging_params':spindles_tagging_params,
    'slowwaves_tagging_params':slowwaves_tagging_params,
    'stage':compute_stage,
    'chan':compute_chan,
    'delta_spsw' : 1.5,
    'delta_t_by_bin_spsw' : 0.05,
    'timestamps_labels':timestamps_labels,
    'delta_resp':4,
    'delta_t_by_bin_resp':0.2,
}

sigma_power_params = {
    'preproc_params':preproc_params,
    'chans':channels_events_select,
    'decimate_factor':5, # down sample sig before tf computation for faster computing and less memory used
    'f_start':10, # start frequency of computing, in Hz
    'f_stop':16, # stop frequency of computing, in Hz
    'n_step' :60, # number of frequencies computed between f_start and f_stop
    'n_cycles' : 20, # number of morlet wavelet cycles
    'amplitude_exponent': 2 # raise factor to amplify values 
}

sigma_coupling_params = {
    'sigma_power_params':sigma_power_params,
    'chans':channels_events_select,
    'stage':compute_stage,
    'transition_ratio':0.4, # phase point (in %) corresponding to inspi to expi transition on phase-frequency matrix
    'nb_point_by_cycle':40 # number of phase bins for phase-frequency matrix
}

sigma_coupling_figs_params = {
    'sigma_coupling_params':sigma_coupling_params,
    'sigma_coupling_chan' :compute_chan, # global sigma coupling fig with just one matrix by patient will average phase-freq from these chans
    'fig_global_cycle_type' : 'spindled_N2', # global sigma coupling fig with just one matrix by patient will select this type of tagging of resp cycles 
    'save_article':True # will save outputs in article folder if True
}

sigma_coupling_stats_params = {
    'sigma_coupling_params':sigma_coupling_params,
    'chan_sel':sigma_coupling_figs_params['sigma_coupling_chan']
}
