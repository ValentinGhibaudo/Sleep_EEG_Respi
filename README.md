'The timing of NREM spindles is modulated by the respiratory cycle in human'
V. Ghibaudo, M. Juventin, L. Peter-Derex, N. Buonviso


This repository contains python files used for the whole analysis.


Pipeline: 
- params.py # Key parameters for job computing are imported from this file

# Preproc : 
- preproc_staging.py
    - metadata_job : Compute metadata from .edf raw files
    - preproc_job : Convert .edf raw signals into xarray dataset cropped according to lights on-off data
    - hypnogram_job : Compute sleep staging with YASA on MNE object preprocessed (need preproc_job) 
    - upsample_hypno_job : Upsample hypnogram (need hypnogram_job) to the data srate (256 Hz)
    - spectrogram_job : Compute spectrograms (need preproc_job and upsample_hypno_job)
    - sleep_stats_job : Compute usual sleep statistics (need hypnogram_job)

# Event detection :
- detect_sleep_events.py # Spindle & Slow-waves detection with YASA toolbox from eeg data (running time ~ 30 mins)
    - spindles_detect_job : Spindles detection by yasa (need preproc_job + upsample_hypno_job)
    - slowwaves_detect_job : Slowwaves detection by yasa (need preproc_job + upsample_hypno_job)
    - spindles_tag_job : Tag spindles according to co-occurence with slowwaves and being a slow or fast spindle (need spindles_detect_job + slowwaves_detect_job)
    - slowwaves_tag_job : Tag slowwaves according to co-occurence with spindles (need spindles_detect_job + slowwaves_detect_job)
- events_stats.py # Descriptive statistics and figures on detected events (need spindles_tag_job + slowwaves_tag_job + sleep_stats_job)

# Respi : 
- rsp_detection.py 
    - resp_features_job : Detect respiratory cycles and features on each cycle (need preproc_job)
    - resp_tag_job : Tag respiratory cycle according to the occurence or not of spindles or slowwaves inside, and do staging of resp cycles (need resp_features_job + upsample_hypno_job)

- rsp_stats.py # Compute statistics and figures about respiratory cycles features : presence of events inside... sleep staging effect. 
    - stats_resp_event_job : compute amount of resp cycles with events inside (need resp_tag_job)
    - stats_resp_event_stage_job : compute amount of resp cycles with events inside by stage (need resp_tag_job)

- correlation_resp_events.py # Compute correlations between respiration features and both event types features for each subject (need resp_tag_job + spindles_tag_job + slowwaves_tag_job)
- cross_correlogram.py # Compute cross-correlograms of spindles times vs slow waves times and respi times (need spindles_tag_job + slowwaves_tag_job + resp_tag_job) 

# Events coupling to respiration
- events_coupling.py
    - event_coupling_job :  # Compute phase angles timestamps of spindles or slowwaves along respiration phase (need spindles_tag_job + slowwaves_tag_job + resp_tag_job)

- events_coupling_stats.py # Compute circular stats of phase angles of event occurence according to respiration phase 
- events_coupling_figs.py # Polarplot distributions of events according to respiration phase (need event_coupling_job + resp_tag_job)

# Sigma coupling to respiration
- sigma_coupling.py 
    - sigma_power_job : Compute TF maps of the whole night recordings of all derivations centered on sigma frequencies (need preproc_job)
    - sigma_coupling_job : Epoching of time-frequency maps and conversion to phase-frequency maps beginning by inspi, and average accross cycles (need sigma_power_job + resp_tag_job)
- sigma_coupling_stats.py 
    - stats_sigma_coupling_job : Compute circular statistics of modulation of sigma power along respiration phase (need sigma_coupling_job)
- sigma_coupling_figures.py # Generates 3 types of figs more or less detailed of phase-frequency maps (need sigma_coupling_job)



# Note : Important outputs like figures or tables are computed thanks to entangled jobs that are automatically run if not yet computed. It means that running these "outputs" scripts is sufficient to call all required jobs. The list of these "outputs" scripts is : 
- events_stats.py
- rsp_stats.py
- correlation_resp_events.py
- cross_correlogram.py
- events_coupling_stats.py
- events_coupling_figs.py
- sigma_coupling_stats.py 
- sigma_coupling_figures.py


# Tools : 
- deform_tools.py # stretching data tools (time-frequency to phase-frequency sigma maps)