'The timing of NREM spindles is modulated by the respiratory cycle in human'
V. Ghibaudo, M. Juventin, L. Peter-Derex, N. Buonviso


This repository contains python files and notebooks used for the whole analysis.


Pipeline: 13 Scripts
- params.py # Key parameters are defined before computation
- preproc_staging.py # Convert .edf raw signals preprocessed signals organized in multi-dimensional and labelized datarray (from Xarray) & Sleep staging with YASA on MNE object preprocessed & Compute spectrograms + hypnogram & Compute Sleep statistics (running time ~ 30 mins)
- detect_sleep_events.py # Spindle & Slow-waves detection with YASA toolbox from eeg data (running time ~ 40 mins)
- rsp_detection.py # Detect respiratory cycles and compute features for each cycle (running time ~5 mins)
- rsp_tagging_by_sleep.py # Tag resp cycles with corresponding sleep stage, and notion of spindle or slow wave present inside (running time ~ 3 mins)
- rsp_stats.ipynb # Compute statistics about respiratory cycles features : presence of events inside... sleep staging effect. (running time ~ 1 min)
- events_stats.py # Descriptive statistics on detected events (running time ~ 30 secs)
- events_coupling.py # Compute phase angles of NegPeak of slow waves and Peak of spindles along respiration phase (running time ~ 1 min)
- events_coupling_stats.py # Compute circular stats of phase angles of event occurence according to respiration phase (running time < 10 secs)
- events_coupling_figs.py # Polarplot distributions of events according to respiration phase (running time ~ 1 min)
- morlet_sigma_power.py # Compute TF maps of the whole night recordings of all derivations centered on sigma frequencies (running time ~ 30 mins)
- sigma_coupling.py # Epoching of time-frequency maps and conversion to phase-frequency maps beginning by inspi (running time ~ 3h)
- sigma_coupling_figures.py # Generates 4 types of figs more or less detailed of phase-frequency maps 

Bonus : 
- hilbert_sigma.py # compute sigma filtered eeg signals and sigma envelope for the detailed viewer of events
- run_scripts.py # run list of scripts 

Viewers : 
- global_viewer_events.py # Viewer of eeg signals reref to mastoid or bipolarized with spindles and slowwaved detected markers (one subject run)
- detailed_viewer_events.ipynb # Notebook : Viewer of precise events with time-frequency maps of the eeg signals (one subject run)
- viewer_respi.py # Viewer of raw resp signal with detected inspi and expi points (one subject run)

Tools : 
- deform_tools/py # stretching data tools (time-frequency to phase-frequency sigma maps)