'The timing of NREM spindles is modulated by the respiratory cycle in human'
V. Ghibaudo, M. Juventin, L. Peter-Derex, N. Buonviso

This repository contains python files and notebooks used for the whole analysis.

Tools : 
- cycle_detection.py # detection of respiratory cycles
- deform_tools/py # stretching data tools (time-frequency to phase-frequency sigma maps)
- respiration_features.py # compute respiration features from respiration cycle detection outputs

Pipeline:
1 - params.py # Key parameters are defined before computation
2 - preproc_staging.py # Convert .edf raw signals preprocessed signals organized in multi-dimensional and labelized datarray (from Xarray) & Sleep staging with YASA on MNE object preprocessed
3 - sleep_statistics.py # Compute usual sleep statistics from hypnograms
4 - rsp_detection.py # Detect respiratory cycles and compute features for each cycle
5 - rsp_staging.py # Tag resp cycles with corresponding sleep stage
6 - detect_sleep_events.py # Spindle & Slow-waves detection with YASA toolbox from eeg data (running time ~ 40 mins)
7 - events_coupling.py # Compute phase angles of NegPeak of slow waves and Peak of spindles along respiration phase (running time ~ 30 mins, depending on amount of detected events)
8 - events_coupling_stats.py # Polarplot distributions of events according to respiration phase & Circular stats (running time ~ 1 min)
9 - events_stats.py # Descriptive statistics on detected events (running time ~ 30 secs)
10 - extract_sigma_power.py # Compute TF maps of the whole night recordings of all derivations centered on sigma frequencies (running time ~ 30 mins)
11 - sigma_coupling.py # Epoching of time-frequency maps and conversion to phase-frequency maps beginning by inspi (running time ~ 3h)