'The timing of NREM spindles is modulated by the respiratory cycle in human'
V. Ghibaudo, L. Peter-Derex, N. Buonviso

This repository contains python files and notebooks used for the whole analysis.

Pipeline:
1 - params.py # Key parameters are defined before computation
2 - raw_to_da_and_staging.ipynb # Convert .edf raw signals to staged preprocessed signals organized in multi-dimensional and labelized datarray (from Xarray)
3 - rsp_analysis.ipynb (check detection) # Cycle detection of breaths : from respiratory signal to respiration features saved in dataframes 
4 - sp_sw_detection.ipynb # Spindle and Slow-wave detection with YASA toolbox from eeg / eog / emg (chin) data
5 - spindle_to_resp.ipynb # Compute polar plots of spindles and slow waves using respiration features
6 - wavelet_analysis.ipynb # Compute time-frequency on NREM sleep preprocessed signals
7 - stretch_tf_inspi.ipynb # Epoching of time-frequency maps and conversion to phase-frequency trials beginning by inspi

Tools : 
- cycle_detection.py # detection of respiratory cycles
- deform_tools/py # stretching data tools (time-frequency to phase-frequency trials)
- respiration_features.py # compute respiration features from respiration cycle detection ouputs



Pipeline:
1 - params.py # Key parameters are defined before computation
2 - preproc.py # Convert .edf raw signals preprocessed signals organized in multi-dimensional and labelized datarray (from Xarray)
3 - detect_spindles.py # Spindle detection with YASA toolbox from eeg data
