import numpy as np
import pandas as pd
import yasa
import xarray as xr
from params import patients, srate

save = True

for patient in patients: # loop on run keys
	print(patient)
	data = xr.open_dataarray(f'../preproc/{patient}.nc') # open lazy data for hypno upsampling
	hypno = pd.read_excel(f'../hypnos/hypno_{patient}.xlsx', index_col = 0) # load hypno of the run key
	hypno_upsampled = yasa.hypno_upsample_to_data(hypno = hypno['yasa hypnogram'].values, sf_hypno=1/30, data=data, sf_data=srate) # upsample hypno
	rsp_features = pd.read_excel(f'../resp_features/{patient}_rsp_features.xlsx', index_col = 0) # load rsp features
	idx_start = rsp_features['start'].values # select start indexes of the cycles
	stage_of_the_start_idxs = hypno_upsampled[idx_start] # keep stages corresponding to start resp cycle indexes
	rsp_features_staged = rsp_features.copy()
	rsp_features_staged['sleep_stage'] = stage_of_the_start_idxs # append sleep stage column to resp features
	if save:
		rsp_features_staged.to_excel(f'../resp_features/{patient}_rsp_features_staged.xlsx') # save



