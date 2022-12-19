import xarray as xr
import pandas as pd
import ghibtools as gh
from params import subjects, sigma_coupling_chan

da_all = None

N_rows = []

for subject in subjects:
    phase_freqs = xr.open_dataarray(f'../sigma_coupling/{subject}_phase_freq_sigma.nc').dropna('cycle').sel(chan = sigma_coupling_chan)  # 3 D xarray :  cycle * point * freq
    resp_features_stretched = pd.read_excel(f'../resp_features/{subject}_resp_features_tagged_stretch.xlsx', index_col = 0) 
    
    c_spindled = resp_features_stretched[resp_features_stretched['Spindle_Tag'] == 1].index.to_numpy()
    c_unspindled = resp_features_stretched[resp_features_stretched['Spindle_Tag'] == 0].index.to_numpy()
    c_N2 = resp_features_stretched[resp_features_stretched['sleep_stage'] == 'N2'].index.to_numpy()
    c_N3 = resp_features_stretched[resp_features_stretched['sleep_stage'] == 'N3'].index.to_numpy()
    c_all = resp_features_stretched.dropna().index.to_numpy()
    
    concat = []
    c_labels = ['all','spindled','unspindled','N2','N3','diff']
    for c_label, c_type in zip(c_labels,[c_all, c_spindled, c_unspindled, c_N2, c_N3,'diff']):
        if c_label == 'diff':
            N_rows.append([subject,c_label,'NA'])
            phase_freq_mean = phase_freqs.sel(cycle = c_spindled).mean('cycle') - phase_freqs.sel(cycle = c_unspindled).mean('cycle')
        else:
            N_rows.append([subject,c_label,c_type.size])
            phase_freq_mean = phase_freqs.sel(cycle = c_type).mean('cycle')
            
        concat.append(phase_freq_mean)
        
        if da_all is None: 
            da_all = gh.init_da({'subject':subjects, 'cycle_type':c_labels, 'freq':phase_freqs.coords['freq'], 'point':phase_freqs.coords['point']})
        
        data = phase_freq_mean.data.T
        da_all.loc[subject, c_label, : ,:] = data
        
Ns = pd.DataFrame(N_rows , columns = ['subject','cycle_type', 'N'])

Ns.to_excel('../sigma_coupling/n_cycles_by_type.xlsx')
da_all.to_netcdf('../sigma_coupling/mean_phase_freq_sigma.nc')