import pandas as pd
from params import patients

concat = []
for patient in patients:
    rsp_staged = pd.read_excel(f'../resp_features/{patient}_rsp_features_staged.xlsx', index_col = 0)
    concat.append(rsp_staged)

concat_features = pd.concat(concat)

print(concat_features.groupby('sleep_stage').mean()[['inspi_duration','expi_duration','cycle_freq','cycle_ratio']])