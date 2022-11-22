from params import patients
import yasa
import pandas as pd
import numpy as np

save = True

for patient in patients:
    print(patient)
    hypno = pd.read_excel(f'/crnldata/cmo/Etudiants/Valentin_G/Sleep_EEG_Respi/hypnos/hypno_{patient}.xlsx', index_col = 0)['yasa hypnogram'].values # load hypnogram
    sf_hyp = 1/30 # sampling frequency of hypnogram = 1 value per 30 seconds
    hypno_int = yasa.hypno_str_to_int(hypno) # transform letter to int code (0 = Wake, 1 = N1 sleep, 2 = N2 sleep, 3 = N3 sleep, 4 = REM sleep)
    sleep_stats = yasa.sleep_statistics(hypno_int, sf_hyp) # compute usual stats
    stats = pd.DataFrame.from_dict(sleep_stats, orient = 'index').T # put in in dataframe
    if save:
        stats.to_excel(f'../participant_characteristics/{patient}_sleep_stats.xlsx') # save
    