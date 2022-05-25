patients = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10']

# patient = 'P1' # Oui
# patient = 'P2' # Oui
# patient = 'P3' # Oui
# patient = 'P4' # Oui
# patient = 'P5' # Oui mais détection respi très mauvaise
# patient = 'P6' # Oui mais très léger
# patient = 'P7' # Oui
# patient = 'P8' # Oui
# patient = 'P9' # Oui léger
patient = 'P10' # Oui

compute_stages = ['W','R','N2','N3']

stage_to_study = 'N2'

eeg_chans = ['Fp2-C4','C4-T4','T4-O2','Fz-Cz','Cz-Pz','Fp1-C3','C3-T3','T3-O1']
respi_chan = 'DEBIT'
ecg_chan = 'ECG'
eog_chans = ['EOGDt-A2','EOGG-A1']
sel_chans = ['Fp2-C4','C4-T4','T4-O2','Fz-Cz','Cz-Pz','Fp1-C3','C3-T3','T3-O1','DEBIT','THERM','ECG']
stages_labels = ['W','R','N1','N2','N3']
srate = 256

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
    'P10':'DEBIT'
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
    'P10':'+'
}