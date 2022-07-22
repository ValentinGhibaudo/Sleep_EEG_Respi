patients = ['P1','P2','P3','P4','P5','P6','P7','P9','P10'] #  P8 exclude because of truncated signal and bad respi signal

patient = 'P1' 
# patient = 'P2' 
# patient = 'P3'
# patient = 'P4' 
# patient = 'P5' 
# patient = 'P6' 
# patient = 'P7' 
# patient = 'P8' 
# patient = 'P9' 
# patient = 'P10' 

labelling_method = 'ia' # set ia or human hypnogram chosen to label signals epochs

eeg_chans = ['Fp2-C4','C4-T4','T4-O2','Fz-Cz','Cz-Pz','Fp1-C3','C3-T3','T3-O1']
respi_chan = 'DEBIT'
ecg_chan = 'ECG'
eog_chans = ['EOGDt-A2','EOGG-A1']
sel_chans = ['Fp2-C4','C4-T4','T4-O2','Fz-Cz','Cz-Pz','Fp1-C3','C3-T3','T3-O1','DEBIT','THERM','ECG']
stages_labels = ['W','R','N1','N2','N3']
srate = 256
HP = 0.17
LP = 100

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