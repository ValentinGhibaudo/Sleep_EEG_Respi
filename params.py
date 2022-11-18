# patients = ['P1','P2','P3','P4','P5','P6','P7','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20'] #  P8 exclude because of truncated signal and bad respi signal
# patients = ['P11','P12','P13','P14','P15','P16','P18','P19','P20']
# patients = ['P14','P18']
# patients = ['P11','P12','P13','P15','P16','P17','P18','P19','P20']
patients = ['P1','P2','P3','P4','P5','P6','P7','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20'] 
# patients = ['P14']

# patient = 'P1' 
# patient = 'P2' 
# patient = 'P3'
# patient = 'P4' 
# patient = 'P5' 
# patient = 'P6' 
# patient = 'P7' 
# patient = 'P8' 
# patient = 'P9' 
# patient = 'P10' 
# patient = 'P11' 
# patient = 'P12' 
# patient = 'P13' 
# patient = 'P14' 
# patient = 'P15' 
# patient = 'P16' 
patient = 'P17' # mismatch staging human vs ia : human staging stoppe à 39120 secs alors que signal dure 39365 secs
# patient = 'P18' 
# patient = 'P19' 
# patient = 'P20' 

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
    'P10':'DEBIT',
    'P11':'DEBIT',
    'P12':'DEBIT',
    'P13':'DEBIT',
    'P14':'DEBIT',
    'P15':'DEBIT',
    'P16':'DEBIT',
    'P17':'DEBIT',
    'P18':'DEBIT',
    'P19':'DEBIT',
    'P20':'DEBIT'
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
    'P10':'+',
    'P11':'+',
    'P12':'+',
    'P13':'+',
    'P14':'+',
    'P15':'+',
    'P16':'+',
    'P17':'+',
    'P18':'+',
    'P20':'+'
}