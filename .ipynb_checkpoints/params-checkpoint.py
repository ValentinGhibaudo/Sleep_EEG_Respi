patients = ['P1','P2','P3','P4','P5','P6','P7','P9','P10'] # P8 exclu : Pour le sujet P8 : effectivement, on perd le signal de débit de temps en temps et il est artefacté sur la fin de la nuit. De plus, il manque un morceau de la nuit (le tracé parait tronqué)

patient = 'P1' # Oui, detection respi 5/5
# patient = 'P2' # Oui , detection 2/5
# patient = 'P3' # Oui, detection 4/5
# patient = 'P4' # Oui, detection 4/5
# patient = 'P5' # Oui, detection 2/5
# patient = 'P6' # Oui, detection 4/5
# patient = 'P7' # Oui, detection 5/5
# patient = 'P8' # Oui # to exclude
# patient = 'P9' # Oui, detection 4/5
# patient = 'P10' # Oui, signal mauvais mais outliers bien detectés = 4/5

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