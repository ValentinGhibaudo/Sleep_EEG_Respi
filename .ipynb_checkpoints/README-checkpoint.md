Change patient selection in params.py and run : 
1 - raw_to_da_and_staging (verif signal quality)
2 - rsp_analysis (check detection)
3 - spindle_analysis
4 - spindle_to_resp

Notes générales:
- IA semble voir du wake toujours bcp ++ que human

P1 : done, bonne detection respi, et + spindle en expi
Srate : 256.0
Total duration : 31779 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
YASA trouve 1059 époques de 30 secs, hypnogramme human made en note 1072 soit 13 de plus soit 390 secondes de plus
N3 2039
N1 89
R 5249
N2 8489
W 15839
cycle_duration    2.896755
insp_duration     1.046347
exp_duration      1.850408
cycle_freq        0.348793

P2 : done et + spindle en inspi ? --> régler la detection respi
Srate : 256.0
Total duration : 33999 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
YASA trouve 1133 époques de 30 secs, hypnogramme human made en note 1133 soit 0 de plus soit 0 secondes de plus
N1 269
W 9029
N2 15539
R 6089
N3 2999
cycle_duration    4.073967
insp_duration     1.771820
exp_duration      2.302147
cycle_freq        0.249923

P3 : done, bonne detection respi, 
Srate : 256.0
Total duration : 35930 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
YASA trouve 1197 époques de 30 secs, hypnogramme human made en note 1197 soit 0 de plus soit 0 secondes de plus
W 13709
R 5759
N1 89
N2 14549
N3 1799
cycle_duration    4.092895
insp_duration     1.544467
exp_duration      2.548428
cycle_freq        0.248720

P4 : moyenne detection des cycles, voir pour faire mieux, + spindles en expi
Srate : 256.0
Total duration : 33093 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
YASA trouve 1103 époques de 30 secs, hypnogramme human made en note 1103 soit 0 de plus soit 0 secondes de plus
W 15899
R 3869
N1 239
N2 12119
N3 959
cycle_duration    3.958135
insp_duration     1.526646
exp_duration      2.431489
cycle_freq        0.256229
Frequency    13.215680
Duration      0.827104

P5 : respi pas top du tout, à revoir avec THERM, + de spindles en expi
Srate : 256.0
Total duration : 33343 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
YASA trouve 1111 époques de 30 secs, hypnogramme human made en note 1111 soit 0 de plus soit 0 secondes de plus
W 15299
R 3359
N1 119
N2 14279
N3 269
cycle_duration    4.291502
insp_duration     1.649126
exp_duration      2.642376
cycle_freq        0.237204
Frequency    13.589569
Duration      0.815013

P6 : Checker les signaux neuro, semblent parasités, respi à zoomer mais semble bonne. Résultats moins forts mais toujours + de spindles en expi
Srate : 256.0
Total duration : 30753 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
YASA trouve 1025 époques de 30 secs, hypnogramme human made en note 1025 soit 0 de plus soit 0 secondes de plus
W 13979
R 2669
N1 239
N2 11129
N3 2729
cycle_duration    5.053472
insp_duration     1.419231
exp_duration      3.634241
cycle_freq        0.247790
Frequency    13.211783
Duration      0.875123

P7 : bons signaux, bonne detection rsp, + spindles en expi
Srate : 256.0
Total duration : 36834 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
YASA trouve 1227 époques de 30 secs, hypnogramme human made en note 1227 soit 0 de plus soit 0 secondes de plus
W 18299
R 3569
N1 209
N2 10649
N3 4079
cycle_duration    3.743958
insp_duration     1.389074
exp_duration      2.354883
cycle_freq        0.270063
Frequency    13.152285
Duration      0.96912

P8 : 
Bug epochs

P9 : respi à zoomer mais bonne detection à priori
Srate : 256.0
Total duration : 38798 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
YASA trouve 1293 époques de 30 secs, hypnogramme human made en note 1294 soit 1 de plus soit 30 secondes de plus
W 18329
R 4829
N1 29
N2 14909
N3 689
cycle_duration    5.066785
insp_duration     1.655112
exp_duration      3.411673
cycle_freq        0.203750
No spindles found in data

P10: bonne detection rsp, + spindles expi
Srate : 256.0
Total duration : 32216 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
YASA trouve 1073 époques de 30 secs, hypnogramme human made en note 1073 soit 0 de plus soit 0 secondes de plus
W 12929
R 3929
N1 149
N2 12509
N3 2669
cycle_duration    4.200316
insp_duration     1.213641
exp_duration      2.986676
cycle_freq        0.304107
Frequency    13.399181
Duration      0.907352


En somme, régler la détection mais ce qui semble sûr c'est que poue une certaine raison... les spindles sont phasés sur la respi

