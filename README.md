Change patient selection in params.py and run : 
1 - raw_to_da_and_staging (verif signal quality), select epochs that get same label by human and by ia, and without microarousal
2 - rsp_analysis (check detection)
3 - spindle_analysis
4 - spindle_to_resp

Notes générales:
- IA semble voir du wake toujours bcp ++ que human

P1 : done, bonne detection respi, et + spindle en expi
P1
Srate : 256.0
Total duration : 31779 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
Patient stages : ['W', 'N2', 'N3', 'R']
Duration by stage :
W 6809
N2 6809
N3 1799
R 4529
Mismatch ia vs human by stage :
N2    0.530488
N3    0.234756
R     0.134146
N1    0.079268
W     0.021341


P2 : done et + spindle en inspi ? --> régler la detection respi # bug detection spindles
P2
Srate : 256.0
Total duration : 33999 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
Patient stages : ['W', 'N2', 'N3', 'R', 'N1']
Duration by stage :
W 4379
N2 9779
N3 2369
R 5099
N1 119
Mismatch ia vs human by stage :
N3    0.351171
N2    0.244147
N1    0.187291
R     0.167224
W     0.050167


P3 : done, bonne detection respi, # bug detection spindles
P3
Srate : 256.0
Total duration : 35930 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
Patient stages : ['W', 'N2', 'N3', 'R']
Duration by stage :
W 2339
N2 9029
N3 1319
R 4919
Mismatch ia vs human by stage :
N2    0.443340
N3    0.328032
R     0.178926
N1    0.043738
W     0.005964

P4 : moyenne detection des cycles, voir pour faire mieux, + spindles en expi
P4
Srate : 256.0
Total duration : 33093 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
Patient stages : ['W', 'N2', 'N3', 'R', 'N1']
Duration by stage :
W 5219
N2 8189
N3 599
R 3719
N1 29
Mismatch ia vs human by stage :
N2    0.458613
N3    0.270694
R     0.140940
N1    0.120805
W     0.008949

P5 : bonne detection respi, bug detection spindles
P5
Srate : 256.0
Total duration : 33343 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
Patient stages : ['W', 'N3', 'N2', 'R']
Duration by stage :
W 5489
N3 209
N2 7589
R 2279
Mismatch ia vs human by stage :
N3    0.395556
N1    0.246667
N2    0.226667
R     0.122222
W     0.008889

P6 : Checker les signaux neuro, semblent parasités, respi detection bonne. bug detection spindles
P6
Srate : 256.0
Total duration : 30753 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
Patient stages : ['W', 'N2', 'N3', 'R', 'N1']
Duration by stage :
W 4349
N2 5579
N3 2159
R 2399
N1 59
Mismatch ia vs human by stage :
N3    0.414487
N2    0.344064
R     0.164990
N1    0.062374
W     0.014085

P7 : bons signaux, bonne detection rsp, + spindles en expi
P7
Srate : 256.0
Total duration : 36834 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
Patient stages : ['W', 'N2', 'N3', 'R']
Duration by stage :
W 6959
N2 9629
N3 3929
R 3359
Mismatch ia vs human by stage :
N2    0.584416
R     0.162338
N3    0.133117
N1    0.110390
W     0.009740

P8 : 
P8
Srate : 256.0
Total duration : 14399 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
Patient stages : ['W', 'N2', 'N3', 'R']
Duration by stage :
W 3749
N2 3659
N3 1559
R 449
Mismatch ia vs human by stage :
N3    0.401408
N2    0.366197
N1    0.140845
R     0.049296
W     0.042254

P9 : respi à zoomer mais bonne detection à priori
P9
Srate : 256.0
Total duration : 38798 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
Patient stages : ['W', 'N2', 'N3', 'R']
Duration by stage :
W 6629
N2 7859
N3 329
R 3359
Mismatch ia vs human by stage :
N2    0.406654
N3    0.319778
R     0.173752
N1    0.073937
W     0.025878

P10: bonne detection rsp
P10
Srate : 256.0
Total duration : 32216 seconds
Nb of eeg electrodes : 13
Nb physios electrodes : 20
Patient stages : ['W', 'N2', 'N3', 'R']
Duration by stage :
W 6869
N2 9149
N3 2369
R 3689
Mismatch ia vs human by stage :
N2    0.548507
N3    0.250000
N1    0.085821
R     0.082090
W     0.033582


En somme, régler la détection mais ce qui semble sûr c'est que pour une certaine raison... les spindles sont phasés sur la respi

