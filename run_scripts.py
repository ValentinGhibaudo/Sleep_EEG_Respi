import subprocess
import time

# set list of script to run in the desired order
# program_list = ['rsp_detection.py','rsp_tagging_by_sleep.py','rsp_stats.py','events_stats.py','events_coupling.py','events_coupling_stats.py','events_coupling_figs.py']
# program_list = ['events_coupling.py','events_coupling_stats.py','events_coupling_figs.py']
program_list = ['morlet_sigma_power.py','sigma_coupling.py']


# RUN
for program in program_list:
    print(f'Running {program}')
    t1 = time.perf_counter()
    subprocess.run(['python', program])
    t2 = time.perf_counter()
    run_time = t2-t1
    run_time_mins = run_time / 60
    print(f'Finished {program} in {round(run_time_mins,2)} mins')