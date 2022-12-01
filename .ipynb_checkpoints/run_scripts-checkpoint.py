import subprocess

program_list = ['events_coupling.py', 'events_coupling_stats.py','events_stats.py']

for program in program_list:
    print(f'Running {program}')
    subprocess.run(['python', program])
    print(f'Finished {program}')