import subprocess

program_list = ['rsp_detection.py','rsp_tagging_by_sleep.py']

for program in program_list:
    print(f'Running {program}')
    subprocess.run(['python', program])
    print(f'Finished {program}')