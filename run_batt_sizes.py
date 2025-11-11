import os
import itertools
import numpy as np

log_file = 'results/batt_sizes/experiment_log.txt'
no_of_months = 12

    
start_time = []
hr = [0, 6, 9, 12, 15, 18, 22]
for h in hr:
    np.random.seed(42)
    day = np.random.randint(1,28)
    if no_of_months == 12:
        start_time = [(f"{i+1},{day},{h}") for i in range(no_of_months)]
    else:
        start_time.append(f"{5},{day},{h}")


param_grid = {
    'energy_profile_path' : ['datasets/home-energy-profile/8kw_pv_med_home1907_tou1.parquet'],
    'start_time' : start_time,
    'b_max': [60, 70, 80, 90, 100],
    'output_file_path': ['results/batt_sizes']
}

for combo in itertools.product(*param_grid.values()):
    args_str = " ".join(f"--{key} {val}" for key, val in zip(param_grid.keys(), combo))
    log = f"Running experiment with parameters: {args_str}"
    print(log)
    os.system(f"echo {log} >> {log_file}")
    os.system(f"python main.py {args_str} >> {log_file}")
