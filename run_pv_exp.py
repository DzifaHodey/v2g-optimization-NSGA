import os
import itertools
import numpy as np

log_file = 'results/pv_exp/experiment_log.txt'
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
    'energy_profile_path' : ['datasets/home-energy-profile/8kw_pv_med_home1907_tou1.parquet', 'datasets/home-energy-profile/4kw_pv_med_home1907_tou1.parquet', 'datasets/home-energy-profile/12kw_pv_med_home1907_tou1.parquet'],
    'start_time' : start_time,
    'output_file_path': ['results/pv_exp']
}

for combo in itertools.product(*param_grid.values()):
    args_str = " ".join(f"--{key} {val}" for key, val in zip(param_grid.keys(), combo))
    log = f"Running experiment with parameters: {args_str}"
    print(log)
    os.system(f"echo {log} >> {log_file}")
    os.system(f"python main.py {args_str} >> {log_file}")
