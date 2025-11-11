import argparse
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from exp_params import Params, fit_dod_curve, get_full_household_energy_profile, get_soc_on_arrival, get_soh_initial
from nsga import run_nsga, plot_pareto_front 
from rulebased_v2g import rule_based_v2g
from uncontrolled import uncontrolled_charging

start_time = datetime.now()

parser = argparse.ArgumentParser(description='Define configuration for genetic algorithm.')
parser.add_argument('--pop_size', type=int, default=150, help='Population size')
parser.add_argument('--generations', type=int, default=200, help='Generation size')
parser.add_argument('--crossover_rate', type=float, default=0.85, help='Crossover rate for GA')
parser.add_argument('--mutation_rate', type=float, default=0.07, help='Mutation rate for GA')
parser.add_argument('--b_max', type=float, default=76.1, help='Max battery capacity in kWh')
parser.add_argument('--r_cmax', type=float, default=2.75, help='Charging rate')
parser.add_argument('--soc_target', type=float, default=0.8, help='Target SOC level')
parser.add_argument('--energy_cost_model', type=str, choices=['Fixed', 'ToU'], default='ToU', help='Energy cost model')
parser.add_argument('--start_time', type=str, default="5,7,6", help='Time of day to start charging session: month,day,hour')
parser.add_argument('--energy_profile_path', type=str,  default='datasets/home-energy-profile/8kw_pv_small_home19_tou1.parquet', help='PV and home energy profile file path')
parser.add_argument('--output_file_path', type=str, default='results', help='Output file to save results')

args = parser.parse_args()

#setup
month, day, hour = args.start_time.split(",")
p_load, p_pv, energy_buying_prices, energy_selling_prices, home_size, home_id, pv_size = get_full_household_energy_profile(args.energy_profile_path, args.energy_cost_model, month, day, hour)
soc_on_arrival = get_soc_on_arrival()
soh_initial = get_soh_initial()
dod_cycle_curve = fit_dod_curve()


params = Params(
        population_size=args.pop_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,

        soc_arr=float(soc_on_arrival),
        soc_target=args.soc_target,
        soh_initial=soh_initial,
        b_max=float(args.b_max),
        r_cmax=args.r_cmax,
        r_dmin= -1*args.r_cmax,
        dod_cycle_curve=dod_cycle_curve,
        p_load=p_load,
        p_pv=p_pv,
        energy_buying_costs=energy_buying_prices,
        energy_selling_costs=energy_selling_prices

    )

print("Starting SOC: ", soc_on_arrival)

# uncontrolled
uc_charging_schedule, uc_total_energy_cost, uc_total_batt_deg_cost, uc_net_load, uc_soc, uc_soh, uc_dod = uncontrolled_charging(params)
uc_total_cost = uc_total_energy_cost + uc_total_batt_deg_cost

print(f"Uncontrolled Charging Strategy - Total Energy Cost: ${uc_total_energy_cost:.2f}, Total Battery Degradation Cost: ${uc_total_batt_deg_cost:.2f}, Total Cost: ${uc_total_cost:.2f}")
print(f"Final SOC: {uc_soc}, Final SOH: {uc_soh}")


rb_charging_schedule, rb_total_cost, rb_total_energy_cost, rb_total_batt_deg_cost, rb_net_load, rb_soc, rb_soh, rb_dod = rule_based_v2g(params)
print("\nRule-based charging schedule: ", rb_charging_schedule)
print(f"Rule-Based V2G Strategy - Total Energy Cost: ${rb_total_energy_cost:.2f}, Total Battery Degradation Cost: ${rb_total_batt_deg_cost:.2f}, Total Cost: ${rb_total_cost:.2f}")
print(f"Final SOC: {rb_soc}, Final SOH: {rb_soh}")



# NSGA-II
pareto_solutions, pareto_objectives, nsga_best_solution, best_obj, best_net_load, nsga_socs, nsga_sohs, nsga_dods, lambda_e_s = run_nsga(params)
nsga_best_fitness = sum(best_obj)
nsga_best_energy_cost = best_obj[0]
nsga_best_batt_deg_cost = best_obj[1]

end_time = datetime.now()


timestamp = end_time.strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(args.output_file_path, f"{timestamp}_start{args.start_time}_batt{args.b_max}_{pv_size}kwpv_{home_size}_{home_id}")
os.makedirs(save_dir, exist_ok=True)

duration_of_run = end_time - start_time

plot_pareto_front(pareto_objectives, save_dir)


plt.figure()
plt.plot(p_load, label='Total Energy Consumption (kWh)')
plt.plot(p_pv, label='PV Energy Generation (kWh)')
plt.xlabel('Time (15min interval)')
plt.ylabel('Consumption (kWh)')
plt.tight_layout()
plt.legend()
plt.title('Load and PV Generation Profile')
plt.savefig(os.path.join(save_dir, "load_pv_profile.png"))
plt.close()

plt.figure()
plt.plot(params.energy_buying_costs, label='Energy buying costs ($/kWh)')
plt.plot(params.energy_selling_costs, label='Energy selling costs ($/kWh)')
plt.xlabel('Time (15min interval)')
plt.ylabel('Consumption (kWh)')
plt.title('Energy Cost Profile')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "energy_cost_profile.png"))
plt.close()

plt.figure()
plt.plot(nsga_best_solution, label='NSGA-II Optimal Schedule')
plt.plot(uc_charging_schedule, label='Uncontrolled Schedule')
plt.plot(rb_charging_schedule, label='Rule-based Schedule')
plt.title('Charging/Discharging Schedule over Time')
plt.ylabel('Charging Rate (kWh)')
plt.xlabel('Time step')
plt.ylim(-3, 3)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "charging_schedule.png"))
plt.close()

plt.figure()
plt.plot(nsga_socs, label='NSGA-II SOC')
plt.plot(uc_soc, label='Uncontrolled Charging SOC')
plt.plot(rb_soc, label='Rule-based SOC')
plt.title('Battery SOC over time')
plt.xlabel("Time step")
plt.ylabel("Level")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, "soc_curve.png"))
plt.close()

plt.figure()
plt.plot(nsga_sohs, label='NSGA-II SOH')
plt.plot(uc_soh, label='Uncontrolled Charging SOH')
plt.plot(rb_soh, label='Rule-based SOH')
plt.title('Battery SOH over time')
plt.xlabel("Time step")
plt.ylabel("Level")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, "soh_curve.png"))
plt.close()

# # === Save summary values ==

exp_parameters = {
    "population_size": params.population_size,
    "generations": params.generations,
    "crossover_rate": params.crossover_rate,
    "mutation_rate": params.mutation_rate,
    "start_time(month,day,hr)": args.start_time,  
    "soc_arrival": params.soc_arr,
    "soc_target": params.soc_target,
    "soh_initial": params.soh_initial,
    "b_max": params.b_max,
    "r_cmax": params.r_cmax,
    "r_dmin": params.r_dmin,
    "energy_cost_model": args.energy_cost_model,
    "p_load":params.p_load,
    "p_pv":params.p_pv,
    "energy_buying_costs":params.energy_buying_costs,
    "energy_selling_costs":params.energy_selling_costs,
    "home_size": home_size,
    "home_id": int(home_id),
    "pv_size": pv_size,
    "nsga_net_load": best_net_load,
    "nsga_best_solution": nsga_best_solution,
    "nsga_best_total_cost": float(nsga_best_fitness),
    "nsga_best_energy_cost": float(nsga_best_energy_cost),
    "nsga_best_batt_deg_cost": float(nsga_best_batt_deg_cost),
    "nsga_soc": nsga_socs,
    "nsga_soh": nsga_sohs,
    "nsga_dod": nsga_dods,
    "uncontrolled_charging_schedule": uc_charging_schedule,
    "uncontrolled_total_cost": float(uc_total_cost),
    "uncontrolled_total_energy_cost": float(uc_total_energy_cost),
    "uncontrolled_total_batt_deg_cost": float(uc_total_batt_deg_cost),
    "uncontrolled_soc": uc_soc,
    "uncontrolled_soh": uc_soh,
    "uncontrolled_dod": uc_dod,
    "rulebased_charging_schedule": rb_charging_schedule,
    "rulebased_total_cost": float(rb_total_cost),
    "rulebased_total_energy_cost": float(rb_total_energy_cost),
    "rulebased_total_batt_deg_cost": float(rb_total_batt_deg_cost),
    "rulebased_soc": rb_soc,
    "rulebased_soh": rb_soh,
    "rulebased_dod": rb_dod,
    "duration_of_experiment": str(duration_of_run)
}

with open(os.path.join(save_dir, "params_and_outputs.json"), "w") as f:
    json.dump(exp_parameters, f, indent=4, default=str)


output_summary_file = os.path.join(args.output_file_path, 'results_summary.csv')
# Ensure parent directory exists (if any) and append or create the CSV
if not os.path.exists(output_summary_file):
    pd.DataFrame([exp_parameters]).to_csv(output_summary_file, index=False)
else:
    pd.DataFrame([exp_parameters]).to_csv(output_summary_file, mode='a', header=False, index=False)
