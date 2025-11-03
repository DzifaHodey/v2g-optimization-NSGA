import argparse
# import matplotlib.pyplot as plt
# import pandas as pd
from genetic_algorithm import Params, genetic_algorithm
from exp_params import fit_dod_curve, get_full_household_energy_profile, get_soc_on_arrival


parser = argparse.ArgumentParser(description='Define configuration for genetic algorithm.')
parser.add_argument('--pop_size', default=50, help='Population size')
parser.add_argument('--generations', default=100, help='Generation size')
parser.add_argument('--crossover_rate', default=0.8, help='Crossover rate for GA')
parser.add_argument('--mutation_rate', default=0.1, help='Mutation rate for GA')
parser.add_argument('--penalty_enabled', default=True, help='Penalty to limit rapid fluctuations')
parser.add_argument('--w_e', default=0.7, help='Multi-objective weight for net energy cost')
parser.add_argument('--w_b', default=0.3, help='Multi-objective weight for battery degradation cost')
parser.add_argument('--b_max', default=76.1, help='Max battery capacity in kWh')
parser.add_argument('--r_cmax', default=2.75, help='Charging rate')
parser.add_argument('--soc_target', default=1.0, help='Target SOC level')
parser.add_argument('--energy_cost_model', choices=['Fixed', 'ToU'], default='ToU', help='Energy cost model')
parser.add_argument('--start_time', help='Time of day to start charging session: month,day,hour')


args = parser.parse_args()

#setup
month, day, hour = args.start_time.split(",")
p_load, p_pv, energy_buying_prices, energy_selling_prices = get_full_household_energy_profile(args.energy_cost_model, month, day, hour)
soc_on_arrival = get_soc_on_arrival()
dod_cycle_curve = fit_dod_curve()


params = Params(
        population_size=args.pop_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        w_e=args.w_e,
        w_b=args.w_b,
        penalty_enabled=args.penalty_enabled,
        soc_arr=soc_on_arrival,
        soc_target=args.soc_target,
        b_max=args.b_max,
        r_cmax=args.r_cmax,
        r_dmin= -1*args.r_cmax,
        dod_cycle_curve=dod_cycle_curve,
        p_load=p_load,
        p_pv=p_pv,
        energy_buying_costs=energy_buying_prices,
        energy_selling_costs=energy_selling_prices

    )

best_solution, best_fitness, best_energy_cost, best_batt_deg_cost, final_S_norm, final_target_deviation_penalty, final_socs, final_sohs, final_dods, net_load, lambda_e_s, best_performers = genetic_algorithm(params)

print(f"Best Solution: {best_solution}\n fitness: {best_fitness}\n energy_cost: {best_energy_cost}\n batt_deg_cost: {best_batt_deg_cost}\n S_norm: {final_S_norm}\n target_deviation_penalty: {final_target_deviation_penalty}\n\n final_socs: {final_socs}\n final_sohs: {final_sohs}\n final_dods: {final_dods}\n net_load: {net_load}\n lambda_e_s: {lambda_e_s}")

