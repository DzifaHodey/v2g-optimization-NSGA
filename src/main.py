import argparse
from datetime import datetime
import json
import os
import pandas as pd
import numpy as np
from data_loader import fit_dod_curve, get_household_energy_profile, get_soc_on_arrival, get_soh_initial
from v2g_strategies.nsga import run_nsga 
from v2g_strategies.rulebased_v2g import run_rule_based_v2g
from v2g_strategies.no_v2g import run_no_v2g
from plotter import plot_all
from utils import Params, set_global_seed



def parse_args():
    parser = argparse.ArgumentParser(description='Define configuration for genetic algorithm.')
    parser.add_argument('--exp_name', type=str, default='med_home1907_8kwpv_1', help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pop_size', type=int, default=150, help='Population size for NSGA')
    parser.add_argument('--generations', type=int, default=200, help='Generation size for NSGA')
    parser.add_argument('--crossover_rate', type=float, default=0.85, help='Crossover rate for NSGA')
    parser.add_argument('--mutation_rate', type=float, default=0.07, help='Mutation rate for NSGA')
    parser.add_argument('--methods', nargs='+', choices=['no_v2g', 'rulebased', 'nsga'], default=['no_v2g', 'rulebased', 'nsga'], help='Which methods to run in the experiment')
    parser.add_argument('--b_max', type=float, default=76.1, help='Max battery capacity in kWh')
    parser.add_argument('--r_cmax', type=float, default=2.75, help='Charging rate')
    parser.add_argument('--soc_target', type=float, default=0.8, help='Target SOC level')
    parser.add_argument('--pv_size', type=int, choices=[0, 4, 8, 12], default=8, help='PV system size in kW: 0, 4, 8, or 12 kW')
    parser.add_argument('--home_size', type=str, choices=['small', 'med', 'large'], default='med', help='Home size: small, med, or large')
    parser.add_argument('--home_id', type=int, default=1907, help='Home ID from the dataset')
    parser.add_argument('--electricity_cost_model', type=str, choices=['Fixed', 'ToU'], default='ToU', help='Electricity cost model')
    parser.add_argument('--start_datetime', nargs='+', type=str, default=["3,2,19"], help='Time of day to start charging session: month,day,hour')
    parser.add_argument('--output_file_path', type=str, default='results', help='Output folder path to save results')
    parser.add_argument('--plot_results', type=str, choices=['True', 'False'], default='True', help='Whether to plot results or not')
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    set_global_seed(args.seed)
    home_id = args.home_id
    home_size = args.home_size
    pv_size = args.pv_size
    electricity_cost_model = args.electricity_cost_model
    dod_cycle_curve = fit_dod_curve()
    full_results = []

    start_datetimes = args.start_datetime
    for start_datetime in start_datetimes:
        month, day, hour = start_datetime.split(",")
        soc_on_arrival = get_soc_on_arrival()
        soh_initial = get_soh_initial()
        p_load, p_pv, energy_buying_prices, energy_selling_prices = get_household_energy_profile(month, day, hour, {
            "home_size": home_size,
            "home_id": home_id,
            "pv_size": pv_size,
            "electricity_cost_model": electricity_cost_model
        })

        params = Params(
            methods=args.methods,
            population_size=args.pop_size,
            generations=args.generations,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            soc_arr=float(soc_on_arrival),
            soc_target=float(args.soc_target),
            soh_initial=float(soh_initial),
            b_max=float(args.b_max),
            r_cmax=args.r_cmax,
            r_dmin= -1*args.r_cmax,
            dod_cycle_curve=dod_cycle_curve,
            p_load=p_load,
            p_pv=p_pv,
            energy_buying_costs=energy_buying_prices,
            energy_selling_costs=energy_selling_prices
        )

        single_exp_name = f"{args.exp_name}_{month},{day},{hour}"
        meta = {
        "full_exp_name": args.exp_name,
        "exp_name": single_exp_name,
        "start_datetime": start_datetime,
        "home_size": home_size,
        "home_id": home_id,
        "pv_size": pv_size,
        "electricity_cost_model": electricity_cost_model,
        "output_file_path": args.output_file_path,
        "plot_results": args.plot_results
        }

        print(f"Running experiment {args.exp_name} for start time: month={month}, day={day}, hour={hour}, home_id={home_id}, home_size={home_size}, pv_size={pv_size}, electricity_cost_model={electricity_cost_model}")
        results = run_experiment(params, meta)

        parent_dir = os.path.join(results["meta"]["output_file_path"], results["meta"]["full_exp_name"])
        os.makedirs(parent_dir, exist_ok=True)
        save_dir = os.path.join(parent_dir, f"{results['meta']['exp_end_time']}_start{results['meta']['start_datetime']}_batt{params.b_max}_{results['meta']['pv_size']}kwpv_{results['meta']['home_size']}_{results['meta']['home_id']}")
        save_single_result(results, save_dir)
        full_results.append(results)
        if args.plot_results == 'True':
            plot_all(results, params, save_dir)
    save_all_results(full_results, params, parent_dir)


def run_experiment(params: Params, meta) -> dict:
    start = datetime.now()
    results = {
            "meta": meta,
            "methods": {}
        }
    if 'no_v2g' in params.methods:
        results["methods"]["no_v2g"] = run_no_v2g(params)

    if 'rulebased' in params.methods:
        results["methods"]["rulebased"] = run_rule_based_v2g(params)

    if 'nsga' in params.methods:
        results["methods"]["nsga"] = run_nsga(params)
    
    end_time = datetime.now()
    results["meta"]["methods"] = params.methods
    results["meta"]["exp_start_time"] = start.strftime("%Y%m%d_%H%M%S")
    results["meta"]["exp_end_time"] = end_time.strftime("%Y%m%d_%H%M%S")
    results["meta"]["exp_duration_sec"] = (end_time - start).total_seconds()

    return results


def save_single_result(result, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(result["meta"], f, indent=2)

    summary_rows = []
    timeseries = []
    
    for method, results in result['methods'].items():
        scalars, arrays, pareto_front = {}, {}, {}
        for key, value in results.items():
            short_key = key.removeprefix(f"{method}_")

            if short_key.startswith("pareto"):
                pareto_front['exp_name'] = result["meta"]["exp_name"]
                pareto_front[short_key] = value
            elif isinstance(value, (int, float, np.floating, np.integer)):
                scalars['exp_name'] = result["meta"]["exp_name"]
                scalars[short_key] = float(value)
                
            else:
                if short_key != "energy_prices":
                    arrays[key] = np.asarray(value).flatten()

        ts_df = pd.DataFrame(arrays)
        timeseries.append(ts_df)

        if pareto_front:
            np.save(f"{save_dir}/{method}_pareto.npy", pareto_front, allow_pickle=True)
        
        summary_rows.append({"method": method, **scalars})

    # Save combined timeseries
    timeseries_dfs = pd.concat(timeseries, axis=1)
    timeseries_dfs.insert(0, "exp_name", result["meta"]["exp_name"])
    timeseries_dfs.insert(1, "energy prices", results[method+"_energy_prices"])
    timeseries_dfs.to_parquet(f"{save_dir}/timeseries.parquet", index=False, engine='pyarrow')
    pd.DataFrame(summary_rows).set_index("method").to_csv(f"{save_dir}/summary.csv")


def save_all_results(full_results, params, parent_dir):
    file_path = os.path.join(parent_dir, f"all_results_{full_results[0]['meta']['full_exp_name']}.csv")
    all_results = []

    for exp_result in full_results:
        row = {}
        for field, value in params._asdict().items():             
            if isinstance(value, (int, float, np.floating, np.integer)):
                row[field] = value

        row.update(exp_result["meta"])
        for result in exp_result['methods'].values():
            for key, value in result.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    row[key] = float(value)
        all_results.append(row)

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        pd.concat([existing_df, pd.DataFrame(all_results)], ignore_index=True).to_csv(file_path, index=False)
    else:
        pd.DataFrame(all_results).to_csv(file_path, index=False)


    

if __name__ == "__main__":
    main()
