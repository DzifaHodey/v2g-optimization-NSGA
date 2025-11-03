from typing import NamedTuple
import numpy as np
import pandas as pd
from datetime import datetime


class Params(NamedTuple):
    population_size: int    # Population size
    generations: int         # Number of Generations
    crossover_rate: float
    mutation_rate: float 
    w_e: float            # weight for energy cost
    w_b: float             # weight for battery degradation cost
    penalty_enabled: bool  # If penalty against rapid fluctuations is enabled
    soc_arr: float           # SOC on arrival
    soc_target: float       # Target SOC
    b_max: float        # Max battery capacity (kWh)
    r_cmax: float       # charging rate
    r_dmin: float       # discharging rate
    p_load: list          # household load (kWh)
    p_pv: list          # PV generation (kWh)
    energy_buying_costs: list    # Cost of buying energy from grid ($/kWh)
    energy_selling_costs: list   # Cost of selling energy to grid ($/kWh)
    dod_cycle_curve: np.poly1d


def fit_dod_curve():
# Fit polynomial to DoD vs Cycles data
    data = pd.read_csv("datasets/dod_to_cycles.csv") 
    x = data.iloc[:,0].to_numpy()
    y = data.iloc[:,1].to_numpy()

    # Take log of y
    logy = np.log(y)

    # Polynomial fit in log-space
    degree = 14
    coeffs = np.polyfit(x, logy, degree)
    poly_log = np.poly1d(coeffs)

    # Function in original scale
    def fitted_func(xx):
        return np.exp(poly_log(xx))

    # # Plot
    # xx = np.linspace(min(x), max(x), 500)
    # plt.scatter(x, y, label="Data", color="red")
    # plt.plot(xx, fitted_func(xx), label=f"Poly fit (deg={degree})")
    # plt.yscale("log")  # set log scale
    # plt.xlabel("DoD%")
    # plt.ylabel("Number of Cycles")
    # plt.legend()
    # plt.show()

    # Print formula
    formula_terms = " + ".join([f"{c:.6e}*x^{degree-i}" for i, c in enumerate(coeffs[:-1])])
    formula = f"y = exp({formula_terms} + {coeffs[-1]:.6e})"
    print("Polynomial degree:", degree)
    print("Formula:", formula)
    return poly_log


def get_number_of_cycles(poly_log, dod):
    """Get number of cycles for a given DoD using the fitted polynomial."""
    return np.exp(poly_log(dod))


def get_full_household_energy_profile(energy_cost_model, month, day, hour):        
    household_energy_profile = pd.read_parquet("datasets/household_energy_profile15min.parquet", engine='pyarrow')
    data = household_energy_profile[(household_energy_profile["month"] == int(month)) & (household_energy_profile['day'] == int(day)) & (household_energy_profile['hour'] == int(hour))].index
    if not data.empty:
        start_idx = data[0]
        end_idx = start_idx + 24 #for 6hrs
        p_load =  household_energy_profile['total_consumption_kwh'][start_idx:end_idx].values
        p_pv = household_energy_profile['pv_energy_gen_kWh'][start_idx:end_idx].values

        if energy_cost_model == 'Fixed':
            energy_buying_prices = [0.1]*24
            energy_selling_prices = [0.08]*24
        else:
            energy_buying_prices = household_energy_profile['energy_buying_price($/kWh)'][start_idx:end_idx].values
            energy_selling_prices = household_energy_profile['energy_selling_price($/kWh)'][start_idx:end_idx].values
        return p_load, p_pv, energy_buying_prices, energy_selling_prices
    else:
        return "error"
    

def get_soc_on_arrival():
    data = pd.read_csv("datasets/trips_with_estimated_soc.csv")
    index = np.random.randint(data.shape[0])
    soc_arr = (data['estimated_soc'].values)[index]
    return soc_arr


    
