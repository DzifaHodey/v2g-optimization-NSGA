from typing import NamedTuple
import numpy as np
import pandas as pd

class Params(NamedTuple):
    population_size: int    # Population size
    generations: int         # Number of Generations
    crossover_rate: float
    mutation_rate: float 
    soc_arr: float           # SOC on arrival
    soc_target: float       # Target SOC
    soh_initial: float   # Initial SOH
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
    return poly_log


def get_number_of_cycles(poly_log, dod):
    """Get number of cycles for a given DoD using the fitted polynomial."""
    dod = 14 if dod < 14 else dod
    return min(np.exp(poly_log(dod)), 10e5)


def get_full_household_energy_profile(filepath, energy_cost_model, month, day, hour):        
    household_energy_profile = pd.read_parquet(filepath, engine='pyarrow')
    data = household_energy_profile[(household_energy_profile["month"] == int(month)) & (household_energy_profile['day'] == int(day)) & (household_energy_profile['hour'] == int(hour))].index
    if not data.empty:
        start_idx = data[0]
        end_idx = start_idx + 24 #for 6hrs
        p_load =  household_energy_profile['total_consumption_kwh'][start_idx:end_idx].values
        p_pv = household_energy_profile['pv_energy_gen_kWh'][start_idx:end_idx].values
        home_size = household_energy_profile['home_size'].iloc[0]
        home_id = household_energy_profile['home_id'].iloc[0]
        pv_size = household_energy_profile['pv_size'].iloc[0]

        if energy_cost_model == 'Fixed':
            energy_buying_prices = [0.1]*24
            energy_selling_prices = [0.08]*24
        else:
            energy_buying_prices = household_energy_profile['energy_buying_price($/kWh)'][start_idx:end_idx].values
            energy_selling_prices = household_energy_profile['energy_selling_price($/kWh)'][start_idx:end_idx].values
        return p_load, p_pv, energy_buying_prices, energy_selling_prices, home_size, home_id, pv_size
    else:
        return "error"
    

def get_soc_on_arrival():
    sessions = pd.read_csv('datasets/soc_arrival_data.csv')[['SoC_start']]
    soc_arr = sessions[(sessions['SoC_start'] >= 0) & (sessions['SoC_start'] <= 100)]['SoC_start'].sample(n=1).iloc[0]
    soc_arr = soc_arr/100
    return soc_arr

def get_soh_initial():
    return np.random.normal(loc=0.95, scale=0.03)


    
