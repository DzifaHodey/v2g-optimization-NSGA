from typing import NamedTuple
import numpy as np
import random
from constants import *


class Params(NamedTuple):
    methods: list[str]
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



def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def get_number_of_cycles(poly_log, dod):
    """Get number of cycles for a given DoD using the fitted polynomial."""
    dod = 14 if dod < 14 else dod
    return min(np.exp(poly_log(dod)), 10e5)

def update_battery_state(SOC_t, SOH_t, DOD_t, delta_dod, delta_soh):
    SOC_t1 = SOC_t - delta_dod
    SOH_t1 = SOH_t - delta_soh
    DOD_t1 = DOD_t + delta_dod
    return SOC_t1, SOH_t1, DOD_t1

def enforce_constraints(SOC_t1, SOH_t1, DOD_t1, soc_min, soc_max, soh_min):
    if SOC_t1 > soc_max:
        SOC_t1 = soc_max
           # if soc is at max, don't charge?
    elif SOC_t1 < soc_min:
        SOC_t1 = soc_min
       # charge if soc is below min ?

    if SOH_t1 < soh_min:
        SOH_t1 = soh_min

    if DOD_t1 < 0:
        DOD_t1 = 0
    elif DOD_t1 > 1 - soc_min:
        DOD_t1 = 1 - soc_min

    return SOC_t1, SOH_t1, DOD_t1


def calculate_batt_deg_costs(current_charging_rate, b_max, dod, t, dod_cycle_curve):
    # change in DOD
    delta_dod = (-current_charging_rate) / b_max

    # Decrease in SOH
    if current_charging_rate <= 0:  # Discharging
        delta_soh = (1 - SOH_MIN)/get_number_of_cycles(dod_cycle_curve, ((dod[t-1] + delta_dod) if t > 0 else dod[t])*100)  # Convert to percentage
    else:  # Charging
        delta_soh = 0
    # Battery degradation cost
    C_b = LAMBDA_B * delta_soh
    return C_b, delta_soh, delta_dod

