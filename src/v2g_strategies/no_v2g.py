import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import Params, update_battery_state, enforce_constraints
from constants import *


# NO V2G CHARGING STRATEGY
 

def run_no_v2g(params:Params):
    """Only charging strategy implementation.
    Charges the EV at maximum rate until target SOC is reached.

    Args:
        params (Params): Experiment parameters. """
    
    # if PV generation is available, use it first before grid energy
    p_load = params.p_load
    p_pv = params.p_pv
    b_max = params.b_max
    soc_arr = params.soc_arr
    soh_initial = params.soh_initial
    energy_buying_costs=params.energy_buying_costs
    energy_selling_costs=params.energy_selling_costs
    current_rate = params.r_cmax

    total_energy_cost = 0
    total_batt_deg_cost = 0
    net_load = []
    lambda_e_values = []

    soc_target = params.soc_target
    soc, dod = np.zeros(N), np.zeros(N)
    soc[0] = soc_arr
    dod[0] = 1 - soc_arr
    soh = [soh_initial] * N
    delta_soh = 0
    charging_schedule = [params.r_cmax] * N

    for t in range(N):
        if soc[t] >= soc_target:
            current_rate = 0
            charging_schedule[t] = 0
        r_net = current_rate / ALPHA  # Effective charging rate considering efficiency
        L_t = p_pv[t] - p_load[t] - r_net   # Net load at time t (positive means excess generation)
        if L_t < 0:    # if pv is not sufficient
            # Buying energy from grid
            total_energy_cost += abs(L_t) * energy_buying_costs[t]
            lambda_e_values.append(energy_buying_costs[t])
        elif L_t > 0:
            # Selling energy to grid
            total_energy_cost -= L_t * energy_selling_costs[t]
            lambda_e_values.append(-energy_selling_costs[t])

        net_load.append(L_t)

        delta_dod = (-current_rate) / b_max
        # Update SOC, SOH, DOD
        if t < N - 1:
            soc_t1, soh_t1, dod_t1 = update_battery_state(soc[t], soh[t], dod[t], delta_dod, delta_soh)
            soc_t1, soh_t1, dod_t1 = enforce_constraints(soc_t1, soh_t1, dod_t1, SOC_MIN, soc_target, SOH_MIN)
            soc[t+1], soh[t+1], dod[t+1] = soc_t1, soh_t1, dod_t1

    print("----------No V2G Results:--------------")
    print("Charging schedule (kWh):", charging_schedule)
    print(f"Energy cost: ${total_energy_cost:.3f}")
    print(f"Degradation cost: ${total_batt_deg_cost:.3f}")
    print(f"Total cost: ${total_energy_cost + total_batt_deg_cost:.3f}")


    results = {
        'no_v2g_charging_schedule': charging_schedule,
        'no_v2g_energy_cost': total_energy_cost,
        'no_v2g_batt_deg_cost': total_batt_deg_cost,
        'no_v2g_total_cost': total_energy_cost + total_batt_deg_cost,
        'no_v2g_net_load': net_load,
        'no_v2g_socs': soc,
        'no_v2g_sohs': soh,
        'no_v2g_dods': dod,
        'no_v2g_energy_prices': lambda_e_values
    }
    return results
