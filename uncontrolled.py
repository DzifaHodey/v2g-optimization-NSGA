import numpy as np
from exp_params import Params
from genetic_algorithm import update_battery_state, enforce_constraints


# UNCONTROLLED CHARGING STRATEGY

# CONSTANTS
# EV is 2025 Kia EV9 specs

T = 6        # Time horizon (hours)
N = T*4          #total number of data points
# B_MAX = 76.1    #76.1  # Battery capacity (kWh)
B_MIN = 0         # Minimum battery capacity (kWh)
DELTA_T = 1         # 15mins Time step  (hours)
SOC_MIN = 0.3        # Minimum state of charge (percentage of total capacity)
SOH_MIN = 0.7       # Minimum state of health (percentage of nominal capacity)
ALPHA = 0.99        # Battery charging efficiency (percentage)
BETA = 0.85         # Battery discharging efficiency (percentage)
LAMBDA_B = 200.0     # Conversion coeefficient of battery degradation to monetary cost ($/kWh) 

def uncontrolled_charging(params:Params):
    """Uncontrolled (immediate) charging strategy implementation.
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

    # global SOCs, SOHs, DODs
    soc, dod = np.zeros(N), np.zeros(N)
    soc[0] = soc_arr
    dod[0] = 1 - soc_arr
    soh = [soh_initial] * N
    delta_soh = 0
    charging_schedule = [params.r_cmax] * N

    for t in range(N):
        r_net = current_rate * ALPHA  # Effective charging rate considering efficiency
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
            soc_t1, soh_t1, dod_t1 = enforce_constraints(soc_t1, soh_t1, dod_t1)
            soc[t+1], soh[t+1], dod[t+1] = soc_t1, soh_t1, dod_t1

    return charging_schedule, total_energy_cost, total_batt_deg_cost, net_load, soc, soh, dod
