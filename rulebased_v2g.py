import numpy as np
from exp_params import Params
from genetic_algorithm import update_battery_state, enforce_constraints, calculate_batt_deg_costs


# RULE-BASED CHARGING STRATEGY

T = 6    # Time horizon (hours)
N = T*4     #total number of data points
B_MIN = 0       # Minimum battery capacity (kWh)
DELTA_T = 1         # 15mins Time step  (hours)
SOC_MIN = 0.3       # Minimum state of charge (percentage of total capacity)
SOH_MIN = 0.7       # Minimum state of health (percentage of nominal capacity)
ALPHA = 0.99       # Battery charging efficiency (percentage)
BETA = 0.85         # Battery discharging efficiency (percentage)
LAMBDA_B = 200.0        # Conversion coeefficient of battery degradation to monetary cost ($/kWh)

def rule_based_v2g(params:Params):
    """rule-based charging strategy implementation.
    Charges the EV at maximum rate until target SOC is reached.

    Args:
        params (Params): Experiment parameters. """
    
    # if PV generation is available, use it first before grid energy
    p_load = params.p_load
    p_pv = params.p_pv
    b_max = params.b_max
    soc_arr = params.soc_arr
    soc_target = params.soc_target
    soh_initial = params.soh_initial
    energy_buying_costs=params.energy_buying_costs
    energy_selling_costs=params.energy_selling_costs
    dod_cycle_curve = params.dod_cycle_curve

    total_energy_cost = 0
    total_batt_deg_cost = 0
    net_load = []
    lambda_e_values = []
    charging_schedule = ()

    r_net_charging = params.r_cmax * ALPHA  # Effective charging rate considering efficiency
    r_net_discharging = params.r_dmin / BETA

    # global SOCs, SOHs, DODs
    soc, dod, soh = np.zeros(N), np.zeros(N), np.zeros(N)
    soc[0] = soc_arr
    dod[0] = 1 - soc_arr
    soh[0] = soh_initial

    for t in range(N):
        # the decision to charge/discharge depends on pv and load

        p_load_t = p_load[t]
        p_pv_t = p_pv[t]
        energy_buying_cost_t = energy_buying_costs[t]
        energy_selling_cost_t = energy_selling_costs[t]

        excess, deficit = 0, 0

        
        # if PV generation is more than load, charge the EV battery
        if p_pv_t > p_load_t:
            # excess pv available, charge the EV battery
            if soc[t] >= soc_target:
                current_rate = 0
                excess = p_pv_t - p_load_t
            else:
                current_rate = params.r_cmax
                excess = p_pv_t - p_load_t - r_net_charging
            if excess > 0:
                # sell to the grid 
                total_energy_cost -= excess * energy_selling_cost_t
                lambda_e_values.append(-energy_selling_cost_t)
            elif excess < 0: 
                # not enough pv, buy from grid
                total_energy_cost += abs(excess) * energy_buying_cost_t
                lambda_e_values.append(energy_buying_cost_t)
            
        # if load is more than pv generation, discharge the EV battery
        elif p_pv_t < p_load_t:
            # no excess pv, check if we can discharge the EV battery to meet load
            if soc[t] > SOC_MIN:
                current_rate = params.r_dmin  # discharging
                deficit = p_load_t - p_pv_t - r_net_discharging
                if deficit > 0:
                    # still not enough, buy from grid
                    total_energy_cost += abs(deficit) * energy_buying_cost_t
                    lambda_e_values.append(energy_buying_cost_t)
                elif deficit < 0:
                    # excess after discharging, sell to grid
                    total_energy_cost -= abs(deficit) * energy_selling_cost_t
                    lambda_e_values.append(-energy_selling_cost_t)
            else:
                # cannot discharge, buy from grid
                current_rate = params.r_cmax  # charge
                deficit = p_pv_t - p_load_t - r_net_charging
                total_energy_cost += abs(deficit) * energy_buying_cost_t
                lambda_e_values.append(energy_buying_cost_t)

        net_load.append(excess if excess else deficit)

        
        C_b, delta_soh, delta_dod = calculate_batt_deg_costs(current_rate, b_max, dod, t, dod_cycle_curve)
        total_batt_deg_cost += C_b

        # Update SOC, SOH, DOD
        if t < N - 1:
            soc_t1, soh_t1, dod_t1 = update_battery_state(soc[t], soh[t], dod[t], delta_dod, delta_soh)
            soc_t1, soh_t1, dod_t1 = enforce_constraints(soc_t1, soh_t1, dod_t1)
            soc[t+1], soh[t+1], dod[t+1] = soc_t1, soh_t1, dod_t1

        charging_schedule += (current_rate,)
        total_cost = total_energy_cost + total_batt_deg_cost
    return charging_schedule, total_cost, total_energy_cost, total_batt_deg_cost, net_load, soc, soh, dod
