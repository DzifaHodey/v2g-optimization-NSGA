import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import Params, update_battery_state, calculate_batt_deg_costs
from constants import *


# RULE-BASED CHARGING STRATEGY

def run_rule_based_v2g(params:Params):
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
    charging_schedule = []

    r_net_charging = params.r_cmax/ ALPHA  # Effective charging rate considering efficiency
    r_net_discharging = params.r_dmin * BETA

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
            if soc[t] > 0.7:
                current_rate = params.r_dmin  # discharging
                deficit = p_load_t - p_pv_t + r_net_discharging
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
            soc[t+1], soh[t+1], dod[t+1] = soc_t1, soh_t1, dod_t1

        charging_schedule.append(current_rate)
        total_cost = total_energy_cost + total_batt_deg_cost

    print("--------------Rule-based V2G Results:--------------")
    print("Charging schedule (kWh):", charging_schedule)
    print(f"Energy cost: ${total_energy_cost:.3f}")
    print(f"Degradation cost: ${total_batt_deg_cost:.3f}")
    print(f"Total cost: ${total_cost:.3f}")

    results = {
        "rulebased_charging_schedule": charging_schedule,
        "rulebased_energy_cost": total_energy_cost,
        "rulebased_batt_deg_cost": total_batt_deg_cost,
        "rulebased_total_cost": total_cost,
        "rulebased_net_load": net_load,
        "rulebased_socs": soc,
        "rulebased_sohs": soh,
        "rulebased_dods": dod,
        "rulebased_energy_prices": lambda_e_values
    }
    return results