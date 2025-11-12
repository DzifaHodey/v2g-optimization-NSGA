import os
import numpy as np
import matplotlib.pyplot as plt

from exp_params import Params, get_number_of_cycles

T = 6       # Time horizon (hours)
N = T*4         #total number of timesteps
B_MIN = 0       # Minimum battery capacity (kWh)
DELTA_T = 1      # 15mins Time step  (hours)
SOC_MIN = 0.3    # Minimum state of charge (percentage of total capacity)
SOH_MIN = 0.7       # Minimum state of health (percentage of nominal capacity)
ALPHA = 0.99        # Battery charging efficiency (percentage)
BETA = 0.85         # Battery discharging efficiency (percentage)
LAMBDA_B = 200.0    # Conversion coeefficient of battery degradation to monetary cost ($/kWh)

# === STEP 1: INITIALIZE POPULATION ===
def initialize_population(pop_size, r_dmin, r_cmax):
    return np.random.uniform(r_dmin, r_cmax, (pop_size, N))  # e.g. charge/discharge rates


# === STEP 2: EVALUATE OBJECTIVES ===
def evaluate_objectives(charging_schedule, params: Params) -> float: 
    p_load = params.p_load
    p_pv = params.p_pv
    b_max = params.b_max
    soc_arr = params.soc_arr
    soc_target = params.soc_target
    soh_initial = params.soh_initial
    dod_cycle_curve = params.dod_cycle_curve
    energy_buying_costs=params.energy_buying_costs
    energy_selling_costs=params.energy_selling_costs
    
    total_energy_cost = 0
    total_batt_deg_cost = 0
    net_load = []
    delta_soh_values = []
    lambda_e_values = []

    soc, soh, dod = np.zeros(N), np.zeros(N), np.zeros(N)
    soc[0] = soc_arr
    dod[0] = 1 - soc_arr
    soh[0] = soh_initial
    
    for t in range(N):
        current_charging_rate = charging_schedule[t]
        C_e, L_t, lambda_e = calculate_net_energy_cost(current_charging_rate, p_load[t], p_pv[t], energy_buying_costs[t], energy_selling_costs[t])
        C_b, delta_soh, delta_dod = calculate_batt_deg_costs(current_charging_rate, b_max, dod, t, dod_cycle_curve)
        
        # Update total costs
        total_energy_cost += C_e
        total_batt_deg_cost += C_b

        net_load.append(L_t)
        delta_soh_values.append(delta_soh)
        lambda_e_values.append(lambda_e)
        
        # Update battery state
        if t < N - 1:
            soc_t1, soh_t1, dod_t1 = update_battery_state(soc[t], soh[t], dod[t], delta_dod, delta_soh)
            soc[t+1], soh[t+1], dod[t+1] = soc_t1, soh_t1, dod_t1

            # Add penalty if SOC and SOH are at or below minimum
            if soc_t1 <= SOC_MIN + 1e-6:  # allow tiny numerical tolerance
                total_batt_deg_cost += 1 

            if soh_t1 <= SOH_MIN + 1e-6:
                total_batt_deg_cost += 1

            
    # Add penalty for not reaching target SOC
    SOC_error = abs(soc[-1] - soc_target)
    penalty_weight = 1e5  # tune this value depending on the scale of your costs
    target_penalty = penalty_weight * (SOC_error ** 2)

    # Combine penalties with existing costs
    total_energy_cost += target_penalty/2
    total_batt_deg_cost += (target_penalty/2)

    return (total_energy_cost, total_batt_deg_cost), (net_load, soc, soh, dod, lambda_e_values)

def calculate_net_energy_cost(current_charging_rate, p_load_t, p_pv_t, energy_buying_cost, energy_selling_cost):
    # net charge/discharge rate
    r_net_t = (current_charging_rate / ALPHA) if current_charging_rate > 0 else (current_charging_rate * BETA)
    
    # Net load (kWh), negative load means excess energy from pv/battery
    L_t = r_net_t + p_load_t - p_pv_t
    lambda_e = energy_buying_cost if L_t>=0 else energy_selling_cost

    C_e = lambda_e * L_t
    return C_e, L_t, lambda_e

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

def update_battery_state(SOC_t, SOH_t, DOD_t, delta_dod, delta_soh):
    SOC_t1 = SOC_t - delta_dod
    SOH_t1 = SOH_t - delta_soh
    DOD_t1 = DOD_t + delta_dod
    return SOC_t1, SOH_t1, DOD_t1

# === STEP 3: NON-DOMINATED SORTING ===
def non_dominated_sort(objective_values):
    """
    Returns a list of fronts. Each front is a list of individual indices.
    """
    S = [[] for _ in range(len(objective_values))]
    n = [0] * len(objective_values)
    rank = [0] * len(objective_values)
    fronts = [[]]

    for p in range(len(objective_values)):
        for q in range(len(objective_values)):
            if dominates(objective_values[p], objective_values[q]):
                S[p].append(q)
            elif dominates(objective_values[q], objective_values[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts



# # === DOMINATION CHECK ===
def dominates(ind1, ind2):
    """True if ind1 dominates ind2 (all objs <=, one strictly <)."""
    return all(i <= j for i, j in zip(ind1, ind2)) and any(i < j for i, j in zip(ind1, ind2))

# === STEP 4: CROWDING DISTANCE ===
def compute_crowding_distance(front, objective_values):
    n = len(front)
    if n == 0:
        return np.array([])
    distances = np.zeros(n)
    
    # Two objectives → loop over each
    for m in range(2):
        sorted_idx = sorted(range(n), key=lambda i: objective_values[front[i]][m])
        obj_vals = [objective_values[front[i]][m] for i in sorted_idx]
        minv, maxv = obj_vals[0], obj_vals[-1]
        distances[sorted_idx[0]] = distances[sorted_idx[-1]] = np.inf
        if maxv == minv:
            continue
        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (obj_vals[k + 1] - obj_vals[k - 1]) / (maxv - minv)
    return distances


# # === STEP 5: SELECTION ===
def select_parents(population, fronts, num_parents):
    selected = []
    for front in fronts:
        if len(selected) + len(front) > num_parents:
            distances = compute_crowding_distance(front, population)
            sorted_indices = np.argsort(-distances)  # descending
            selected.extend([front[i] for i in sorted_indices[:num_parents - len(selected)]])
            break
        else:
            selected.extend(front)
    return [population[i] for i in selected]

# === STEP 6: CROSSOVER AND MUTATION ===
def crossover(parent1, parent2, crossover_rate=0.9):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1)-1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(individual, r_dmin, r_cmax, mutation_rate=0.1):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.normal(0, 0.1)
    return np.clip(individual, r_dmin, r_cmax)



# === MAIN LOOP ===
def nsga2(params: Params):
    pop_size = params.population_size
    generations = params.generations
    crossover_rate = params.crossover_rate
    mutation_rate = params.mutation_rate
    r_dmin = params.r_dmin
    r_cmax = params.r_cmax
    fronts = []
    final_population = []
    objective_values = (0,0)

    population = initialize_population(pop_size, r_dmin, r_cmax)
    for gen in range(generations):
        # Evaluate all objectives
        objective_values, tracked_values = zip(*[evaluate_objectives(ind, params) for ind in population])
        fronts = non_dominated_sort(objective_values)
        final_population = population

        # Create offspring
        offspring = []
        parents = select_parents(population, fronts, pop_size // 2)
        while len(offspring) < pop_size:
            parent_indices = np.random.choice(len(parents), 2, replace=False)
            p1, p2 = parents[parent_indices[0]], parents[parent_indices[1]]

            c1, c2 = crossover(p1, p2, crossover_rate)
            offspring.extend([mutate(c1, r_dmin, r_cmax, mutation_rate), mutate(c2, r_dmin, r_cmax, mutation_rate)])
        
        # Combine and re-sort
        combined = np.vstack((population, offspring))
        combined_obj, _ = zip(*[evaluate_objectives(ind, params) for ind in combined])
        combined_fronts = non_dominated_sort(combined_obj)

        # Select next generation
        new_population = []
        for front in combined_fronts:
            if len(new_population) + len(front) > pop_size:
                distances = compute_crowding_distance(front, combined_obj)
                sorted_front = sorted(zip(front, distances), key=lambda x: -x[1])
                for idx, _ in sorted_front[:pop_size - len(new_population)]:
                    new_population.append(combined[idx])
                break
            else:
                new_population.extend([combined[i] for i in front])
        population = new_population

        print(f"Generation {gen}: Pareto front size = {len(fronts[0])}")

    pareto_solutions = [final_population[i] for i in fronts[0]]
    pareto_objectives = [objective_values[i] for i in fronts[0]]
    tracked = [tracked_values[i] for i in fronts[0]]
    print(len(tracked))
    net_loads, socs, sohs, dods, lambda_e_s = zip(*tracked)

    return pareto_solutions, pareto_objectives, net_loads, socs, sohs, dods, lambda_e_s



def run_nsga(params: Params):
    pareto_solutions, pareto_objectives,  net_loads, socs, sohs, dods, lambda_e_s = nsga2(params)

    fitness_values = [sum(obj) for obj in pareto_objectives]
    best_idx = np.argmin(fitness_values)
    best_solution = pareto_solutions[best_idx]
    # best_obj = pareto_objectives[best_idx]
    best_net_load = net_loads[best_idx]
    best_socs = socs[best_idx]
    best_sohs = sohs[best_idx]
    best_dods = dods[best_idx]
    lambda_e_s = lambda_e_s[best_idx]

    batt_deg_cost, energy_cost = 0,0 
    for t in range(N):
        b_cost, _, _ = calculate_batt_deg_costs(best_solution[t], params.b_max, best_dods, t, params.dod_cycle_curve)
        batt_deg_cost += b_cost
        
        e_cost, _, _= calculate_net_energy_cost(best_solution[t], params.p_load[t], params.p_pv[t], params.energy_buying_costs[t], params.energy_selling_costs[t])
        energy_cost += e_cost

    print("Best solution charging schedule (kWh):", best_solution)
    print(f"Energy cost: ${energy_cost:.3f}")
    print(f"Degradation cost: ${batt_deg_cost:.3f}")
    print(f"Total cost: ${energy_cost + batt_deg_cost:.3f}")
    return pareto_solutions, pareto_objectives, best_solution, (energy_cost, batt_deg_cost), best_net_load, best_socs, best_sohs, best_dods, lambda_e_s

def plot_pareto_front(pareto_objectives, dir):
    energy_costs, degradations = zip(*pareto_objectives)
    plt.scatter(energy_costs, degradations)
    plt.xlabel("Energy Cost")
    plt.ylabel("Battery Degradation")
    plt.title("Pareto Front (Cost of Charging EV")
    plt.savefig(os.path.join(dir, "pareto_front.png"))