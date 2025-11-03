import numpy as np
import random
from exp_params import Params, get_number_of_cycles
# import os

# GENETIC ALGORITHM FOR V2G OPTIMIZATION
# output for logs
OUTPUT_LOGS = []  

# CONSTANTS
# EV is 2025 Kia EV9 specs

T = 6  # Time horizon (hours)
N = T*4 #total number of data points
B_MAX = 60     # should be in AH #76.1  # Battery capacity (kWh)
B_MIN = 0  # Minimum battery capacity (kWh)
DELTA_T = 1  # 15mins Time step  (hours)
SOC_MAX = 1  # Maximum state of charge (percentage of total capacity)
SOC_MIN = 0.3  # Minimum state of charge (percentage of total capacity)
SOH_MIN = 0.7  # Minimum state of health (percentage of nominal capacity)
ALPHA = 0.99  # Battery charging efficiency (percentage)
BETA = 0.85  # Battery discharging efficiency (percentage)
LAMBDA_B = 200.0 # Conversion coeefficient of battery degradation to monetary cost ($/kWh)
# R_DMIN = -2.75    # 11kWh for 1 hr => 11*0.25 = 2.75kW (for many level 2 charging stations) # Maximum discharge rate (kWh) 
# R_CMAX = 2.75  # 11kWh for 1 hr  Maximum charge rate (kWh) (for 




def create_initial_population(population_size, r_dmin, r_cmax):
    population = []
    for _ in range(population_size):
        individual = [random.choice([r_dmin, 0, r_cmax]) for _ in range(N)]  # Random charge/discharge rate
        population.append(tuple(individual))
    return population


def calculate_net_energy_cost(current_rate, p_load_t, p_pv_t, energy_buying_cost, energy_selling_cost):
    if current_rate > 0:
        r_net_t = ALPHA * current_rate  # Net charge rate (kWh)
    else:
        r_net_t = (1 / BETA) * current_rate  # Net discharge rate (kWh)
    # Net load (kw)
    L_t = r_net_t + p_load_t - p_pv_t

    lambda_e = energy_buying_cost if current_rate >= 0 else -1 * energy_selling_cost
    # Energy cost
    C_e = lambda_e * L_t  * DELTA_T
    return C_e, L_t, lambda_e


def calculate_batt_deg_costs(current_rate, b_max, dod, t, dod_cycle_curve):
    # change in DOD
    delta_dod = (-current_rate * DELTA_T) / b_max
    # print("delta_dod:", delta_dod)

    # Decrease in SOH
    if current_rate <= 0:  # Discharging
        delta_soh = (1 - SOH_MIN)/get_number_of_cycles(dod_cycle_curve, ((dod[t-1] + delta_dod) if t > 0 else dod[t])*100)  # Convert to percentage
        # print("delta_soh:", delta_soh)
    else:  # Charging
        # delta_dod = -delta_dod
        delta_soh = 0

    # Battery degradation cost
    C_b = LAMBDA_B * delta_soh
    return C_b, delta_soh, delta_dod


def evaluate_fitness(charging_schedule, params: Params) -> float:
    # calculate fitness based on the cost function
    # get parameters
    
    p_load = params.p_load
    p_pv = params.p_pv
    b_max = params.b_max
    penalty_enabled = params.penalty_enabled
    soc_arr = params.soc_arr
    soc_target = params.soc_target
    w_e = params.w_e
    w_b = params.w_b
    r_cmax = params.r_cmax
    dod_cycle_curve = params.dod_cycle_curve
    energy_buying_costs=params.energy_buying_costs
    energy_selling_costs=params.energy_selling_costs
    
    total_energy_cost = 0
    total_batt_deg_cost = 0
    # net_energy_costs = []
    # batt_deg_costs = []
    net_load = []
    delta_soh_values = []
    lambda_e_values = []
    ind = list(charging_schedule)

    # global SOCs, SOHs, DODs
    soc, soh, dod = np.zeros(N), np.zeros(N), np.zeros(N)
    soc[0] = soc_arr
    dod[0] = 1 - soc_arr
    soh[0] = 0.9
    
    for t in range(N):
        ind[t] = r_cmax if t==0 else ind[t]
        current_rate = ind[t]

        C_e, L_t, lambda_e = calculate_net_energy_cost(current_rate, p_load[t], p_pv[t], energy_buying_costs[t], energy_selling_costs[t])
        C_b, delta_soh, delta_dod = calculate_batt_deg_costs(current_rate, b_max, dod, t, dod_cycle_curve)
        # Update total costs
        total_energy_cost += C_e
        total_batt_deg_cost += C_b
        # net_energy_costs.append(total_energy_cost)
        # batt_deg_costs.append(total_batt_deg_cost)
        net_load.append(L_t)
        delta_soh_values.append(delta_soh)
        lambda_e_values.append(lambda_e)
        
        # Update battery state
        if t < N - 1:
            soc_t1, soh_t1, dod_t1 = update_battery_state(soc[t], soh[t], dod[t], delta_dod, delta_soh)
            soc_t1, soh_t1, dod_t1, next_rate = enforce_constraints(soc_t1, soh_t1, dod_t1, ind[t+1], r_cmax)
            soc[t+1], soh[t+1], dod[t+1], ind[t+1] = soc_t1, soh_t1, dod_t1, next_rate
            
    if penalty_enabled:
        # penalty calculation based on smoothness to reduce rapid fluctuations`
        diffs = np.diff(ind)
        ind_max = np.max(np.abs(ind)) if np.any(ind) else 1.0
        S = np.mean(diffs**2)
        S_norm = S / (ind_max**2 + 1e-9)
    else:
        S_norm = 0
    
    # add penalty to ensure final SOC meets target
    SOC_error = abs(soc[-1] - soc_target)
    target_deviation_penalty = 100000000 * (SOC_error**2)  # e.g., penalty_weight = 1000

    # calculate total cost
    total_cost = (w_e * total_energy_cost) + (w_b * total_batt_deg_cost) + S_norm + target_deviation_penalty
    return total_cost, total_energy_cost, total_batt_deg_cost, S_norm, target_deviation_penalty, ind, soc, soh, dod, net_load, lambda_e_values

# Updating battery state:

def update_battery_state(SOC_t, SOH_t, DOD_t, delta_dod, delta_soh):
    SOC_t1 = SOC_t - delta_dod
    SOH_t1 = SOH_t - delta_soh
    DOD_t1 = DOD_t + delta_dod
    return SOC_t1, SOH_t1, DOD_t1

def enforce_constraints(SOC_t, SOH_t, DOD_t, next_rate, r_cmax):
    if SOC_t > SOC_MAX:
        SOC_t = SOC_MAX
        next_rate = 0    # if soc is at max, don't charge
    elif SOC_t < SOC_MIN:
        SOC_t = SOC_MIN
        next_rate = r_cmax    # charge if soc is below min

    if SOH_t < SOH_MIN:
        SOH_t = SOH_MIN

    if DOD_t < 0:
        DOD_t = 0
    elif DOD_t > 1 - SOC_MIN:
        DOD_t = 1 - SOC_MIN

    return SOC_t, SOH_t, DOD_t, next_rate

def selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = min(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

# Crossover function
def crossover(parent1, parent2):    
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return tuple(child1), tuple(child2)

def mutation(individual, mutation_rate, r_dmin, r_cmax):
    """Mutate an individual by changing genes to one of the other two possible values."""
    possible_values = [r_dmin, r_cmax, 0]
    new_individual = []
    for gene in individual:
        if random.random() < mutation_rate:
            # Pick a new value different from the current one
            new_value = random.choice([v for v in possible_values if v != gene])
            new_individual.append(new_value)
        else:
            new_individual.append(gene)
    return tuple(new_individual)


def genetic_algorithm(params: Params,
                    # population_size=50,
                    # generations=100,
                    # crossover_rate=0.8,
                    # mutation_rate=0.1,
                    # r_dmin=R_DMIN,
                    # r_cmax=R_CMAX
                    ):


    population_size = params.population_size
    generations = params.generations
    crossover_rate = params.crossover_rate
    mutation_rate = params.mutation_rate
    r_dmin = params.r_dmin
    r_cmax = params.r_cmax

    # # global SOCs, SOHs, DODs, final_soc, final_soh, final_dod
    # global final_soc, final_soh, final_dod, best_energy_cost, best_batt_deg_cost
    # Initialize population
    population = create_initial_population(population_size, r_dmin, r_cmax)
    best_solution = None
    best_fitness = float("inf")
    final_socs = []
    final_sohs = []
    final_dods = []
    best_energy_cost = 0
    best_batt_deg_cost = 0
    net_load = 0
    lambda_e_s = 0

    best_performers = []
    print("pop:", population[0])
    
    for gen in range(generations):
        print("generation ", gen)
            # Evaluate fitness
        fitnesses, net_energy_costs, batt_deg_costs, S_norm, target_deviation_penalty, ind, SOCs, SOHs, DODs, net_loads, lambda_e_values = zip(*([evaluate_fitness(ind, params) for ind in population]))
        # print("ind: ", ind)
        # Track best
        gen_best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        # population[gen_best_idx] = tuple(ind)
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_solution = population[gen_best_idx]
            final_socs = SOCs[gen_best_idx]
            final_sohs = SOHs[gen_best_idx]
            final_dods = DODs[gen_best_idx]
            best_energy_cost = net_energy_costs[gen_best_idx]
            best_batt_deg_cost = batt_deg_costs[gen_best_idx]
            final_S_norm = S_norm[gen_best_idx]
            final_target_deviation_penalty = target_deviation_penalty[gen_best_idx]
            net_load = net_loads[gen_best_idx]
            lambda_e_s = lambda_e_values[gen_best_idx]

            # print(f"New best fitness: {-best_fitness:.4f} at generation {gen+1}")
            # print(f"Best solution: {best_solution}")
            # print(f"Best energy cost: {best_energy_cost}")
            # print(f"Best battery degradation cost: {best_batt_deg_cost}")
            # print(f"S_norm: {final_S_norm}")
            # print(f"Target deviation penalty: {final_target_deviation_penalty}\n")
            best_performers.append((best_solution, best_fitness, best_energy_cost, best_batt_deg_cost, final_S_norm, final_target_deviation_penalty, final_socs, final_sohs, final_dods, net_load, lambda_e_s))
            # all_populations.append(population[:])
            # print("best_solution:", best_solution)

        # Selection
        selected = selection(population, fitnesses)

        # Crossover
        offspring = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[(i+1) % len(selected)]
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            offspring.extend([child1, child2])

        # Mutation
        mutated = [mutation(ind, mutation_rate, r_dmin, r_cmax) for ind in offspring]

        # New generation
        # Replace the old population with the new one, preserving the best individual
        # mutated[0] = best_solution
        population = mutated

        output_log = f"Generation {gen+1}: Best Fitness = {best_fitness:.4f}\n"
        OUTPUT_LOGS.append(output_log)
    # final_population = all_populations[-1]
    # final_fitnesses = [evaluate_fitness(ind, params) for ind in final_population]
    # print(output_log)
    # print(SOCs)
    return best_solution, best_fitness, best_energy_cost, best_batt_deg_cost, final_S_norm, final_target_deviation_penalty, final_socs, final_sohs, final_dods, net_load, lambda_e_s, best_performers

    

# def run_ga(params: Params):
#     fitness_value = 0
#     net_energy_costs = []
#     batt_deg_costs = []
#     final_soh = []
#     final_soc = []
#     final_dod = []

#     best_solution, best_fitness, final_socs, final_sohs, final_dods, best_energy_cost, best_batt_deg_cost, net_load, lambda_e_s, best_performers = genetic_algorithm(params)

    
    