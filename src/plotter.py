import matplotlib.pyplot as plt
import os


def plot_all(results, params,save_dir):
    plot_load_and_pv(params, save_dir)
    plot_energy_costs(params, save_dir)
    plot_charging_schedules(results, save_dir)
    plot_soc_soh(results, save_dir)
    plot_pareto_front(results["methods"]["nsga"]["pareto_objectives"], save_dir)

def plot_load_and_pv(params, save_dir):
    plt.figure()
    plt.plot(params.p_load, label='Total Energy Consumption (kWh)')
    plt.plot(params.p_pv, label='PV Energy Generation (kWh)')
    plt.xlabel('Time (15min interval)')
    plt.ylabel('Consumption (kWh)')
    plt.tight_layout()
    plt.legend()
    plt.title('Load and PV Generation Profile')
    plt.savefig(os.path.join(save_dir, "load_pv_profile.png"))
    plt.close()

def plot_energy_costs(params, save_dir):
    plt.figure()
    plt.plot(params.energy_buying_costs, label='Energy buying costs ($/kWh)')
    plt.plot(params.energy_selling_costs, label='Energy selling costs ($/kWh)')
    plt.xlabel('Time (15min interval)')
    plt.ylabel('Consumption (kWh)')
    plt.title('Energy Cost Profile')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "energy_cost_profile.png"))
    plt.close()


def plot_charging_schedules(results, save_dir):
    plt.figure()
    plt.plot(results["methods"]["nsga"]["nsga_charging_schedule"], label='NSGA-II Optimal Schedule')
    plt.plot(results["methods"]["no_v2g"]["no_v2g_charging_schedule"], label='No_V2G Schedule')
    plt.plot(results["methods"]["rulebased"]["rulebased_charging_schedule"], label='Rule-based Schedule')
    plt.title('Charging/Discharging Schedule over Time')
    plt.ylabel('Charging Rate (kWh)')
    plt.xlabel('Time step')
    plt.ylim(-3, 3)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "charging_schedule.png"))
    plt.close()


def plot_soc_soh(results, save_dir):
    plt.figure()
    plt.plot(results["methods"]["nsga"]["nsga_socs"] , label='NSGA-II SOC')
    plt.plot(results["methods"]["no_v2g"]["no_v2g_socs"], label='No_V2G Charging SOC')
    plt.plot(results["methods"]["rulebased"]["rulebased_socs"], label='Rule-based SOC')
    plt.title('Battery SOC over time')
    plt.xlabel("Time step")
    plt.ylabel("Level")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "soc_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(results["methods"]["nsga"]["nsga_sohs"], label='NSGA-II SOH')
    plt.plot(results["methods"]["no_v2g"]["no_v2g_sohs"], label='No_V2G Charging SOH')
    plt.plot(results["methods"]["rulebased"]["rulebased_sohs"], label='Rule-based SOH')
    plt.title('Battery SOH over time')
    plt.xlabel("Time step")
    plt.ylabel("Level")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "soh_curve.png"))
    plt.close()

def plot_pareto_front(pareto_objectives, save_dir):
    energy_costs, degradations = zip(*pareto_objectives)
    plt.scatter(energy_costs, degradations)
    plt.xlabel("Energy Cost Objective ($)")
    plt.ylabel("Battery Degradation Objective ($)")
    plt.title("Trade-off between Energy Cost and Battery Degradation")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "pareto_front.png"))
    plt.close()