import numpy as np
import pandas as pd


PV_GEN_BASE_PATH = 'datasets/pv-generation/'
HOME_LOAD_BASE_PATH = 'datasets/home-load-profiles/'
ELECTRICITY_RATE_PATH = 'datasets/electricity_cost_profile_15min.csv'
SOC_ARRIVAL_PATH = 'datasets/soc_arrival_data.csv'
DOD_TO_CYCLES_PATH = 'datasets/dod_to_cycles.csv'

columns = {
    "p_load": "total_consumption_kwh",
    "p_pv": "pv_energy_gen_kWh",
    "energy_buying_prices": "energy_buying_price($/kWh)",
    "energy_selling_prices": "energy_selling_price($/kWh)",
}


def fit_dod_curve():
# Fit polynomial to DoD vs Cycles data
    data = pd.read_csv(DOD_TO_CYCLES_PATH) 
    x = data.iloc[:,0].to_numpy()
    y = data.iloc[:,1].to_numpy()

    # Take log of y
    logy = np.log(y)

    # Polynomial fit in log-space
    degree = 14
    coeffs = np.polyfit(x, logy, degree)
    poly_log = np.poly1d(coeffs)
    return poly_log
    

def get_household_energy_profile(month, day, hour, data_args):
    pv_filepath, home_load_filepath, electricity_rate_filepath = get_filepaths(data_args)

    electricity_cost_model = data_args['electricity_cost_model']
    if electricity_cost_model == 'Fixed':
        energy_buying_prices = [0.1]*24
        energy_selling_prices = [0.08]*24
    else:
        energy_buying_prices = get_profile_data(electricity_rate_filepath, month, day, hour, "energy_buying_prices")
        energy_selling_prices = get_profile_data(electricity_rate_filepath, month, day, hour, "energy_selling_prices")

    p_load = get_profile_data(home_load_filepath, month, day, hour, "p_load")
    p_pv = get_profile_data(pv_filepath, month, day, hour, "p_pv")
    return p_load, p_pv, energy_buying_prices, energy_selling_prices



def get_filepaths(data_args):
    pv_size = data_args['pv_size']
    home_size = data_args['home_size']
    home_id = data_args['home_id']
    pv_filepath = f"{PV_GEN_BASE_PATH}pvwatts_{pv_size}kw_15min.csv"
    home_load_filepath = f"{HOME_LOAD_BASE_PATH}{home_size}_home_{home_id}.csv"
    electricity_rate_filepath = ELECTRICITY_RATE_PATH
    return pv_filepath, home_load_filepath, electricity_rate_filepath


def get_profile_data(filepath, month, day, hour, column_name):
    df = pd.read_csv(filepath)
    data = df[(df["month"] == int(month)) & (df['day'] == int(day)) & (df['hour'] == int(hour))].index
    if not data.empty:
        start_idx = data[0]
        end_idx = start_idx + 24 #for 6hrs
        data = df[columns.get(column_name)][start_idx:end_idx].values
        return data
    else:
        return "error"


def get_soc_on_arrival():
    sessions = pd.read_csv(SOC_ARRIVAL_PATH)[['SoC_start']]
    soc_arr = sessions[(sessions['SoC_start'] >= 0) & (sessions['SoC_start'] <= 100)]['SoC_start'].sample(n=1).iloc[0]
    soc_arr = soc_arr/100
    return soc_arr

def get_soh_initial():
    return np.random.normal(loc=0.95, scale=0.03)


    
