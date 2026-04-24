# CONSTANTS
# EV is 2025 Kia EV9
  
T = 6       # Time horizon (hours)
N = T*4         #total number of timesteps
B_MIN = 0       # Minimum battery capacity (kWh)
DELTA_T = 1      # 15mins Time step  (hours)
SOC_MIN = 0.3    # Minimum state of charge (percentage of total capacity)
SOH_MIN = 0.7       # Minimum state of health (percentage of nominal capacity)
ALPHA = 0.99        # Battery charging efficiency (percentage)
BETA = 0.85         # Battery discharging efficiency (percentage)
LAMBDA_B = 200.0    # Conversion coeefficient of battery degradation to monetary cost ($/kWh)
