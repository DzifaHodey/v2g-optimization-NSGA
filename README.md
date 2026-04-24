# V2G Optimization using NSGA-II

## Overview

This repository presents a **multi-objective optimization framework for Vehicle-to-Grid (V2G) and Vehicle-to-Home (V2H) operations** in PV-integrated residential homes. The framework determines optimal EV charging and discharging schedules to simultaneously minimize:

1. Household energy costs
2. Cyclic battery degradation

The problem is formulated as a multi-objective optimization task and solved using the **Non-dominated Sorting Genetic Algorithm II (NSGA-II)**.


## Key Features
- NSGA-II for Pareto optimization
- Battery degradation modeling based on Depth of Discharge
- Realistic data:
  - Home load
  - PV generation
  - Electricity prices
  - EV State of Charge (SoC) on arrival
- Comparative analysis against 2 charging strategies
  - No V2G: unidirectional EV charging
  - Rulebased V2G: rulebased bidirectional charging 
- Comprehensive analysis across different configurations of home load, PV generation, EV arrival data.


## Project Structure

```
.
├── datasets/                      # Raw and processed data (load, PV, electricity pricing, SoC arrival data)
├── src/
│   ├── v2g_strategies/               # Charging/V2G strategies
│   │   ├── nsga.py               # NSGA-II V2G optimization approach
│   │   ├── rule_based.py         # Heuristic-based V2G strategy
│   │   └── no_v2g.py             # Baseline (no V2G participation)
│   │
│   ├── main.py                   # Experiment runner 
│   ├── dataloader.py             # Data loader
│   ├── constants.py              # Constants across all strategies
│   ├── utils.py                  # Shared utilities
│   ├── plotter.py                # Plot results of single experiments
│   └── analysis.ipynb            # Post-processing and result visualization
│
├── results/                     # Saved outputs (plots, logs, Pareto fronts)
├── README.md
└── requirements.txt
```



## ⚙️ Core Components

### 1. `main.py` for running experiments
- Supports CLI arguments for methods enabled, algorithm parameters, load, pv, and electricity rate configurations.
- Supports multiple start times for experiments
- Generates plots based on results and stores both results and plots. 


### 2. NSGA-II Algorithm (`nsga.py`)
Implements the full optimization pipeline:

- Population initialization
- Objective evaluation
- Non-dominated sorting
- Crowding distance calculation
- Selection, crossover, mutation

### 3. Objective Function
Each candidate charging schedule is evaluated based on:

- **Energy Cost**
  - Buying/selling electricity rate
  - Net household load

- **Battery Degradation Cost**
  - Based on Depth of Discharge (DoD)
  - Uses empirical cycle-life models


## ▶️ How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Configure parameters
Update constants across charging strategies in `constants.py` as necessary, including:
- Time horizon
- Number of timesteps
- Battery parameters


### 3. Run experiments
Use default CLI arguments or pass your preferred values.
```
python main.py
```

## 📊 Outputs

- Timeseries data
- Summary of costs
- Full results for multi-configuration experiments
- Plots:
    - Charging Schedules
    - Trade-off curve (pareto plot) between energy cost and battery degradation
    - Load and pv profile
    - Battery State trend


##  Research Context

This code supports the thesis:

> *Multi-Objective Optimization of Energy Costs and EV Battery Health in V2G-Enabled Homes*

It is designed for:
- Academic research
- Energy systems simulation
- Smart grid optimization studies


