# Wildfire Simulation Project

This project simulates wildfire sensor deployment using various algorithms. It includes modules for:
- **Data loading** (`fire_data.py`),
- **Simulation** (`simulation.py`),
- **Metrics calculation** (`metrics.py`),
- **Deployment algorithms** (`algorithms/`),
- **Visualization** (`visualization/`), and
- **Configuration** (`configs/`).

The overall goal is to place sensors on a grid representing a forest to detect wildfires as they spread over multiple time steps. Different deployment algorithms can be selected via a YAML configuration file.

---

## Directory Structure
```
wildfire_simulation/
├── algorithms/
│   ├── __init__.py
│   ├── base_deployment.py
│   ├── greedy_deployment.py
│   ├── genetic_deployment.py
│   ├── reinforcement_learning_deployment.py
│   └── ilp_deployment.py          # (Optional / placeholder if using ILP approach)
├── configs/
│   ├── default_config.yaml
│   ├── genetic_config.yaml
│   ├── rl_config.yaml
│   └── another_config.yaml
├── data/
│   └── Sub40x40/
│       ├── grids/
│       │   ├── Grids0.csv
│       │   ├── Grids1.csv
│       │   └── ...
│       └── Stats/
│           └── BProb.csv          # (Optional burn probability file)
├── logs/
│   └── Wildfire_Greedy_Exp02/
│       ├── timesteps/
│       └── bp_maps/
├── visualization/
│   ├── dashboard.py
│   └── __init__.py
├── fire_data.py
├── main.py
├── metrics.py
├── simulation.py
└── README.md
```

Below is a brief explanation of each key directory:

- **algorithms/**  
  Contains Python files implementing different sensor deployment algorithms.  
  - `base_deployment.py` is an abstract base class specifying the required interface (`place_sensors(...)`).  
  - `greedy_deployment.py` implements a simple greedy strategy.  
  - `genetic_deployment.py` implements a genetic algorithm for sensor placement.  
  - `reinforcement_learning_deployment.py` uses an RL-inspired (epsilon-greedy) approach.  
  - `ilp_deployment.py` could hold an Integer Linear Programming approach (placeholder).

- **configs/**  
  Holds YAML configuration files. Each file specifies parameters for the simulation (e.g., number of sensors, sensor radius, which algorithm to use, number of time steps, etc.).  
  - `default_config.yaml` is an example default configuration.  
  - `genetic_config.yaml` and `rl_config.yaml` are specialized configs for the genetic and RL algorithms, respectively.  

- **data/**  
  Contains subfolders with CSV files representing fire spread over multiple time steps. The simulator reads these files to determine how the fire evolves.  
  - `Sub40x40/` is an example dataset. It can also include a `Stats/` folder with optional burn probability data.

- **logs/**  
  Stores output logs and results generated during simulation runs, including sensor placement at each time step and burn probability maps. The subdirectory structure is typically `logs/<experiment_name>/`.

- **visualization/**  
  Holds modules for creating dashboards, animations, and interactive plots to visualize the fire spread and sensor deployment.  
  - `dashboard.py` provides an interactive interface to step through each time step of the simulation.

- **fire_data.py**  
  Loads the CSV files from the `data/` directory into NumPy arrays.

- **simulation.py**  
  Runs the time-stepped simulation. It applies a chosen deployment algorithm, updates burn probability maps, logs sensor positions, and calculates coverage.

- **metrics.py**  
  Calculates metrics (e.g., coverage over time) and can compare different algorithms’ performance.

- **main.py**  
  The main entry point for running the simulation. It:
  1. Parses command-line arguments (e.g., `--config`).
  2. Loads the specified config file.
  3. Instantiates the chosen deployment algorithm.
  4. Runs the simulation and (optionally) launches a visualization dashboard.

---

## Running the Simulation

1. **Install Dependencies:**  
   Make sure you have Python 3.x and any necessary libraries (e.g., `numpy`, `pandas`, `matplotlib`, etc.) installed.

2. **Choose or Create a Config File:**  
   Select one of the YAML files in `configs/` (e.g., `genetic_config.yaml`) or create a new one.

3. **Run the Script:**  
   ```bash
   python main.py --config configs/genetic_config.yaml

