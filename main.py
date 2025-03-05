'''
To load another config: python main.py --config configs/another_config.yaml
'''

import os
import numpy as np
from fire_data import FireData
from simulation import WildfireSimulation
from metrics import SimulationMetrics, compare_metrics
from config import parse_arguments
from algorithms.greedy_deployment import GreedyDeployment
from algorithms.reinforcement_learning_deployment import ReinforcementLearningDeployment
from algorithms.genetic_deployment import GeneticDeployment
from visualization.dashboard import interactive_dashboard, load_time_step_logs

def main():
    config = parse_arguments()
    exp_name = config.get("experiment_name", "DefaultExperiment")
    run_mode = config.get("run_mode", "simulate")
    
    summary_log_dir = os.path.join(config["logs"]["output_directory"], exp_name)
    os.makedirs(summary_log_dir, exist_ok=True)
    
    # Retrieve burn probability configuration.
    bp_config = config.get("burn_probability", {})
    bp_file = bp_config.get("file", "")
    bp_decay = bp_config.get("decay_factor", 0.9)
    
    if run_mode.lower() == "simulate":
        data_dir = config["data"]["directory"]
        num_time_steps = config["data"]["num_time_steps"]
        fire_data = FireData(data_dir, num_time_steps)
        
        num_sensors = config["deployment"]["num_sensors"]
        sensor_radius = config["deployment"]["sensor_radius"]
        algorithm_choice = config["deployment"]["algorithm"]
        
        if algorithm_choice.lower() == "greedy":
            deployment_algo = GreedyDeployment(num_sensors, sensor_radius)
        elif algorithm_choice.lower() == "reinforcementlearning":
            # Retrieve additional RL parameters from the config.
            epsilon = config["deployment"].get("epsilon", 0.2)
            alpha = config["deployment"].get("alpha", 0.7)
            beta = config["deployment"].get("beta", 0.3)
            deployment_algo = ReinforcementLearningDeployment(num_sensors, sensor_radius, epsilon, alpha, beta)
        elif algorithm_choice.lower() == "genetic":
            population_size = config["deployment"].get("population_size", 50)
            num_generations = config["deployment"].get("num_generations", 100)
            mutation_rate = config["deployment"].get("mutation_rate", 0.1)
            alpha = config["deployment"].get("alpha", 0.5)
            beta = config["deployment"].get("beta", 0.5)
            deployment_algo = GeneticDeployment(num_sensors, sensor_radius, population_size, num_generations, mutation_rate, alpha, beta)
        else:
            raise ValueError(f"Unknown algorithm specified: {algorithm_choice}")
        
        detection_threshold = config["simulation"]["detection_threshold"]
        simulation = WildfireSimulation(fire_data, deployment_algo, detection_threshold, exp_name, bp_file, bp_decay)
        simulation.run_simulation()
        
        coverage_history = simulation.coverage_history
        metrics_obj = SimulationMetrics()
        metrics_obj.compute_metrics(coverage_history)
        print("Simulation Results:")
        print(metrics_obj)
        metrics_obj.save_metrics(summary_log_dir, algorithm_choice.lower())
        
        # Launch interactive dashboard visualization if enabled.
        if config.get("dashboard", {}).get("enabled", False):
            interactive_dashboard(fire_data.fire_maps,
                                    simulation.sensor_positions_history,
                                    coverage_history,
                                    simulation.bp_maps)
    
    elif run_mode.lower() == "visualize_only":
        data_dir = config["data"]["directory"]
        num_time_steps = config["data"]["num_time_steps"]
        fire_data = FireData(data_dir, num_time_steps)
        
        timestep_log_dir = os.path.join(config["logs"]["output_directory"], exp_name, "timesteps")
        time_step_logs = load_time_step_logs(timestep_log_dir)
        sensor_positions_history = [time_step_logs[t]["sensor_positions"] for t in sorted(time_step_logs.keys())]
        coverage_history = [time_step_logs[t]["coverage"] for t in sorted(time_step_logs.keys())]
        
        bp_map_dir = os.path.join(config["logs"]["output_directory"], exp_name, "bp_maps")
        bp_maps = []
        for t in range(num_time_steps):
            bp_file_path = os.path.join(bp_map_dir, f"bp_map_{t:02d}.csv")
            if os.path.exists(bp_file_path):
                bp_map = np.genfromtxt(bp_file_path, delimiter=",")
                bp_maps.append(bp_map)
        
        if config.get("dashboard", {}).get("enabled", False):
            interactive_dashboard(fire_data.fire_maps,
                                    sensor_positions_history,
                                    coverage_history,
                                    bp_maps)
    
    else:
        raise ValueError(f"Unknown run_mode specified: {run_mode}")

if __name__ == "__main__":
    main()
