# Module to run the wildfire simulation.
from typing import List, Tuple
import numpy as np
import os
import json
import pandas as pd
from fire_data import FireData
from algorithms.base_deployment import SensorDeployment

class WildfireSimulation:
    def __init__(self, 
                 fire_data: FireData, 
                 deployment_algo: SensorDeployment, 
                 detection_threshold: float,
                 experiment_name: str,
                 bp_file: str = "",
                 bp_decay: float = 0.9):
        """
        :param fire_data: FireData object with fire progression maps.
        :param deployment_algo: A SensorDeployment instance for sensor placement.
        :param detection_threshold: Threshold to decide if a cell is burning.
        :param experiment_name: Name of the experiment (from config). 
                                Used to create the log folder: logs/<experiment_name>/timesteps.
        :param bp_file: Path to a burn probability CSV. If empty or file not found, a random map is generated.
        :param bp_decay: Decay factor for burn probability over time.
        """
        self.fire_data = fire_data
        self.deployment_algo = deployment_algo
        self.detection_threshold = detection_threshold
        self.num_time_steps = fire_data.num_time_steps
        self.sensor_positions_history = []
        self.coverage_history = []
        self.bp_maps = []  # to store burn probability maps over time
        
        self.experiment_name = experiment_name  
        self.log_dir = os.path.join("logs", self.experiment_name, "timesteps")
        os.makedirs(self.log_dir, exist_ok=True)
        self.bp_log_dir = os.path.join("logs", self.experiment_name, "bp_maps")
        os.makedirs(self.bp_log_dir, exist_ok=True)
        self.bp_decay = bp_decay
        
        # Initialize burn probability map
        sample_map = self.fire_data.get_fire_map(0)
        rows, cols = sample_map.shape
        if bp_file:
            if os.path.exists(bp_file):
                print(f"Loading burn probability CSV from: {bp_file}")
                # Use whitespace delimiter for the burn probability CSV
                self.bp_map = pd.read_csv(bp_file, header=None, delim_whitespace=True).to_numpy().astype(float)
            else:
                print(f"Burn probability CSV not found at {bp_file}. Generating random burn probability map.")
                self.bp_map = np.random.rand(rows, cols)
        else:
            print("No burn probability CSV provided. Generating random burn probability map.")
            self.bp_map = np.random.rand(rows, cols)


        
        # Save the initial BP map
        self.bp_maps.append(self.bp_map.copy())

    def run_simulation(self):
        prev_sensor_positions = None
        for t in range(self.num_time_steps):
            fire_map = self.fire_data.get_fire_map(t)
            
            # Use the BP map from the previous time step for sensor placement:
            if t == 0:
                bp_for_placement = self.bp_map
            else:
                bp_for_placement = self.bp_maps[t-1]
            
            sensor_positions = self.deployment_algo.place_sensors(fire_map, bp_for_placement, t, prev_positions=prev_sensor_positions)
            self.sensor_positions_history.append(sensor_positions)
            
            # For t > 0, apply decay to the current BP map.
            if t > 0:
                self.bp_map = self.bp_map * self.bp_decay
            
            # Update BP map based on sensor discoveries: 
            # If a sensor detects fire (cell value in fire_map >= threshold),
            # update cells within its radius to 1.0.
            for (r, c) in sensor_positions:
                if fire_map[r, c] >= self.detection_threshold:
                    self._update_bp_map_with_sensor(r, c)
            
            # Save the updated BP map for this time step.
            self.bp_maps.append(self.bp_map.copy())
            self._save_bp_map(t, self.bp_map)
            
            # Compute coverage.
            coverage = self._compute_coverage(fire_map, sensor_positions)
            self.coverage_history.append(coverage)
            prev_sensor_positions = sensor_positions
            
            # Log the current time step's sensor positions and coverage.
            self.log_time_step(t, sensor_positions, coverage)

    def _update_bp_map_with_sensor(self, r_center: int, c_center: int):
        """
        Update the burn probability map: set cells within sensor_radius to 1.0.
        """
        rows, cols = self.bp_map.shape
        r_min = max(0, r_center - int(self.deployment_algo.sensor_radius))
        r_max = min(rows, r_center + int(self.deployment_algo.sensor_radius) + 1)
        c_min = max(0, c_center - int(self.deployment_algo.sensor_radius))
        c_max = min(cols, c_center + int(self.deployment_algo.sensor_radius) + 1)
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                if np.sqrt((r - r_center)**2 + (c - c_center)**2) <= self.deployment_algo.sensor_radius:
                    self.bp_map[r, c] = 1.0

    def _compute_coverage(self, fire_map: np.ndarray, sensor_positions: List[Tuple[int, int]]) -> float:
        burning_cells = [(r, c) for r in range(fire_map.shape[0])
                         for c in range(fire_map.shape[1]) if fire_map[r, c] >= self.detection_threshold]
        if not burning_cells:
            return 1.0
        detected = 0
        radius_sq = self.deployment_algo.sensor_radius ** 2
        for r_fire, c_fire in burning_cells:
            for r_sens, c_sens in sensor_positions:
                if (r_fire - r_sens)**2 + (c_fire - c_sens)**2 <= radius_sq:
                    detected += 1
                    break
        return detected / len(burning_cells)
    
    def log_time_step(self, time_step: int, sensor_positions: List[Tuple[int, int]], coverage: float):
        """
        Logs sensor positions and coverage for a given time step to a JSON file.
        Files are saved under logs/<experiment_name>/timesteps.
        """
        log_data = {
            "time_step": time_step,
            "sensor_positions": sensor_positions,
            "coverage": coverage
        }
        log_file = os.path.join(self.log_dir, f"time_step_{time_step:02d}.json")
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
    
    def _save_bp_map(self, time_step: int, bp_map: np.ndarray):
        """
        Save the burn probability map as a CSV file.
        """
        bp_file = os.path.join(self.bp_log_dir, f"bp_map_{time_step:02d}.csv")
        np.savetxt(bp_file, bp_map, delimiter=",")
