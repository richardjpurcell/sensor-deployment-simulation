from .base_deployment import SensorDeployment
from typing import List, Tuple
import numpy as np
import random

class ReinforcementLearningDeployment(SensorDeployment):
    """
    Reinforcement Learning-based sensor deployment using an epsilon-greedy strategy.
    
    This algorithm assigns a value to each candidate grid cell based on a weighted
    combination of the burn probability (bp_map) and the fire map intensity. With a
    probability 'epsilon', it explores by choosing a random valid candidate; otherwise,
    it exploits by selecting the candidate with the highest computed value.
    
    Selected sensor positions are then excluded (along with nearby cells within the sensor
    radius) to ensure a minimum separation between sensors.
    """
    def __init__(self, num_sensors: int, sensor_radius: float, epsilon: float = 0.2, alpha: float = 0.7, beta: float = 0.3):
        """
        :param num_sensors: Number of sensors to deploy.
        :param sensor_radius: Minimum separation radius for sensors.
        :param epsilon: Probability of exploration in epsilon-greedy selection.
        :param alpha: Weight for the burn probability map.
        :param beta: Weight for the fire intensity map.
        """
        self.num_sensors = num_sensors
        self.sensor_radius = sensor_radius
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta

    def place_sensors(self, fire_map: np.ndarray, bp_map: np.ndarray, time_step: int, prev_positions=None) -> List[Tuple[int, int]]:
        """
        Place sensors using an epsilon-greedy approach.
        
        :param fire_map: Numpy array representing the current fire map.
        :param bp_map: Numpy array representing the burn probability map.
        :param time_step: The current time step (can be used to adjust exploration rate).
        :param prev_positions: Previously deployed sensor positions (optional) to penalize re-selection.
        :return: List of (row, col) tuples indicating sensor positions.
        """
        rows, cols = bp_map.shape
        candidates = []
        
        # Compute a value for each candidate cell based on weighted sum of bp_map and fire_map.
        # Optionally penalize cells that were used in the previous time step.
        for r in range(rows):
            for c in range(cols):
                value = self.alpha * bp_map[r, c] + self.beta * fire_map[r, c]
                if prev_positions and (r, c) in prev_positions:
                    value *= 0.5  # Penalize previously used positions
                candidates.append(((r, c), value))
        
        selected = []
        available_candidates = candidates.copy()
        
        # Epsilon-greedy sensor selection while enforcing a minimum separation.
        while len(selected) < self.num_sensors and available_candidates:
            # Filter candidates that are at least sensor_radius away from already selected sensors.
            valid_candidates = [
                (cand, val) for (cand, val) in available_candidates
                if all(np.sqrt((cand[0]-s[0])**2 + (cand[1]-s[1])**2) >= self.sensor_radius for s in selected)
            ]
            if not valid_candidates:
                break  # No valid candidates remaining.
            
            # With probability epsilon, explore by choosing a random valid candidate.
            if random.random() < self.epsilon:
                chosen = random.choice(valid_candidates)
            else:
                # Otherwise, choose the candidate with the maximum computed value.
                chosen = max(valid_candidates, key=lambda x: x[1])
            
            selected.append(chosen[0])
            
            # Remove candidates that are too close to the chosen sensor.
            new_available = []
            for cand, val in available_candidates:
                if np.sqrt((cand[0] - chosen[0][0])**2 + (cand[1] - chosen[0][1])**2) >= self.sensor_radius:
                    new_available.append((cand, val))
            available_candidates = new_available
        
        return selected
