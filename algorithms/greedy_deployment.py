from .base_deployment import SensorDeployment
from typing import List, Tuple
import numpy as np

class GreedyDeployment(SensorDeployment):
    """
    Greedy algorithm for sensor deployment guided by the burn probability (BP) map.
    At time step t, it uses the provided BP map (ideally from t-1) to rank candidate cells.
    Additionally, it applies:
      - A penalty for any candidate that was selected in the previous time step,
        so that even if the BP value is high, that cell's effective score is reduced.
      - A repulsive mechanism: once a sensor is selected, candidates within a minimum
        separation distance are penalized (their effective BP is set to 0).
    """
    def place_sensors(self, fire_map: np.ndarray, bp_map: np.ndarray, time_step: int, prev_positions=None) -> List[Tuple[int, int]]:
        rows, cols = bp_map.shape
        candidates = []
        for r in range(rows):
            for c in range(cols):
                bp_val = bp_map[r, c]
                # If a sensor was located at this cell in the previous time step,
                # apply a penalty (reduce effective BP by 90%).
                if prev_positions and (r, c) in prev_positions:
                    effective_bp = bp_val * 0.1
                else:
                    effective_bp = bp_val
                candidates.append(((r, c), effective_bp))
        
        # Sort candidates by effective BP (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        # Set a minimum separation distance (here, equal to sensor_radius; adjust if needed)
        sensor_min_distance = self.sensor_radius
        
        # Greedily select sensor positions while enforcing separation.
        while len(selected) < self.num_sensors and candidates:
            candidate, score = candidates.pop(0)
            # Check if candidate is far enough from already selected sensors.
            if all(np.sqrt((candidate[0] - s[0])**2 + (candidate[1] - s[1])**2) >= sensor_min_distance for s in selected):
                selected.append(candidate)
                # Penalize candidates that are too close to this newly selected candidate.
                new_candidates = []
                for cand, sc in candidates:
                    distance = np.sqrt((candidate[0] - cand[0])**2 + (candidate[1] - cand[1])**2)
                    if distance < sensor_min_distance:
                        new_candidates.append((cand, 0))
                    else:
                        new_candidates.append((cand, sc))
                candidates = new_candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
        
        return selected
