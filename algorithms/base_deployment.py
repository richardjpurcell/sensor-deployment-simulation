from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class SensorDeployment(ABC):
    """
    Abstract base class for sensor deployment algorithms.
    """
    def __init__(self, num_sensors: int, sensor_radius: float):
        self.num_sensors = num_sensors
        self.sensor_radius = sensor_radius

    @abstractmethod
    def place_sensors(self, fire_map: np.ndarray, bp_map: np.ndarray, time_step: int, prev_positions=None) -> List[Tuple[int, int]]:
        """
        Determines sensor positions for the given time step.
        Uses both the fire_map and the burn probability (bp) map to guide placement.
        """
        pass
