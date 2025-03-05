import os
import pandas as pd
import numpy as np

class FireData:
    def __init__(self, data_dir: str, num_time_steps: int):
        self.data_dir = data_dir
        self.num_time_steps = num_time_steps
        self.fire_maps = []
        self._load_data()

    def _load_data(self):
        """
        Loads fire progression CSV files, checking for missing files.
        """
        for t in range(self.num_time_steps):
            filename = os.path.join(self.data_dir, f"ForestGrid{t:02d}.csv")
            if not os.path.exists(filename):
                print(f"WARNING: Missing file {filename}, skipping.")
                continue
            print(f"Loading: {filename}")  # Debugging output
            df = pd.read_csv(filename, header=None)
            fire_array = df.to_numpy()
            self.fire_maps.append(fire_array)

    def get_fire_map(self, time_step: int) -> np.ndarray:
        return self.fire_maps[time_step]
