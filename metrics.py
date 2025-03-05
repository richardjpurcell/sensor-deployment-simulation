# Module for computing and comparing simulation metrics.
import os
import json
from typing import List

class SimulationMetrics:
    def __init__(self):
        self.coverage_over_time = []
        self.average_coverage = 0.0
        self.final_coverage = 0.0

    def compute_metrics(self, coverage_history: List[float]):
        self.coverage_over_time = coverage_history
        if coverage_history:
            self.average_coverage = sum(coverage_history) / len(coverage_history)
            self.final_coverage = coverage_history[-1]

    def save_metrics(self, log_dir: str, algo_name: str):
        data = {
            "coverage_over_time": self.coverage_over_time,
            "average_coverage": self.average_coverage,
            "final_coverage": self.final_coverage
        }
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, f"{algo_name}_metrics.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def __str__(self):
        return (f"Coverage over time: {self.coverage_over_time}\n"
                f"Average coverage: {self.average_coverage:.2f}\n"
                f"Final coverage: {self.final_coverage:.2f}\n")

def compare_metrics(algo_metrics: dict):
    for algo_name, metrics_obj in algo_metrics.items():
        print(f"=== {algo_name} ===")
        print(metrics_obj)
