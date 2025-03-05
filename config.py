# Module to parse CLI arguments and configuration files.
import argparse
import yaml
import os

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description="Run wildfire sensor deployment simulations.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="Path to the experiment configuration file.")
    parser.add_argument("--experiment_name", type=str, help="Override the experiment name from the config file.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    log_dir = config.get("logs", {}).get("output_directory", "wildfire_simulation/logs")
    os.makedirs(log_dir, exist_ok=True)
    return config
