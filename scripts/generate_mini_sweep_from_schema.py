#!/usr/bin/env python3
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_schema import BenchmarkMasterConfig, RFConfig, DataConfig, ProximityConfig

def create_mini_sweep(output_dir="results/mini_sweep_test", functions=None, rf_configs=None, seeds=None):
    if functions is None:
        functions = ["sin", "damped_osc"]
    if rf_configs is None:
        rf_configs = ["A"]
    if seeds is None:
        seeds = [1]

    os.makedirs(output_dir, exist_ok=True)
    tasks_file = os.path.join(output_dir, "tasks.txt")
    config_file = os.path.join(output_dir, "master_config.json")

    # Create custom master hyperparameter config using schema
    master_cfg = BenchmarkMasterConfig(
        data=DataConfig(gap_type="empty", ood_type="hypercube", seed=1),
        proximity=ProximityConfig(
            topological_decay_lambda=[0.5, 1.0],
            k_neighbors=["10", "20"],
            density_scaling_alpha=[1.0]
        )
    )

    # Save master hyperparameter schema snapshot
    with open(config_file, "w") as f:
        f.write(master_cfg.to_json())

    lines = []
    raw_dir = os.path.join(output_dir, "raw")

    for func in sorted(functions):
        for rf_preset in rf_configs:
            for seed in seeds:
                # Update task-specific configs
                task_cfg = BenchmarkMasterConfig(
                    data=DataConfig(gap_type="empty", ood_type="hypercube", seed=seed),
                    rf=RFConfig.from_preset(rf_preset),
                    proximity=master_cfg.proximity
                )
                cmd_lines = task_cfg.generate_sweep_task_lines(func_name=func, output_dir=raw_dir)
                lines.extend(cmd_lines)

    with open(tasks_file, "w") as f:
        for l in lines:
            f.write(l + "\n")

    print(f"Generated {len(lines)} task lines in {tasks_file}")
    print(f"Saved master config snapshot in {config_file}")
    return tasks_file, config_file

if __name__ == "__main__":
    create_mini_sweep()
