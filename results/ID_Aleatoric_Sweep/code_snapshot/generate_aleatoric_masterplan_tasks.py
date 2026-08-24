#!/usr/bin/env python3
"""Task Generator for Aleatoric Masterplan Sweep (750 Tasks).

Generates task commands for 5 benchmark functions x 6 noise regimes x 5 RF configs x 5 seeds.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ep_extractors.aleatoric_shaker_masterplan import get_benchmark_functions, get_noise_regimes, get_rf_configs

def generate_aleatoric_masterplan_tasks():
    funcs = list(get_benchmark_functions().keys())
    noises = list(get_noise_regimes().keys())
    rf_configs = list(get_rf_configs().keys())
    seeds = [1, 2, 3, 4, 5]

    os.makedirs("results", exist_ok=True)
    task_file = "results/aleatoric_masterplan_tasks.txt"

    tasks = []
    for f in funcs:
        for n in noises:
            for rf_cfg in rf_configs:
                for s in seeds:
                    cmd = (
                        f"python -c 'from ep_extractors.aleatoric_shaker_masterplan import run_single_aleatoric_experiment, evaluate_aleatoric_metrics; "
                        f"import json, os; "
                        f"res = run_single_aleatoric_experiment(\"{f}\", \"{n}\", \"{rf_cfg}\", seed={s}); "
                        f"os.makedirs(\"results/aleatoric_masterplan\", exist_ok=True); "
                        f"open(\"results/aleatoric_masterplan/res_{f}_{n}_{rf_cfg}_seed{s}.json\", \"w\").write(json.dumps(res, indent=2))'"
                    )
                    tasks.append(cmd)

    with open(task_file, "w") as tf:
        for t in tasks:
            tf.write(t + "\n")

    print(f"Generated {len(tasks)} tasks in '{task_file}'.")
    return tasks

if __name__ == "__main__":
    generate_aleatoric_masterplan_tasks()
