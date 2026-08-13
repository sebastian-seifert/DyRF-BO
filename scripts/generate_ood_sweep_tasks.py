#!/usr/bin/env python3
"""Master Task Generator for Synthetic OOD Benchmark Sweep (630 Tasks).

Generates task lines for the 5 new custom functions across 2 gap types (empty, sparse),
9 approaches (8 custom extractors + smac3_variance baseline), and 7 seeds (1..7).
"""

import os

def generate_ood_sweep_tasks():
    functions = ["ackley_2d", "rosenbrock_2d", "ackley_4d", "rosenbrock_4d", "hartmann_6d"]
    gap_types = ["empty", "sparse"]
    approaches = [
        "standard_disagreement",
        "chen_variance",
        "shaker_entropy",
        "likelihood_credal",
        "standard_proximity",
        "proximity_b",
        "proximity_bc",
        "proximity_auto_lambda",
        "smac3_variance"
    ]
    seeds = [1, 2, 3, 4, 5, 6, 7]

    os.makedirs("results", exist_ok=True)
    task_file = "results/ood_sweep_tasks.txt"

    tasks = []
    for func in functions:
        for gap in gap_types:
            for app in approaches:
                for seed in seeds:
                    cmd = (
                        f"python ep_extractors/synthetic_ood_benchmarks.py "
                        f"--func_name={func} --gap_type={gap} --approach={app} "
                        f"--seed={seed} --output_dir=results/ood_sweep"
                    )
                    tasks.append(cmd)

    with open(task_file, "w") as f:
        for t in tasks:
            f.write(t + "\n")

    print(f"Generated {len(tasks)} OOD sweep tasks in '{task_file}'")
    return tasks

if __name__ == "__main__":
    generate_ood_sweep_tasks()
