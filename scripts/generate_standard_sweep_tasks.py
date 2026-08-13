#!/usr/bin/env python3
"""Master Task Generator for Standard 1D-15D Synthetic OOD Sweep (6,930 Tasks).

Generates task commands for all 55 standard synthetic functions (1D to 15D)
across 2 gap types (empty, sparse), 9 approaches (8 custom + smac3_variance baseline),
and 7 seeds (1..7).
"""

import os
from ep_extractors.synthetic_standard_benchmarks import get_all_standard_functions


def generate_standard_sweep_tasks():
    # Gather exactly 3 functions per dimension across 1D to 15D (45 functions)
    funcs = get_all_standard_functions()


    function_names = list(funcs.keys())
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
    task_file = "results/standard_sweep_tasks.txt"

    tasks = []
    for func in function_names:
        for gap in gap_types:
            for app in approaches:
                for seed in seeds:
                    cmd = (
                        f"python ep_extractors/synthetic_standard_benchmarks.py "
                        f"--func_name={func} --gap_type={gap} --approach={app} "
                        f"--seed={seed} --output_dir=results/standard_sweep"
                    )
                    tasks.append(cmd)

    with open(task_file, "w") as f:
        for t in tasks:
            f.write(t + "\n")

    print(f"Generated {len(tasks)} standard sweep tasks in '{task_file}' across {len(function_names)} functions.")
    return tasks

if __name__ == "__main__":
    generate_standard_sweep_tasks()
