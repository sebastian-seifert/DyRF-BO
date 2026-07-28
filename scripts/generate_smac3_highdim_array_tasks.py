#!/usr/bin/env python3
"""Task generator for standard SMAC3 BO baseline on High-Dimensional (>20D) & NAS benchmarks."""

import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_registry import BenchmarkRegistry

def generate_smac3_highdim_array_tasks(output_path: str = "results/smac3_highdim_array_tasks.txt") -> list:
    seeds = [1, 2, 3, 4, 5]
    trials = 50
    tasks = BenchmarkRegistry.HIGH_DIM_NAS

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    lines = []

    # Standard SMAC3 BO baseline runs across 1 baseline * 18 high-dim tasks * 5 seeds (90 tasks)
    # Uses standard SMAC3 Random Forest surrogate and standard Expected Improvement (EI) acqf
    for task in tasks:
        task_name = task.split("=")[-1]
        for seed in seeds:
            telemetry = f"results/epistemic_ei_highdim/baseline/telemetry_smac3_{task_name}_seed{seed}.json"
            line = (
                f"+optimizer/smac20=hpo "
                f"{task} "
                f"task.optimization_resources.n_trials={trials} "
                f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                f"optimizer_id=SMAC3-HPOFacade optimizer_container_id=SMAC3"
            )
            lines.append(line)

    with open(output_path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")

    print(f"Generated {len(lines)} High-Dim SMAC3 baseline tasks in {output_path}")
    return lines

if __name__ == "__main__":
    generate_smac3_highdim_array_tasks()
