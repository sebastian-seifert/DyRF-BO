#!/usr/bin/env python3
"""Task generator for EU-guided EI Bayesian Optimization on High-Dimensional (>20D) & NAS benchmarks."""

import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors import UQExtractorRegistry
from scripts.benchmark_registry import BenchmarkRegistry

def generate_highdim_array_tasks(output_path: str = "results/epistemic_ei_highdim_array_tasks.txt") -> list:
    seeds = [1, 2, 3, 4, 5]
    trials = 50
    acq = "ei"
    tasks = BenchmarkRegistry.HIGH_DIM_NAS
    approaches = UQExtractorRegistry.list_registered()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    lines = []

    # 1. DyRF-BO Epistemic EI runs across 8 approaches * 21 high-dim tasks * 5 seeds (840 tasks)
    for task in tasks:
        task_name = task.split("=")[-1]
        for approach in approaches:
            for seed in seeds:
                telemetry = f"results/epistemic_ei_highdim/ei/telemetry_epistemic_{acq}_{approach}_{task_name}_seed{seed}.json"
                line = (
                    f"+optimizer=dyrf_epistemic_{acq} "
                    f"++optimizer.acq_func_name={acq} "
                    f"++optimizer.extractor_name={approach} "
                    f"++optimizer.acq_uncertainty_type=epistemic "
                    f"++optimizer.enable_adaptation=false "
                    f"{task} task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"optimizer_id=CARPSDynamicRF_epistemic_{acq} optimizer_container_id=CARPSDynamicRF"
                )
                lines.append(line)

    # 2. Standard SMAC3 BO baseline runs across 1 baseline * 21 high-dim tasks * 5 seeds (105 tasks)
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

    print(f"Generated {len(lines)} High-Dim EI array tasks in {output_path}")
    return lines

if __name__ == "__main__":
    generate_highdim_array_tasks()
