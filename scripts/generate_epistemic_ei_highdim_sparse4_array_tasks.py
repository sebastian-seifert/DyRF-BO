#!/usr/bin/env python3
"""Task generator for Sparse 4-Task EU-guided EI Bayesian Optimization on High-Dimensional (>20D) benchmarks."""

import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors import UQExtractorRegistry

SPARSE_4_HIGHDIM_TASKS = [
    "+task/YAHPO/SO=cfg_nb301_CIFAR10",
    "+task/YAHPO/SO=cfg_rbv2_super_1040",
    "+task/YAHPO/SO=cfg_rbv2_super_1050",
    "+task/YAHPO/SO=cfg_rbv2_super_1457",
]

def generate_highdim_sparse4_array_tasks(output_path: str = "results/epistemic_ei_highdim_sparse4_array_tasks.txt") -> list:
    seeds = [1, 2, 3, 4, 5]
    trials = 50
    acq = "ei"
    tasks = SPARSE_4_HIGHDIM_TASKS
    approaches = UQExtractorRegistry.list_registered()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    lines = []

    # 1. Custom Uncertainty runs across 8 approaches * 4 high-dim tasks * 5 seeds (160 tasks)
    for task in tasks:
        task_name = task.split("=")[-1]
        for approach in approaches:
            for seed in seeds:
                telemetry = f"results/epistemic_ei_highdim_sparse4/ei/telemetry_epistemic_{acq}_{approach}_{task_name}_seed{seed}.json"
                line = (
                    f"+optimizer=smac20_custom_uncertainty "
                    f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={approach} "
                    f"{task} task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"optimizer_id=SMAC20_CustomUncertainty_{approach} optimizer_container_id=SMAC20_CustomUncertainty"
                )
                lines.append(line)

    # 2. Standard SMAC3 BO baseline runs across 1 baseline * 4 high-dim tasks * 5 seeds (20 tasks)
    for task in tasks:
        task_name = task.split("=")[-1]
        for seed in seeds:
            telemetry = f"results/epistemic_ei_highdim_sparse4/baseline/telemetry_smac3_{task_name}_seed{seed}.json"
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

    print(f"Generated {len(lines)} Sparse 4 High-Dim EI array tasks in {output_path}")
    return lines

if __name__ == "__main__":
    generate_highdim_sparse4_array_tasks()
