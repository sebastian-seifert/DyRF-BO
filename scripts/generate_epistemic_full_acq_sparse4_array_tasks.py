#!/usr/bin/env python3
"""Task generator for Sparse 4-Task Epistemic BO Sanity Sweep across EI, PI, and LCB acquisitions."""

import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors import UQExtractorRegistry

SPARSE_4_TASKS = [
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_svm_3",
    "+task/YAHPO/SO=cfg_rbv2_glmnet_375",
    "+task/YAHPO/SO=cfg_lcbench_167168",
    "+task/YAHPO/SO=cfg_rbv2_super_1040",
]

def generate_full_acq_sparse4_array_tasks(output_path: str = "results/epistemic_full_acq_sparse4_array_tasks.txt") -> list:
    seeds = [1, 2, 3, 4, 5]
    trials = 50
    acquisitions = ["ei", "pi", "lcb"]
    tasks = SPARSE_4_TASKS
    approaches = UQExtractorRegistry.list_registered()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    lines = []

    # 1. Custom Uncertainty runs across 3 acqs * 8 approaches * 4 tasks * 5 seeds (480 tasks)
    for acq in acquisitions:
        for task in tasks:
            task_name = task.split("=")[-1]
            for approach in approaches:
                for seed in seeds:
                    telemetry = f"results/epistemic_full_acq_sparse4/{acq}/telemetry_epistemic_{acq}_{approach}_{task_name}_seed{seed}.json"
                    line = (
                        f"+optimizer=smac20_custom_uncertainty "
                        f"++optimizer.acq_func_name={acq} "
                        f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={approach} "
                        f"{task} task.optimization_resources.n_trials={trials} "
                        f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                        f"optimizer_id=SMAC20_CustomUncertainty_{acq}_{approach} optimizer_container_id=SMAC20_CustomUncertainty"
                    )
                    lines.append(line)

    # 2. Standard SMAC3 BO baseline runs across 3 acqs * 1 baseline * 4 tasks * 5 seeds (60 tasks)
    for acq in acquisitions:
        for task in tasks:
            task_name = task.split("=")[-1]
            for seed in seeds:
                telemetry = f"results/epistemic_full_acq_sparse4/baseline/{acq}/telemetry_smac3_{acq}_{task_name}_seed{seed}.json"
                line = (
                    f"+optimizer/smac20=hpo "
                    f"++optimizer.acq_func_name={acq} "
                    f"{task} "
                    f"task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"optimizer_id=SMAC3_HPOFacade_{acq} optimizer_container_id=SMAC3"
                )
                lines.append(line)

    with open(output_path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")

    print(f"Generated {len(lines)} Sparse 4 Full Acq array tasks in {output_path}")
    return lines

if __name__ == "__main__":
    generate_full_acq_sparse4_array_tasks()
