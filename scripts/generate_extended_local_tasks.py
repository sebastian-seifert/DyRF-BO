#!/usr/bin/env python3
"""Task generator for extended local testing across 6 representative benchmark families (HPOBench, YAHPO, NAS301)."""

import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors import UQExtractorRegistry

EXTENDED_BENCHMARKS = [
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_svm_3",
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_xgboost_12",
    "+task/YAHPO/SO=cfg_lcbench_167168",
    "+task/YAHPO/SO=cfg_rbv2_ranger_16",
    "+task/YAHPO/SO=cfg_nb301_CIFAR10",
    "+task/YAHPO/SO=cfg_rbv2_super_1111",
]

def generate_extended_local_tasks(output_path: str = "results/extended_local_smoke_tasks.txt") -> list:
    seeds = [2]
    trials = 10
    acquisitions = ["ei", "pi", "lcb"]
    tasks = EXTENDED_BENCHMARKS
    approaches = UQExtractorRegistry.list_registered()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    lines = []

    # 1. DyRF-BO Epistemic runs across 3 acqs * 8 approaches * 6 tasks * 1 seed (144 tasks)
    for acq in acquisitions:
        for task in tasks:
            task_name = task.split("=")[-1]
            task_arg = f"+{task}" if not task.startswith("+") else task
            for approach in approaches:
                for seed in seeds:
                    telemetry = f"local_results/extended_smoke/{acq}/telemetry_epistemic_{acq}_{approach}_{task_name}_seed{seed}.json"
                    line = (
                        f"--config-dir carps_integration/configs "
                        f"+optimizer=smac20_custom_uncertainty "
                        f"++optimizer.acq_func_name={acq} "
                        f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={approach} "
                        f"{task_arg} task.optimization_resources.n_trials={trials} "
                        f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                        f"optimizer_id=SMAC20_CustomUncertainty_{acq}_{approach} optimizer_container_id=SMAC20_CustomUncertainty"
                    )
                    lines.append(line)

    # 2. Standard SMAC3 BO baseline runs across 3 acqs * 1 baseline * 6 tasks * 1 seed (18 tasks)
    for acq in acquisitions:
        for task in tasks:
            task_name = task.split("=")[-1]
            task_arg = f"+{task}" if not task.startswith("+") else task
            for seed in seeds:
                telemetry = f"local_results/extended_smoke/baseline/{acq}/telemetry_smac3_{acq}_{task_name}_seed{seed}.json"
                line = (
                    f"--config-dir carps_integration/configs "
                    f"+optimizer/smac20=hpo "
                    f"++optimizer.acq_func_name={acq} "
                    f"{task_arg} "
                    f"task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"optimizer_id=SMAC3_HPOFacade_{acq} optimizer_container_id=SMAC3"
                )
                lines.append(line)

    with open(output_path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")

    print(f"Generated {len(lines)} extended local tasks in {output_path}")
    return lines

if __name__ == "__main__":
    generate_extended_local_tasks()
