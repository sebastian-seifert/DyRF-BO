#!/usr/bin/env python3
"""Task generator for local smoke testing across EI, PI, and LCB acquisitions on representative benchmarks."""

import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors import UQExtractorRegistry

SMOKE_BENCHMARKS = [
    "task/YAHPO/SO=cfg_rbv2_super_1049",  # Sample benchmark 1
    "task/YAHPO/SO=cfg_rbv2_super_1040",  # Sample high-dim benchmark 2
]

def generate_local_smoke_tasks(output_path: str = "results/local_smoke_tasks.txt") -> list:
    seeds = [1]
    trials = 10  # Reduced trials for fast local smoke verification
    acquisitions = ["ei", "pi", "lcb"]
    tasks = SMOKE_BENCHMARKS
    approaches = UQExtractorRegistry.list_registered()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    lines = []

    # 1. DyRF-BO Epistemic runs across 3 acqs * 8 approaches * tasks * 1 seed
    for acq in acquisitions:
        for task in tasks:
            task_name = task.split("=")[-1]
            task_arg = f"+{task}" if not task.startswith("+") else task
            for approach in approaches:
                for seed in seeds:
                    telemetry = f"local_results/smoke/{acq}/telemetry_epistemic_{acq}_{approach}_{task_name}_seed{seed}.json"
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

    # 2. Standard SMAC3 BO baseline runs across 3 acqs * 1 baseline * tasks * 1 seed
    for acq in acquisitions:
        for task in tasks:
            task_name = task.split("=")[-1]
            task_arg = f"+{task}" if not task.startswith("+") else task
            for seed in seeds:
                telemetry = f"local_results/smoke/baseline/{acq}/telemetry_smac3_{acq}_{task_name}_seed{seed}.json"
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

    print(f"Generated {len(lines)} local smoke tasks in {output_path}")
    return lines

if __name__ == "__main__":
    generate_local_smoke_tasks()
