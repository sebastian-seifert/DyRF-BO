#!/usr/bin/env python3
"""Task generator for CARP-S BBsubset (Blackbox Single-Objective) Dev Set using Decoupled Additive Epistemic Acquisition."""

import os
import sys
import argparse

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry

# 7 Active Epistemic Approaches (Chen is strictly excluded)
ACTIVE_APPROACHES = [
    "standard_disagreement",
    "shaker_entropy",
    "likelihood_credal",
    "standard_proximity",
    "proximity_b",
    "proximity_bc",
    "proximity_auto_lambda"
]

def generate_bbsubset_additive_hybrid_tasks(
    output_path: str = "results/bbsubset_additive_hybrid_tasks.txt",
    seeds: list = None,
    trials: int = 50,
    beta_max: float = 1.0,
    warmup_ratio: float = 0.20
) -> list:
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    acquisitions = ["ei", "pi", "lcb"]
    tasks = CarpsBBSubsetRegistry.get_dev_tasks()
    approaches = list(ACTIVE_APPROACHES)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    lines = []

    # 1. DyRF-BO Additive Epistemic runs (3 acqs * 7 approaches * 20 dev tasks * 5 seeds = 2,100 tasks)
    for acq in acquisitions:
        for task in tasks:
            task_name = task.split("/")[-1]
            task_arg = f"+{task}" if not task.startswith("+") else task
            for approach in approaches:
                for seed in seeds:
                    telemetry = f"results/bbsubset_additive_hybrid/{acq}/telemetry_additive_{acq}_{approach}_{task_name}_seed{seed}.json"
                    line = (
                        f"--config-dir carps_integration/configs "
                        f"+optimizer=dyrf_additive_epistemic_{acq} "
                        f"++optimizer.extractor_name={approach} "
                        f"++optimizer.beta_max={beta_max} "
                        f"++optimizer.warmup_ratio={warmup_ratio} "
                        f"{task_arg} task.optimization_resources.n_trials={trials} "
                        f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                        f"optimizer_id=CARPSDynamicRF_AdditiveEpistemic_{acq}_{approach} optimizer_container_id=CARPSDynamicRF"
                    )
                    lines.append(line)

    # 2. Standard SMAC3 BO baseline runs (3 acqs * 1 baseline * 20 dev tasks * 5 seeds = 300 tasks)
    for acq in acquisitions:
        for task in tasks:
            task_name = task.split("/")[-1]
            task_arg = f"+{task}" if not task.startswith("+") else task
            for seed in seeds:
                telemetry = f"results/bbsubset_additive_hybrid/baseline/{acq}/telemetry_smac3_{acq}_{task_name}_seed{seed}.json"
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

    print(f"Generated {len(lines)} CARP-S BBsubset additive hybrid tasks in {output_path}")
    return lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CARP-S BBsubset Additive Hybrid Tasks")
    parser.add_argument("--output", type=str, default="results/bbsubset_additive_hybrid_tasks.txt", help="Output file path")
    parser.add_argument("--beta_max", type=float, default=1.0, help="Maximum beta exploration bonus")
    parser.add_argument("--warmup_ratio", type=float, default=0.20, help="Warmup fraction of budget")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials per run")
    args = parser.parse_args()

    generate_bbsubset_additive_hybrid_tasks(
        output_path=args.output,
        beta_max=args.beta_max,
        warmup_ratio=args.warmup_ratio,
        trials=args.trials
    )
