#!/usr/bin/env python3
"""Task generator for 3-Way Expected Improvement (EI) Head-to-Head Comparison:
1. Direct Variance Replacement (SMAC20_CustomUncertainty_ei_*)
2. Decoupled Additive Hybrid (CARPSDynamicRF_AdditiveEpistemic_ei_*)
3. Standard SMAC3 HPOFacade Baseline (SMAC3_HPOFacade_ei)
"""

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

def generate_ei_comparison_sweep_tasks(
    output_path: str = "results/ei_comparison_sweep_tasks.txt",
    seeds: list = None,
    trials: int = 50,
    beta_max: float = 1.0,
    warmup_ratio: float = 0.20
) -> list:
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    tasks = CarpsBBSubsetRegistry.get_dev_tasks()
    approaches = list(ACTIVE_APPROACHES)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    lines = []

    # 1. Paradigm 1: Direct Variance Replacement (EI) -> 7 approaches * 20 tasks * 5 seeds = 700 tasks
    for task in tasks:
        task_name = task.split("/")[-1]
        task_arg = f"+{task}" if not task.startswith("+") else task
        for approach in approaches:
            for seed in seeds:
                telemetry = f"results/ei_head_to_head/direct/telemetry_direct_ei_{approach}_{task_name}_seed{seed}.json"
                line = (
                    f"--config-dir carps_integration/configs "
                    f"+optimizer=smac20_custom_uncertainty "
                    f"++optimizer.acq_func_name=ei "
                    f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={approach} "
                    f"{task_arg} task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"optimizer_id=SMAC20_CustomUncertainty_ei_{approach} optimizer_container_id=SMAC20_CustomUncertainty"
                )
                lines.append(line)

    # 2. Paradigm 2: Decoupled Additive Hybrid (EI) -> 7 approaches * 20 tasks * 5 seeds = 700 tasks
    for task in tasks:
        task_name = task.split("/")[-1]
        task_arg = f"+{task}" if not task.startswith("+") else task
        for approach in approaches:
            for seed in seeds:
                telemetry = f"results/ei_head_to_head/additive/telemetry_additive_ei_{approach}_{task_name}_seed{seed}.json"
                line = (
                    f"--config-dir carps_integration/configs "
                    f"+optimizer=dyrf_additive_epistemic_ei "
                    f"++optimizer.extractor_name={approach} "
                    f"++optimizer.beta_max={beta_max} "
                    f"++optimizer.warmup_ratio={warmup_ratio} "
                    f"{task_arg} task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"optimizer_id=CARPSDynamicRF_AdditiveEpistemic_ei_{approach} optimizer_container_id=CARPSDynamicRF"
                )
                lines.append(line)

    # 3. Paradigm 3: Standard SMAC3 HPOFacade Baseline (EI) -> 1 baseline * 20 tasks * 5 seeds = 100 tasks
    for task in tasks:
        task_name = task.split("/")[-1]
        task_arg = f"+{task}" if not task.startswith("+") else task
        for seed in seeds:
            telemetry = f"results/ei_head_to_head/baseline/telemetry_smac3_ei_{task_name}_seed{seed}.json"
            line = (
                f"--config-dir carps_integration/configs "
                f"+optimizer/smac20=hpo "
                f"++optimizer.acq_func_name=ei "
                f"{task_arg} "
                f"task.optimization_resources.n_trials={trials} "
                f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                f"optimizer_id=SMAC3_HPOFacade_ei optimizer_container_id=SMAC3"
            )
            lines.append(line)

    with open(output_path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")

    print(f"Generated {len(lines)} EI Comparison sweep tasks in {output_path}")
    return lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EI Comparison Sweep Tasks")
    parser.add_argument("--output", type=str, default="results/ei_comparison_sweep_tasks.txt", help="Output task file path")
    parser.add_argument("--beta_max", type=float, default=1.0, help="Maximum beta exploration bonus")
    parser.add_argument("--warmup_ratio", type=float, default=0.20, help="Warmup budget fraction")
    parser.add_argument("--trials", type=int, default=50, help="Trials per run")
    args = parser.parse_args()

    generate_ei_comparison_sweep_tasks(
        output_path=args.output,
        beta_max=args.beta_max,
        warmup_ratio=args.warmup_ratio,
        trials=args.trials
    )
