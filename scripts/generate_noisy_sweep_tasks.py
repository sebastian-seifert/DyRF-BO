#!/usr/bin/env python3
"""Task Generator for Noisy Benchmark EI Head-to-Head Sweeps (BBOB-Noisy & hetGP).

Generates exact, collision-free CARP-S Hydra task lists evaluating:
1. Standard SMAC3 HPOFacade Baseline (1 approach)
2. Direct Variance Drop-in Surrogates (4 approaches: standard_proximity, standard_disagreement, proximity_bc, shaker_entropy)
3. Decoupled Additive Epistemic BO (3 approaches: likelihood_credal, proximity_bc, shaker_entropy)

Total scale: 16 tasks * 8 approaches * 30 seeds = 3,840 task executions.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

NOISY_TASKS_HETGP = [
    "+task/Noisy/hetgp=cfg_yuan_wahba_1d",
    "+task/Noisy/hetgp=cfg_branin_2d",
    "+task/Noisy/hetgp=cfg_goldstein_price_2d",
    "+task/Noisy/hetgp=cfg_sinusoid_2d",
    "+task/Noisy/hetgp=cfg_sinusoid_4d",
]

NOISY_TASKS_BBOB = [
    "+task/Noisy/bbob=cfg_sphere_2d_gaussian",
    "+task/Noisy/bbob=cfg_sphere_2d_cauchy",
    "+task/Noisy/bbob=cfg_sphere_2d_uniform",
    "+task/Noisy/bbob=cfg_rosenbrock_2d_gaussian",
    "+task/Noisy/bbob=cfg_rosenbrock_2d_cauchy",
    "+task/Noisy/bbob=cfg_rastrigin_2d_gaussian",
    "+task/Noisy/bbob=cfg_bent_cigar_2d_gaussian",
    "+task/Noisy/bbob=cfg_attractive_sector_2d_gaussian",
    "+task/Noisy/bbob=cfg_schwefel_2d_gaussian",
    "+task/Noisy/bbob=cfg_sphere_4d_gaussian",
    "+task/Noisy/bbob=cfg_rosenbrock_4d_gaussian",
]


def get_noisy_tasks(suite: str = "all") -> List[str]:
    """Returns task arguments for the chosen suite ('all', 'hetgp', 'bbob')."""
    suite_lower = suite.lower()
    if suite_lower == "hetgp":
        return list(NOISY_TASKS_HETGP)
    elif suite_lower == "bbob":
        return list(NOISY_TASKS_BBOB)
    elif suite_lower == "all":
        return list(NOISY_TASKS_HETGP) + list(NOISY_TASKS_BBOB)
    else:
        raise ValueError(f"Unknown suite '{suite}'. Choose from 'all', 'hetgp', 'bbob'.")


def get_approach_configs() -> List[Dict[str, Any]]:
    """Returns the 8 competitor optimizer configurations for the EI head-to-head sweep."""
    return [
        # 1. Reference Baseline
        {
            "key": "smac3_baseline",
            "paradigm": "baseline",
            "optimizer_id": "SMAC3_HPOFacade_ei",
            "container_id": "SMAC3_HPOFacade",
        },
        # 2. Direct Drop-In Surrogates
        {
            "key": "standard_proximity",
            "paradigm": "direct",
            "extractor_name": "standard_proximity",
            "optimizer_id": "SMAC20_CustomUncertainty_ei_standard_proximity",
            "container_id": "SMAC20_CustomUncertainty",
        },
        {
            "key": "standard_disagreement",
            "paradigm": "direct",
            "extractor_name": "standard_disagreement",
            "optimizer_id": "SMAC20_CustomUncertainty_ei_standard_disagreement",
            "container_id": "SMAC20_CustomUncertainty",
        },
        {
            "key": "proximity_bc",
            "paradigm": "direct",
            "extractor_name": "proximity_bc",
            "optimizer_id": "SMAC20_CustomUncertainty_ei_proximity_bc",
            "container_id": "SMAC20_CustomUncertainty",
        },
        {
            "key": "shaker_entropy",
            "paradigm": "direct",
            "extractor_name": "shaker_entropy",
            "optimizer_id": "SMAC20_CustomUncertainty_ei_shaker_entropy",
            "container_id": "SMAC20_CustomUncertainty",
        },
        # 3. Decoupled Additive Epistemic BO
        {
            "key": "likelihood_credal",
            "paradigm": "additive",
            "extractor_name": "likelihood_credal",
            "optimizer_id": "CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal",
            "container_id": "CARPSDynamicRF",
        },
        {
            "key": "proximity_bc_additive",
            "paradigm": "additive",
            "extractor_name": "proximity_bc",
            "optimizer_id": "CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc",
            "container_id": "CARPSDynamicRF",
        },
        {
            "key": "shaker_entropy_additive",
            "paradigm": "additive",
            "extractor_name": "shaker_entropy",
            "optimizer_id": "CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy",
            "container_id": "CARPSDynamicRF",
        },
    ]


def generate_noisy_sweep_tasks(
    output_file: str = "results/sweep_noisy_ei_head_to_head/tasks.txt",
    runs_dir: str = "results/sweep_noisy_ei_head_to_head/runs",
    n_seeds: int = 30,
    trials: int = 50,
    suite: str = "all",
    paradigm: str = "all",
    beta_max: float = 1.0,
    warmup_ratio: float = 0.20,
) -> List[str]:
    """Generates all Hydra task command lines for the noisy EI head-to-head sweep."""
    tasks = get_noisy_tasks(suite=suite)
    approaches = get_approach_configs()

    if paradigm != "all":
        approaches = [a for a in approaches if a["paradigm"] == paradigm]

    seeds = list(range(1, n_seeds + 1))
    lines: List[str] = []

    # Ensure output and runs directory exist
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    os.makedirs(os.path.abspath(runs_dir), exist_ok=True)

    for task_arg in tasks:
        # task_arg is like +task/Noisy/hetgp=cfg_branin_2d
        task_name = task_arg.split("=")[-1]

        for approach in approaches:
            opt_id = approach["optimizer_id"]
            cont_id = approach["container_id"]

            for seed in seeds:
                telemetry_path = f"{runs_dir}/telemetry_{opt_id}_{task_name}_seed{seed}.json"

                if approach["paradigm"] == "baseline":
                    cmd = (
                        f"--config-dir carps_integration/configs "
                        f"+optimizer/smac20=hpo "
                        f"++optimizer.acq_func_name=ei "
                        f"{task_arg} task.optimization_resources.n_trials={trials} "
                        f"seed={seed} ++optimizer.telemetry_path={telemetry_path} "
                        f"optimizer_id={opt_id} optimizer_container_id={cont_id}"
                    )

                elif approach["paradigm"] == "direct":
                    extractor = approach["extractor_name"]
                    cmd = (
                        f"--config-dir carps_integration/configs "
                        f"+optimizer=smac20_custom_uncertainty "
                        f"++optimizer.acq_func_name=ei "
                        f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={extractor} "
                        f"{task_arg} task.optimization_resources.n_trials={trials} "
                        f"seed={seed} ++optimizer.telemetry_path={telemetry_path} "
                        f"optimizer_id={opt_id} optimizer_container_id={cont_id}"
                    )

                elif approach["paradigm"] == "additive":
                    extractor = approach["extractor_name"]
                    cmd = (
                        f"--config-dir carps_integration/configs "
                        f"+optimizer=dyrf_additive_epistemic_ei "
                        f"++optimizer.extractor_name={extractor} "
                        f"++optimizer.beta_max={beta_max} "
                        f"++optimizer.warmup_ratio={warmup_ratio} "
                        f"{task_arg} task.optimization_resources.n_trials={trials} "
                        f"seed={seed} ++optimizer.telemetry_path={telemetry_path} "
                        f"optimizer_id={opt_id} optimizer_container_id={cont_id}"
                    )
                else:
                    raise ValueError(f"Unknown paradigm: {approach['paradigm']}")

                lines.append(cmd)

    if output_file:
        with open(output_file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    return lines


def main():
    parser = argparse.ArgumentParser(description="Generate CARP-S tasks for Noisy Benchmark EI Head-to-Head Sweep")
    parser.add_argument("-o", "--output-file", type=str, default="results/sweep_noisy_ei_head_to_head/tasks.txt",
                        help="Path to output tasks.txt")
    parser.add_argument("-r", "--runs-dir", type=str, default="results/sweep_noisy_ei_head_to_head/runs",
                        help="Directory for telemetry JSON files")
    parser.add_argument("-s", "--seeds", type=int, default=30, help="Number of random seeds (default: 30)")
    parser.add_argument("-t", "--trials", type=int, default=50, help="Number of trials per run (default: 50)")
    parser.add_argument("--suite", type=str, default="all", choices=["all", "hetgp", "bbob"],
                        help="Filter benchmark suite")
    parser.add_argument("--paradigm", type=str, default="all", choices=["all", "baseline", "direct", "additive"],
                        help="Filter optimizer paradigm")
    parser.add_argument("--beta-max", type=float, default=1.0, help="Max additive beta parameter")
    parser.add_argument("--warmup-ratio", type=float, default=0.20, help="Warmup ratio for beta schedule")

    args = parser.parse_args()
    tasks = generate_noisy_sweep_tasks(
        output_file=args.output_file,
        runs_dir=args.runs_dir,
        n_seeds=args.seeds,
        trials=args.trials,
        suite=args.suite,
        paradigm=args.paradigm,
        beta_max=args.beta_max,
        warmup_ratio=args.warmup_ratio,
    )
    print(f"[✓] Generated {len(tasks)} tasks in '{args.output_file}'!")
    print(f"    Telemetry directory: '{args.runs_dir}'")


if __name__ == "__main__":
    main()
