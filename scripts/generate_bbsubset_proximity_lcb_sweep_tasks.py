#!/usr/bin/env python3
"""Task Generator for CARP-S BBsubset Proximity LCB Benchmark Sweep.

Generates 360 task command lines pairing:
1. Proposed: SMAC20_ProximityLCB_k10 (Proximity Lower Bound Acquisition, lambda=1.0, k=10, alpha=0.05, eps=0.10)
2. Baseline: SMAC3_HPOFacade_lcb (Standard SMAC3 Random Forest with native LCB, kappa=1.96 / beta=3.8416)

Across:
- 18 working BBsubset dev tasks (all 4 BBOB + 3 HPOBench ML + 11 YAHPO Gym; excluding the 2 broken NAS benchmarks)
- 10 paired seeds (seeds 1 to 10)
- 100 trials budget per run (T=100)

Total: 18 tasks * 10 seeds * 2 optimizers = 360 runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence, Union

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


def generate_bbsubset_proximity_lcb_tasks(
    output_file: str = "results/sweep_bbsubset_proximity_lcb/tasks.txt",
    runs_dir: str = "results/bbsubset_proximity_lcb",
    baserundir: str = "runs/bbsubset_proximity_lcb",
    seeds: Union[int, Sequence[int]] = 10,
    trials: int = 100,
    k_neighbors: int = 10,
    alpha: float = 0.05,
    eps_floor: float = 0.10,
    decay_lambda: float = 1.0,
    kappa_baseline: float = 1.96,
) -> List[str]:
    """Generates all 360 Hydra task command lines for the Proximity LCB vs SMAC3 LCB sweep."""
    tasks = CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)
    if isinstance(seeds, int):
        seeds_list = list(range(1, seeds + 1))
    else:
        seeds_list = [int(s) for s in seeds]

    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    if runs_dir:
        os.makedirs(os.path.abspath(runs_dir), exist_ok=True)
    if baserundir:
        os.makedirs(os.path.abspath(baserundir), exist_ok=True)

    level = 1.0 - alpha
    beta_baseline = float(kappa_baseline) ** 2  # 1.96^2 = 3.8416

    lines: List[str] = []

    # 1. Proposed Method: Proximity Lower Bound Acquisition
    for task_arg in tasks:
        task_name = task_arg.split("/")[-1]
        task_cmd = f"+{task_arg}" if not task_arg.startswith("+") else task_arg
        for seed in seeds_list:
            telemetry = f"{runs_dir}/telemetry_SMAC20_ProximityLCB_k10_{task_name}_seed{seed}.json"
            cmd = (
                f"--config-dir carps_integration/configs "
                f"+optimizer=smac20_proximity_lcb "
                f"++optimizer.acq_func_kwargs.k={k_neighbors} "
                f"++optimizer.acq_func_kwargs.level={level} "
                f"++optimizer.acq_func_kwargs.eps={eps_floor} "
                f"++optimizer.smac_cfg.model_kwargs.uncertainty_func=proximity_b "
                f"++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda={decay_lambda} "
                f"{task_cmd} task.optimization_resources.n_trials={trials} "
                f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                f"baserundir={baserundir} "
                f"optimizer_id=SMAC20_ProximityLCB_k10 optimizer_container_id=SMAC20_ProximityLCB"
            )
            lines.append(cmd)

    # 2. Baseline Method: Standard SMAC3 HPOFacade LCB (kappa=1.96)
    for task_arg in tasks:
        task_name = task_arg.split("/")[-1]
        task_cmd = f"+{task_arg}" if not task_arg.startswith("+") else task_arg
        for seed in seeds_list:
            telemetry = f"{runs_dir}/telemetry_SMAC3_HPOFacade_lcb_{task_name}_seed{seed}.json"
            cmd = (
                f"--config-dir carps_integration/configs "
                f"+optimizer/smac20=hpo "
                f"++optimizer.acq_func_name=lcb "
                f"++optimizer.acq_func_kwargs.beta={beta_baseline} "
                f"++optimizer.acq_func_kwargs.update_beta=false "
                f"{task_cmd} task.optimization_resources.n_trials={trials} "
                f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                f"baserundir={baserundir} "
                f"optimizer_id=SMAC3_HPOFacade_lcb optimizer_container_id=SMAC3_HPOFacade"
            )
            lines.append(cmd)

    if output_file:
        with open(output_file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CARP-S BBsubset Proximity LCB vs SMAC3 LCB sweep tasks."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/sweep_bbsubset_proximity_lcb/tasks.txt",
        help="Path to output tasks.txt file (default: results/sweep_bbsubset_proximity_lcb/tasks.txt)",
    )
    parser.add_argument(
        "--runs-dir",
        default="results/bbsubset_proximity_lcb",
        help="Directory to store telemetry and run outputs (default: results/bbsubset_proximity_lcb)",
    )
    parser.add_argument(
        "--baserundir",
        default="runs/bbsubset_proximity_lcb",
        help="Hydra baserundir (default: runs/bbsubset_proximity_lcb)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of seeds to run (default: 10)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of optimization trials per run (default: 100)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="k nearest neighbors for proximity UQ (default: 10)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha miscoverage level (default: 0.05 -> level=0.95)",
    )
    args = parser.parse_args()

    lines = generate_bbsubset_proximity_lcb_tasks(
        output_file=args.output,
        runs_dir=args.runs_dir,
        baserundir=args.baserundir,
        seeds=args.seeds,
        trials=args.trials,
        k_neighbors=args.k,
        alpha=args.alpha,
    )
    print(f"Generated {len(lines)} tasks in {args.output}")


if __name__ == "__main__":
    main()
