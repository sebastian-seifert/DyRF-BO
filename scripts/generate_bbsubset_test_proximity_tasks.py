#!/usr/bin/env python3
"""Task Generator for CARP-S BBsubset Held-Out Test Evaluation Suite.

Generates 1,200 task command lines pairing:
1. Proposed: SMAC20_ProximityLCB_tuned (Tuned Proximity Lower Bound Acquisition:
   k=28, decay_lambda=0.20486, eps=0.080791, level=0.95, uncertainty_func=proximity_b)
2. Baseline: SMAC3_HPOFacade_lcb (Standard SMAC3 Random Forest with native LCB, kappa=1.96 / beta=3.8416)

Across:
- 20 held-out test tasks (8 BBOB + 1 HPOBench ML + 11 YAHPO Gym)
- 30 paired seeds (seeds 1 to 30)
- 100 trials budget per run (T=100)

Total: 20 tasks * 30 seeds * 2 optimizers = 1,200 runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence, Union

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


def generate_bbsubset_test_proximity_tasks(
    output_file: str | None = "results/sweep_bbsubset_test_proximity/tasks.txt",
    runs_dir: str = "results/sweep_bbsubset_test_proximity",
    baserundir: str = "runs/sweep_bbsubset_test_proximity",
    seeds: Union[int, Sequence[int]] = 30,
    trials: int = 100,
    k_neighbors: int = 28,
    level: float = 0.95,
    eps_floor: float = 0.080791,
    decay_lambda: float = 0.20486,
    uncertainty_func: str = "proximity_b",
    kappa_baseline: float = 1.96,
) -> List[str]:
    """Generates all 1,200 Hydra task command lines for the BBsubset test set evaluation."""
    tasks = CarpsBBSubsetRegistry.get_test_tasks()
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

    beta_baseline = round(float(kappa_baseline) ** 2, 4)  # 1.96^2 = 3.8416

    lines: List[str] = []

    # 1. Proposed Method: Tuned Proximity Lower Bound Acquisition
    for task_arg in tasks:
        task_name = task_arg.split("/")[-1]
        task_cmd = f"+{task_arg}" if not task_arg.startswith("+") else task_arg
        for seed in seeds_list:
            telemetry = f"{runs_dir}/telemetry_SMAC20_ProximityLCB_tuned_{task_name}_seed{seed}.json"
            cmd = (
                f"--config-dir carps_integration/configs "
                f"+optimizer=smac20_proximity_lcb "
                f"++optimizer.acq_func_kwargs.k={k_neighbors} "
                f"++optimizer.acq_func_kwargs.level={level} "
                f"++optimizer.acq_func_kwargs.eps={eps_floor} "
                f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={uncertainty_func} "
                f"++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda={decay_lambda} "
                f"{task_cmd} task.optimization_resources.n_trials={trials} "
                f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                f"baserundir={baserundir} "
                f"optimizer_id=SMAC20_ProximityLCB_tuned optimizer_container_id=SMAC20_ProximityLCB"
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
        description="Generate CARP-S BBsubset Held-Out Test Proximity LCB vs SMAC3 LCB sweep tasks."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/sweep_bbsubset_test_proximity/tasks.txt",
        help="Path to output tasks.txt file (default: results/sweep_bbsubset_test_proximity/tasks.txt)",
    )
    parser.add_argument(
        "--runs-dir",
        default="results/sweep_bbsubset_test_proximity",
        help="Directory to store telemetry and run outputs (default: results/sweep_bbsubset_test_proximity)",
    )
    parser.add_argument(
        "--baserundir",
        default="runs/sweep_bbsubset_test_proximity",
        help="Hydra baserundir (default: runs/sweep_bbsubset_test_proximity)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Number of seeds to run (default: 30)",
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
        default=28,
        help="k nearest neighbors for proximity UQ (default: 28)",
    )
    parser.add_argument(
        "--level",
        type=float,
        default=0.95,
        help="Quantile level for proximity LCB (default: 0.95)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.080791,
        help="Epsilon regularizer floor (default: 0.080791)",
    )
    parser.add_argument(
        "--decay-lambda",
        type=float,
        default=0.20486,
        help="Exponential distance decay factor (default: 0.20486)",
    )
    parser.add_argument(
        "--uncertainty-func",
        default="proximity_b",
        help="Uncertainty extractor function (default: proximity_b)",
    )
    parser.add_argument(
        "--kappa-baseline",
        type=float,
        default=1.96,
        help="SMAC3 native LCB kappa parameter (default: 1.96)",
    )

    args = parser.parse_args()

    lines = generate_bbsubset_test_proximity_tasks(
        output_file=args.output,
        runs_dir=args.runs_dir,
        baserundir=args.baserundir,
        seeds=args.seeds,
        trials=args.trials,
        k_neighbors=args.k,
        level=args.level,
        eps_floor=args.eps,
        decay_lambda=args.decay_lambda,
        uncertainty_func=args.uncertainty_func,
        kappa_baseline=args.kappa_baseline,
    )

    print(f"Successfully generated {len(lines)} tasks in {args.output}")


if __name__ == "__main__":
    main()
