#!/usr/bin/env python3
"""Task Generator for BBOB High-D (D=16) and Extreme (D=32) Proximity LCB Sweep.

Generates 8,640 Hydra task command lines pairing:
1. Proposed: SMAC20_ProximityLCB (Tuned Proximity Lower Bound Acquisition:
   k=25, decay_lambda=1.345, eps=0.16, level=0.95, uncertainty_func=proximity_b)
2. Baseline: SMAC3_HPOFacade_lcb (Standard SMAC3 Random Forest with native LCB, kappa=1.96 / beta=3.8416)

Across:
- 144 tasks (72 D=16 BBOB tasks + 72 D=32 BBOB tasks)
- 30 paired seeds (seeds 1 to 30)
- 100 trials budget per run (T=100)

Total: 144 tasks * 30 seeds * 2 optimizers = 8,640 runs.
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

from scripts.carps_bbob_highdim_registry import CarpsBBOBHighDimRegistry


def generate_bbob_highdim_proximity_tasks(
    output_file: str | None = "results/sweep_bbob_highdim_proximity/tasks.txt",
    runs_dir: str = "results/sweep_bbob_highdim_proximity",
    baserundir: str = "runs/sweep_bbob_highdim_proximity",
    dimensions: Sequence[int] = (16, 32),
    seeds: Union[int, Sequence[int]] = 30,
    trials: int = 100,
    k_neighbors: int = 25,
    level: float = 0.95,
    eps_floor: float = 0.16,
    decay_lambda: float = 1.345,
    uncertainty_func: str = "proximity_b",
    kappa_baseline: float = 1.96,
) -> List[str]:
    """Generates all Hydra task command lines for the BBOB High-D & Extreme evaluation."""
    tasks = CarpsBBOBHighDimRegistry.get_tasks_by_dimensions(dimensions)
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
            telemetry = f"{runs_dir}/telemetry_SMAC20_ProximityLCB_{task_name}_seed{seed}.json"
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
                f"optimizer_id=SMAC20_ProximityLCB optimizer_container_id=SMAC20_ProximityLCB"
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
                f.write(f"{line}\n")
        print(f"[SUCCESS] Wrote {len(lines)} tasks to {output_file}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BBOB High-D & Extreme Proximity LCB Sweep Tasks"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/sweep_bbob_highdim_proximity/tasks.txt",
        help="Target task file location",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="results/sweep_bbob_highdim_proximity",
        help="Directory for telemetry JSON files",
    )
    parser.add_argument(
        "--baserundir",
        type=str,
        default="runs/sweep_bbob_highdim_proximity",
        help="Base directory for CARP-S run outputs",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[16, 32],
        help="BBOB dimensions to evaluate (default: 16 32)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Number of paired seeds per task (default: 30)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Budget of function evaluations per run (default: 100)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=25,
        help="Top-k nearest neighbors for proximity acquisition (default: 25)",
    )
    parser.add_argument(
        "--level",
        type=float,
        default=0.95,
        help="Confidence level for lower bound (default: 0.95)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.16,
        help="Exploration floor multiplier eps (default: 0.16)",
    )
    parser.add_argument(
        "--decay-lambda",
        type=float,
        default=1.345,
        help="Topological tree-path decay lambda (default: 1.345)",
    )
    parser.add_argument(
        "--uncertainty-func",
        type=str,
        default="proximity_b",
        help="Surrogate uncertainty extractor (default: proximity_b)",
    )
    parser.add_argument(
        "--kappa-baseline",
        type=float,
        default=1.96,
        help="Exploration parameter kappa for baseline LCB (default: 1.96)",
    )
    args = parser.parse_args()

    generate_bbob_highdim_proximity_tasks(
        output_file=args.output,
        runs_dir=args.runs_dir,
        baserundir=args.baserundir,
        dimensions=args.dimensions,
        seeds=args.seeds,
        trials=args.trials,
        k_neighbors=args.k,
        level=args.level,
        eps_floor=args.eps,
        decay_lambda=args.decay_lambda,
        uncertainty_func=args.uncertainty_func,
        kappa_baseline=args.kappa_baseline,
    )


if __name__ == "__main__":
    main()
