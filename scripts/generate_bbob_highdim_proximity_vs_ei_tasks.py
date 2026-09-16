#!/usr/bin/env python3
"""Task Generator for BBOB High-D (D=16) and Extreme (D=32) Proximity LCB vs. SMAC3 EI Sweep.

Generates 8,640 Hydra task command lines pairing:
1. Proposed: SMAC20_ProximityLCB (Tuned Proximity Lower Bound Acquisition:
   k=25, decay_lambda=1.345, eps=0.16, level=0.95, uncertainty_func=proximity_b)
2. Baseline: SMAC3_HPOFacade_ei (Standard SMAC3 Random Forest with native Expected Improvement)

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


def generate_bbob_highdim_proximity_vs_ei_tasks(
    output_file: str | None = "results/sweep_bbob_highdim_proximity_vs_ei/tasks.txt",
    runs_dir: str = "results/sweep_bbob_highdim_proximity_vs_ei",
    baserundir: str = "runs/sweep_bbob_highdim_proximity_vs_ei",
    dimensions: Sequence[int] = (16, 32),
    seeds: Union[int, Sequence[int]] = 30,
    trials: int = 100,
    k_neighbors: int = 25,
    level: float = 0.95,
    eps_floor: float = 0.16,
    decay_lambda: float = 1.345,
    uncertainty_func: str = "proximity_b",
) -> List[str]:
    """Generates all 8,640 Hydra task command lines for BBOB High-D & Extreme evaluation vs SMAC3 EI."""
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

    # 2. Baseline Method: Standard SMAC3 HPOFacade with Expected Improvement (EI)
    for task_arg in tasks:
        task_name = task_arg.split("/")[-1]
        task_cmd = f"+{task_arg}" if not task_arg.startswith("+") else task_arg
        for seed in seeds_list:
            telemetry = f"{runs_dir}/telemetry_SMAC3_HPOFacade_ei_{task_name}_seed{seed}.json"
            cmd = (
                f"--config-dir carps_integration/configs "
                f"+optimizer=smac3_hpo_facade_ei "
                f"{task_cmd} task.optimization_resources.n_trials={trials} "
                f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                f"baserundir={baserundir} "
                f"optimizer_id=SMAC3_HPOFacade_ei optimizer_container_id=SMAC3_HPOFacade_ei"
            )
            lines.append(cmd)

    if output_file:
        with open(output_file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 8,640 BBOB High-D & Extreme tasks comparing Proximity LCB vs SMAC3 EI."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/sweep_bbob_highdim_proximity_vs_ei/tasks.txt",
        help="Target output path for the generated command lines.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="results/sweep_bbob_highdim_proximity_vs_ei",
        help="Directory to write telemetry output files.",
    )
    parser.add_argument(
        "--baserundir",
        type=str,
        default="runs/sweep_bbob_highdim_proximity_vs_ei",
        help="Base run directory for CARP-S raw execution artifacts.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[16, 32],
        help="Dimensionalities to include (default: 16 32).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Number of random seeds per task/optimizer pair (default: 30).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Optimization evaluation budget per run (default: 100).",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=25,
        help="Number of nearest leaf neighbors for Proximity LCB (default: 25).",
    )
    parser.add_argument(
        "--level",
        type=float,
        default=0.95,
        help="Empirical residual quantile confidence level (default: 0.95).",
    )
    parser.add_argument(
        "--eps-floor",
        type=float,
        default=0.16,
        help="Exploration floor epsilon for Proximity LCB (default: 0.16).",
    )
    parser.add_argument(
        "--decay-lambda",
        type=float,
        default=1.345,
        help="Topological tree-path Laplacian decay rate lambda (default: 1.345).",
    )
    parser.add_argument(
        "--uncertainty-func",
        type=str,
        default="proximity_b",
        help="Surrogate model uncertainty extractor (default: proximity_b).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = generate_bbob_highdim_proximity_vs_ei_tasks(
        output_file=args.output_file,
        runs_dir=args.runs_dir,
        baserundir=args.baserundir,
        dimensions=args.dimensions,
        seeds=args.seeds,
        trials=args.trials,
        k_neighbors=args.k_neighbors,
        level=args.level,
        eps_floor=args.eps_floor,
        decay_lambda=args.decay_lambda,
        uncertainty_func=args.uncertainty_func,
    )
    print(f"Generated {len(lines)} tasks in {args.output_file}")


if __name__ == "__main__":
    main()
