#!/usr/bin/env python3
"""Unified Task Generator for Full CARP-S HPOBench Tabular (ML + NAS) Benchmark Sweep.

Combines:
- 88 HPOBench Tabular ML tasks -> saved under runs/sweep_hpobench_ml & results/sweep_hpobench_ml
- 10 HPOBench Tabular NAS tasks -> saved under runs/sweep_hpobench_nas & results/sweep_hpobench_nas
Across:
- 11 evaluated optimizers (2 Baselines, 1 Entropy, 8 Proximity Family)
- 20 strictly paired seeds (seeds 1 to 20 by default)
- 100 trials budget per run (T=100)

Total: (88 + 10) tasks * 11 optimizers * 20 seeds = 21,560 runs (19,360 ML + 2,200 NAS).
Writes the unified manifest to results/sweep_hpobench_tabular/tasks.txt while keeping
all run and telemetry outputs cleanly isolated in their respective ML and NAS subfolders.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_realworld_registry import CarpsRealworldRegistry
from scripts.generate_bbob_full_sweep_tasks import ALL_OPTIMIZERS
from scripts.generate_hpobench_ml_sweep_tasks import (
    HPOBENCH_ML_OUTPUT_DIR,
    HPOBENCH_ML_RUNS_DIR,
    build_hpobench_ml_command_line,
)
from scripts.generate_hpobench_nas_sweep_tasks import (
    HPOBENCH_NAS_OUTPUT_DIR,
    HPOBENCH_NAS_RUNS_DIR,
    build_hpobench_nas_command_line,
)

HPOBENCH_TABULAR_OUTPUT_DIR = "results/sweep_hpobench_tabular"


def generate_hpobench_tabular_sweep_tasks(
    seeds: Union[int, Sequence[int]] = 20,
    trials: int = 100,
    output_dir: str = HPOBENCH_TABULAR_OUTPUT_DIR,
    ml_runs_dir: str = HPOBENCH_ML_OUTPUT_DIR,
    ml_baserundir: str = HPOBENCH_ML_RUNS_DIR,
    nas_runs_dir: str = HPOBENCH_NAS_OUTPUT_DIR,
    nas_baserundir: str = HPOBENCH_NAS_RUNS_DIR,
) -> List[str]:
    """Generates all Hydra task command lines for the combined HPOBench Tabular (ML + NAS) sweep."""
    if isinstance(seeds, int):
        seeds_list = list(range(1, seeds + 1))
    else:
        seeds_list = list(seeds)

    ml_tasks = CarpsRealworldRegistry.get_hpobench_ml_tasks()
    nas_tasks = CarpsRealworldRegistry.get_hpobench_nas_tasks()

    commands: List[str] = []

    # 1. HPOBench Tabular ML tasks (88 * 11 * N_seeds) -> runs/sweep_hpobench_ml & results/sweep_hpobench_ml
    for opt in ALL_OPTIMIZERS:
        for task_arg in ml_tasks:
            for seed in seeds_list:
                cmd = build_hpobench_ml_command_line(
                    opt_info=opt,
                    task_arg=task_arg,
                    seed=seed,
                    trials=trials,
                    runs_dir=ml_runs_dir,
                    baserundir=ml_baserundir,
                )
                commands.append(cmd)

    # 2. HPOBench Tabular NAS tasks (10 * 11 * N_seeds) -> runs/sweep_hpobench_nas & results/sweep_hpobench_nas
    for opt in ALL_OPTIMIZERS:
        for task_arg in nas_tasks:
            for seed in seeds_list:
                cmd = build_hpobench_nas_command_line(
                    opt_info=opt,
                    task_arg=task_arg,
                    seed=seed,
                    trials=trials,
                    runs_dir=nas_runs_dir,
                    baserundir=nas_baserundir,
                )
                commands.append(cmd)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ml_runs_dir, exist_ok=True)
    os.makedirs(nas_runs_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "tasks.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for c in commands:
            f.write(c + "\n")

    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate unified task manifest for full HPOBench Tabular (ML + NAS) CARP-S sweep."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
        help="Number of sequential seeds (default: 20 -> seeds 1..20).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Optimization evaluation budget per run (default: 100).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=HPOBENCH_TABULAR_OUTPUT_DIR,
        help="Directory to save combined tasks.txt manifest.",
    )
    parser.add_argument(
        "--ml-baserundir",
        type=str,
        default=HPOBENCH_ML_RUNS_DIR,
        help="Base rundir passed to Hydra for HPOBench ML CARP-S logs.",
    )
    parser.add_argument(
        "--nas-baserundir",
        type=str,
        default=HPOBENCH_NAS_RUNS_DIR,
        help="Base rundir passed to Hydra for HPOBench NAS CARP-S logs.",
    )

    args = parser.parse_args()
    tasks = generate_hpobench_tabular_sweep_tasks(
        seeds=args.seeds,
        trials=args.trials,
        output_dir=args.output_dir,
        ml_baserundir=args.ml_baserundir,
        nas_baserundir=args.nas_baserundir,
    )
    print(f"Generated {len(tasks)} combined HPOBench Tabular tasks to {args.output_dir}/tasks.txt")


if __name__ == "__main__":
    main()
