#!/usr/bin/env python3
"""Task Generator for BBOB Dimensionality Timing Sweep.

Generates interleaved CLI task commands across dimensions d in {2, 4, 8, 16, 32}
for BBOB Function 8 (Rosenbrock, instance 0) evaluating 11 optimizers over 20 seeds
with deterministic shuffling to prevent temporal cluster bias.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Sequence, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_bbob_full_sweep_tasks import (
    ALL_OPTIMIZERS,
    build_bbob_command_line,
)

TIMING_SWEEP_OUTPUT_DIR = "results/sweep_bbob_timing"
TIMING_SWEEP_RUNS_DIR = "runs/sweep_bbob_timing"
TIMING_DIMENSIONS = (2, 4, 8, 16, 32)
TIMING_FUNCTION_ID = 8
TIMING_INSTANCE_ID = 0


def generate_bbob_timing_sweep_tasks(
    seeds: Union[int, Sequence[int]] = 20,
    trials: int = 100,
    dimensions: Sequence[int] = TIMING_DIMENSIONS,
    function_id: int = TIMING_FUNCTION_ID,
    instance_id: int = TIMING_INSTANCE_ID,
    output_dir: str = TIMING_SWEEP_OUTPUT_DIR,
    runs_dir: str = TIMING_SWEEP_OUTPUT_DIR,
    baserundir: str = TIMING_SWEEP_RUNS_DIR,
    shuffle: bool = True,
    shuffle_seed: int = 42,
) -> List[str]:
    """Generates task command lines for the BBOB dimensionality timing sweep."""
    if isinstance(seeds, int):
        seeds_list = list(range(1, seeds + 1))
    else:
        seeds_list = list(seeds)

    all_commands: List[str] = []

    for dim in dimensions:
        task_arg = f"+task=BBOB/cfg_{dim}_{function_id}_{instance_id}"
        for opt_info in ALL_OPTIMIZERS:
            for seed in seeds_list:
                cmd = build_bbob_command_line(
                    opt_info=opt_info,
                    task_arg=task_arg,
                    seed=seed,
                    trials=trials,
                    runs_dir=runs_dir,
                    baserundir=baserundir,
                )
                all_commands.append(cmd)

    if shuffle:
        rng = random.Random(shuffle_seed)
        rng.shuffle(all_commands)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        manifest_file = os.path.join(output_dir, "tasks.txt")
        with open(manifest_file, "w", encoding="utf-8") as f:
            for cmd in all_commands:
                f.write(f"{cmd}\n")

    return all_commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BBOB Dimensionality Timing Sweep tasks."
    )
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds (default: 20)")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per run (default: 100)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=TIMING_SWEEP_OUTPUT_DIR,
        help=f"Target output directory (default: {TIMING_SWEEP_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--baserundir",
        type=str,
        default=TIMING_SWEEP_RUNS_DIR,
        help=f"Base run output directory (default: {TIMING_SWEEP_RUNS_DIR})",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Random seed for deterministic interleaving/shuffling (default: 42)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable interleaving/shuffling across tasks",
    )

    args = parser.parse_args()
    tasks = generate_bbob_timing_sweep_tasks(
        seeds=args.seeds,
        trials=args.trials,
        output_dir=args.output_dir,
        runs_dir=args.output_dir,
        baserundir=args.baserundir,
        shuffle=not args.no_shuffle,
        shuffle_seed=args.shuffle_seed,
    )
    print(f"Successfully generated {len(tasks)} timing sweep tasks in '{args.output_dir}/tasks.txt'")


if __name__ == "__main__":
    main()
