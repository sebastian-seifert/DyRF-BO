#!/usr/bin/env python3
"""Task generator for Dedicated Unweighted Proximity Sweep (Milestone 3).

Generates reproducible command-line tasks for unweighted proximity evaluation
across dimensions, training set sizes, benchmark functions, and sampling strategies.

Grid configurations:
- 'pilot' (8 tasks): D in [2, 16], N=112, functions ['sphere', 'ackley'],
  strategies ['natural', 'stratified'], seed 0.
- 'comparison' (32 tasks): D in [2, 5, 16, 32], N in [112, 224],
  functions ['sphere', 'ackley', 'rastrigin', 'rosenbrock'],
  strategies ['natural', 'stratified'], seed 0 (balanced orthogonal design).
- 'full' (64 tasks): Full orthogonal factorial sweep across 4 dimensions x
  2 sample sizes x 4 functions x 2 strategies x 1 seed.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_FILE = "results/unweighted_proximity_sweep/tasks.txt"
DEFAULT_OUTPUT_DIR = "results/unweighted_proximity_sweep/raw"
DEFAULT_SUMMARY_DIR = "results/unweighted_proximity_sweep/summaries"

PILOT_DIMENSIONS = [2, 16]
PILOT_N_TRAINS = [112]
PILOT_FUNCTIONS = ["sphere", "ackley"]
PILOT_STRATEGIES = ["natural", "stratified"]
PILOT_SEEDS = [0]

SWEEP_DIMENSIONS = [2, 5, 16, 32]
SWEEP_N_TRAINS = [112, 224]
SWEEP_FUNCTIONS = ["sphere", "ackley", "rastrigin", "rosenbrock"]
SWEEP_STRATEGIES = ["natural", "stratified"]
SWEEP_SEEDS = [0]


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for unweighted proximity sweep task generation."""
    parser = argparse.ArgumentParser(
        description="Generate task command lines for unweighted proximity sweep."
    )
    parser.add_argument(
        "--mode",
        type=str.lower,
        choices=["pilot", "comparison", "full"],
        default="full",
        help="Sweep mode: 'pilot' (8 tasks), 'comparison' (32 tasks), or 'full' (64 tasks) (default: full).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Target path for task file (default: {DEFAULT_OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Target directory for raw parquet evaluations (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default=DEFAULT_SUMMARY_DIR,
        help=f"Target directory for summary JSON files (default: {DEFAULT_SUMMARY_DIR}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional hint for parallel worker count.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="python",
        help="Python interpreter executable for task commands (default: 'python').",
    )
    parser.add_argument(
        "--skip-if-exists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to append --skip-if-exists to commands (default: True).",
    )
    return parser


def generate_tasks(
    mode: str = "full",
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    summary_dir: str | Path = DEFAULT_SUMMARY_DIR,
    python_bin: str = "python",
    skip_if_exists: bool = True,
    workers: Optional[int] = None,
) -> List[str]:
    """Generate task command lines according to specified sweep mode.

    Parameters
    ----------
    mode : {'pilot', 'comparison', 'full'}, default='full'
        Sweep mode selecting grid configuration.
    output_file : str | Path, default=DEFAULT_OUTPUT_FILE
        Destination text file for task commands.
    output_dir : str | Path, default=DEFAULT_OUTPUT_DIR
        Directory where tasks write raw evaluations.
    summary_dir : str | Path, default=DEFAULT_SUMMARY_DIR
        Directory where tasks write summary JSONs.
    python_bin : str, default='python'
        Python executable binary.
    skip_if_exists : bool, default=True
        Whether to pass --skip-if-exists to each task runner.
    workers : int, optional
        Unused parameter kept for CLI symmetry.

    Returns
    -------
    list[str]
        List of executable task strings.
    """
    mode = mode.lower()
    tasks: List[str] = []

    py_bin = shlex.quote(str(python_bin))
    out_dir_str = shlex.quote(str(output_dir))
    sum_dir_str = shlex.quote(str(summary_dir))
    skip_flag = " --skip-if-exists" if skip_if_exists else ""

    if mode == "pilot":
        # 8 tasks: 2 D x 1 N x 2 funcs x 2 strats x 1 seed
        for dim in PILOT_DIMENSIONS:
            for n_train in PILOT_N_TRAINS:
                for func in PILOT_FUNCTIONS:
                    for strat in PILOT_STRATEGIES:
                        for seed in PILOT_SEEDS:
                            cmd = (
                                f"{py_bin} scripts/run_extrapolation_experiment.py "
                                f"--dimension {dim} "
                                f"--n-train {n_train} "
                                f"--function {func} "
                                f"--strategy {strat} "
                                f"--seed {seed} "
                                f"--output-dir {out_dir_str} "
                                f"--summary-dir {sum_dir_str}"
                                f"{skip_flag}"
                            )
                            tasks.append(cmd)

    elif mode == "comparison":
        # 32 tasks: Balanced orthogonal design over 4 dims x 4 funcs x 2 strats,
        # with n_train balanced between 112 and 224.
        for d_idx, dim in enumerate(SWEEP_DIMENSIONS):
            for f_idx, func in enumerate(SWEEP_FUNCTIONS):
                for s_idx, strat in enumerate(SWEEP_STRATEGIES):
                    n_train = SWEEP_N_TRAINS[(d_idx + f_idx + s_idx) % 2]
                    cmd = (
                        f"{py_bin} scripts/run_extrapolation_experiment.py "
                        f"--dimension {dim} "
                        f"--n-train {n_train} "
                        f"--function {func} "
                        f"--strategy {strat} "
                        f"--seed {SWEEP_SEEDS[0]} "
                        f"--output-dir {out_dir_str} "
                        f"--summary-dir {sum_dir_str}"
                        f"{skip_flag}"
                    )
                    tasks.append(cmd)

    elif mode == "full":
        # 64 tasks: Complete orthogonal sweep over 4 dims x 2 Ns x 4 funcs x 2 strats x 1 seed
        for dim in SWEEP_DIMENSIONS:
            for n_train in SWEEP_N_TRAINS:
                for func in SWEEP_FUNCTIONS:
                    for strat in SWEEP_STRATEGIES:
                        for seed in SWEEP_SEEDS:
                            cmd = (
                                f"{py_bin} scripts/run_extrapolation_experiment.py "
                                f"--dimension {dim} "
                                f"--n-train {n_train} "
                                f"--function {func} "
                                f"--strategy {strat} "
                                f"--seed {seed} "
                                f"--output-dir {out_dir_str} "
                                f"--summary-dir {sum_dir_str}"
                                f"{skip_flag}"
                            )
                            tasks.append(cmd)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of 'pilot', 'comparison', 'full'.")

    # Write tasks to target output file
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(t + "\n")

    return tasks


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    tasks = generate_tasks(
        mode=args.mode,
        output_file=args.output_file,
        output_dir=args.output_dir,
        summary_dir=args.summary_dir,
        python_bin=args.python_bin,
        skip_if_exists=args.skip_if_exists,
        workers=args.workers,
    )
    print(f"Generated {len(tasks)} tasks in mode '{args.mode}' -> '{args.output_file}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
