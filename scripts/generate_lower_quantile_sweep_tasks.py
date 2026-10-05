#!/usr/bin/env python3
"""Task generator for Lower-Quantile UQ Sweep.

Generates reproducible task command lines across benchmark dimensions, sample sizes,
objective functions, sampling protocols, seeds, and surrogates.

Full sweep: 6 dimensions x 4 sample sizes x 4 functions x 2 strategies x 10 seeds x 5 surrogates = 9,600 runs.
Pilot sweep: 2 dimensions ([2, 16]) x 1 sample size (112) x 1 function ('sphere') x 2 strategies x 1 seed (0) x 5 surrogates = 20 runs.
Stress sweep: 16 balanced configurations x 5 surrogates = 80 runs.
Part 1 split: 5,000 runs.
Part 2 split: 4,600 runs.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_FILE = "results/lower_quantile_sweep/tasks.txt"
DEFAULT_OUTPUT_DIR = "results/lower_quantile_sweep/raw"
DEFAULT_SUMMARY_DIR = "results/lower_quantile_sweep/summaries"

DEFAULT_DIMENSIONS = [2, 3, 5, 8, 16, 32]
DEFAULT_N_TRAINS = [112, 224, 448, 896]
DEFAULT_FUNCTIONS = ["sphere", "rosenbrock", "rastrigin", "ackley"]
DEFAULT_STRATEGIES = ["natural", "stratified"]
DEFAULT_SEEDS = list(range(10))
DEFAULT_SURROGATES = ["smac_default", "mature", "shallow", "coarse", "breiman"]

PILOT_DIMENSIONS = [2, 16]
PILOT_N_TRAINS = [112]
PILOT_FUNCTIONS = ["sphere"]
PILOT_STRATEGIES = ["natural", "stratified"]
PILOT_SEEDS = [0]
PILOT_SURROGATES = ["smac_default", "mature", "shallow", "coarse", "breiman"]

STRESS_CONFIGURATIONS = [
    # 16-run balanced orthogonal design:
    # 4 dimensions (2, 5, 16, 32) x 4 functions x 2 sample sizes (112, 224) x 2 strategies
    (2, 112, "sphere", "natural", 0),
    (2, 112, "rosenbrock", "stratified", 0),
    (2, 224, "rastrigin", "natural", 0),
    (2, 224, "ackley", "stratified", 0),
    (5, 112, "rosenbrock", "natural", 0),
    (5, 112, "rastrigin", "stratified", 0),
    (5, 224, "ackley", "natural", 0),
    (5, 224, "sphere", "stratified", 0),
    (16, 112, "rastrigin", "natural", 0),
    (16, 112, "ackley", "stratified", 0),
    (16, 224, "sphere", "natural", 0),
    (16, 224, "rosenbrock", "stratified", 0),
    (32, 112, "ackley", "natural", 0),
    (32, 112, "sphere", "stratified", 0),
    (32, 224, "rosenbrock", "natural", 0),
    (32, 224, "rastrigin", "stratified", 0),
]


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for lower quantile sweep task generation."""
    parser = argparse.ArgumentParser(
        description="Generate task command lines for lower-quantile UQ sweep."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Target path for task file (default: {DEFAULT_OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help="Generate pilot suite: 2 D x 1 N x 1 func x 2 strats x 5 surrogates x 1 seed = 20 tasks.",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        default=False,
        help="Generate stress suite: 16 balanced configs x 5 surrogates = 80 tasks.",
    )
    parser.add_argument(
        "--mode",
        type=str.lower,
        choices=["pilot", "stress", "full"],
        default=None,
        help="Optional sweep mode identifier ('pilot', 'stress', 'full').",
    )
    parser.add_argument(
        "--split-parts",
        action="store_true",
        default=False,
        help="Split generated tasks into part 1 (5,000 tasks) and part 2 (4,600 tasks).",
    )
    parser.add_argument(
        "--output-part1",
        type=str,
        default=None,
        help="Target path for part 1 tasks file (default: <output-file-dir>/tasks_part1.txt).",
    )
    parser.add_argument(
        "--output-part2",
        type=str,
        default=None,
        help="Target path for part 2 tasks file (default: <output-file-dir>/tasks_part2.txt).",
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
        "--dimensions",
        type=int,
        nargs="+",
        default=DEFAULT_DIMENSIONS,
        help=f"List of dimensions (default: {DEFAULT_DIMENSIONS}).",
    )
    parser.add_argument(
        "--n-trains",
        type=int,
        nargs="+",
        default=DEFAULT_N_TRAINS,
        help=f"List of training sample sizes (default: {DEFAULT_N_TRAINS}).",
    )
    parser.add_argument(
        "--functions",
        type=str,
        nargs="+",
        default=DEFAULT_FUNCTIONS,
        help=f"List of benchmark functions (default: {DEFAULT_FUNCTIONS}).",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=DEFAULT_STRATEGIES,
        help=f"List of sampling strategies (default: {DEFAULT_STRATEGIES}).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"List of random seeds (default: {DEFAULT_SEEDS}).",
    )
    parser.add_argument(
        "--surrogates",
        type=str,
        nargs="+",
        choices=DEFAULT_SURROGATES,
        default=DEFAULT_SURROGATES,
        help=f"List of surrogate models (default: {DEFAULT_SURROGATES}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional hint for parallel worker count.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str.lower,
        default="all",
        choices=["all", "unweighted_proximity_only"],
        help="Evaluation mode: 'all' (default: full UQ suite) or 'unweighted_proximity_only'.",
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
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
    pilot: bool = False,
    stress: bool = False,
    mode: Optional[str] = None,
    split_parts: bool = False,
    output_part1: Optional[str | Path] = None,
    output_part2: Optional[str | Path] = None,
    dimensions: Optional[List[int]] = None,
    n_trains: Optional[List[int]] = None,
    functions: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    surrogates: Optional[List[str]] = None,
    output_dir: Optional[str | Path] = DEFAULT_OUTPUT_DIR,
    summary_dir: Optional[str | Path] = DEFAULT_SUMMARY_DIR,
    eval_mode: str = "all",
    python_bin: str = "python",
    skip_if_exists: bool = True,
    workers: Optional[int] = None,
) -> List[str]:
    """Generate task command lines for lower quantile UQ sweep.

    Parameters
    ----------
    output_file : str | Path, default=DEFAULT_OUTPUT_FILE
        Destination text file for task commands.
    pilot : bool, default=False
        If True, generates pilot suite (20 tasks).
    stress : bool, default=False
        If True, generates stress suite (80 tasks).
    mode : Optional[str], default=None
        Sweep mode identifier ('pilot', 'stress', 'full').
    split_parts : bool, default=False
        If True, splits tasks into part 1 (first 5,000) and part 2 (remaining 4,600).
    output_part1 : str | Path, optional
        Destination for part 1 tasks file.
    output_part2 : str | Path, optional
        Destination for part 2 tasks file.
    dimensions : list[int], optional
        Feature space dimensions.
    n_trains : list[int], optional
        Training set sizes.
    functions : list[str], optional
        Benchmark objective functions.
    strategies : list[str], optional
        Sampling strategies ('natural', 'stratified').
    seeds : list[int], optional
        Random seeds.
    surrogates : list[str], optional
        Surrogate types ('smac_default', 'mature', 'shallow', 'coarse', 'breiman').
    output_dir : str | Path, optional
        Target directory for raw parquet evaluations.
    summary_dir : str | Path, optional
        Target directory for summary JSON files.
    eval_mode : str, default='all'
        Evaluation mode ('all' or 'unweighted_proximity_only').
    python_bin : str, default='python'
        Python interpreter binary.
    skip_if_exists : bool, default=True
        Whether to pass --skip-if-exists to each task runner.
    workers : int, optional
        Unused parameter kept for CLI symmetry.

    Returns
    -------
    list[str]
        List of generated task command lines.
    """
    if mode is not None:
        mode_lower = mode.lower()
        if mode_lower == "pilot":
            pilot = True
        elif mode_lower == "stress":
            stress = True

    py_bin = shlex.quote(str(python_bin))
    out_dir_suffix = f" --output-dir {shlex.quote(str(output_dir))}" if output_dir else ""
    sum_dir_suffix = f" --summary-dir {shlex.quote(str(summary_dir))}" if summary_dir else ""
    eval_suffix = f" --eval-mode {shlex.quote(str(eval_mode))}" if eval_mode != "all" else ""
    skip_suffix = " --skip-if-exists" if skip_if_exists else ""
    tasks: List[str] = []

    if pilot:
        surr_list = surrogates if surrogates is not None else PILOT_SURROGATES
        for dim in PILOT_DIMENSIONS:
            for n_train in PILOT_N_TRAINS:
                for func in PILOT_FUNCTIONS:
                    for strat in PILOT_STRATEGIES:
                        for seed in PILOT_SEEDS:
                            for surr in surr_list:
                                cmd = (
                                    f"{py_bin} scripts/run_extrapolation_experiment.py "
                                    f"--dimension {dim} "
                                    f"--n-train {n_train} "
                                    f"--function {func} "
                                    f"--strategy {strat} "
                                    f"--seed {seed} "
                                    f"--surrogate {surr}"
                                    f"{eval_suffix}"
                                    f"{out_dir_suffix}"
                                    f"{sum_dir_suffix}"
                                    f"{skip_suffix}"
                                )
                                tasks.append(cmd)
    elif stress:
        surr_list = surrogates if surrogates is not None else DEFAULT_SURROGATES
        for dim, n_train, func, strat, seed in STRESS_CONFIGURATIONS:
            for surr in surr_list:
                cmd = (
                    f"{py_bin} scripts/run_extrapolation_experiment.py "
                    f"--dimension {dim} "
                    f"--n-train {n_train} "
                    f"--function {func} "
                    f"--strategy {strat} "
                    f"--seed {seed} "
                    f"--surrogate {surr}"
                    f"{eval_suffix}"
                    f"{out_dir_suffix}"
                    f"{sum_dir_suffix}"
                    f"{skip_suffix}"
                )
                tasks.append(cmd)
    else:
        dims = dimensions if dimensions is not None else DEFAULT_DIMENSIONS
        trains = n_trains if n_trains is not None else DEFAULT_N_TRAINS
        funcs = functions if functions is not None else DEFAULT_FUNCTIONS
        strats = strategies if strategies is not None else DEFAULT_STRATEGIES
        seed_list = seeds if seeds is not None else DEFAULT_SEEDS
        surr_list = surrogates if surrogates is not None else DEFAULT_SURROGATES

        for dim in dims:
            for n_train in trains:
                for func in funcs:
                    for strat in strats:
                        for seed in seed_list:
                            for surr in surr_list:
                                cmd = (
                                    f"{py_bin} scripts/run_extrapolation_experiment.py "
                                    f"--dimension {dim} "
                                    f"--n-train {n_train} "
                                    f"--function {func} "
                                    f"--strategy {strat} "
                                    f"--seed {seed} "
                                    f"--surrogate {surr}"
                                    f"{eval_suffix}"
                                    f"{out_dir_suffix}"
                                    f"{sum_dir_suffix}"
                                    f"{skip_suffix}"
                                )
                                tasks.append(cmd)

    target_path = Path(output_file)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        for cmd in tasks:
            f.write(cmd + "\n")

    if split_parts:
        split_idx = min(5000, len(tasks))
        part1_tasks = tasks[:split_idx]
        part2_tasks = tasks[split_idx:]

        p1_path = Path(output_part1) if output_part1 else target_path.parent / "tasks_part1.txt"
        p2_path = Path(output_part2) if output_part2 else target_path.parent / "tasks_part2.txt"

        p1_path.parent.mkdir(parents=True, exist_ok=True)
        p2_path.parent.mkdir(parents=True, exist_ok=True)

        with open(p1_path, "w", encoding="utf-8") as f:
            for cmd in part1_tasks:
                f.write(cmd + "\n")

        with open(p2_path, "w", encoding="utf-8") as f:
            for cmd in part2_tasks:
                f.write(cmd + "\n")

    return tasks


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    tasks = generate_tasks(
        output_file=args.output_file,
        pilot=args.pilot,
        stress=args.stress,
        mode=args.mode,
        split_parts=args.split_parts,
        output_part1=args.output_part1,
        output_part2=args.output_part2,
        dimensions=args.dimensions,
        n_trains=args.n_trains,
        functions=args.functions,
        strategies=args.strategies,
        seeds=args.seeds,
        surrogates=args.surrogates,
        output_dir=args.output_dir,
        summary_dir=args.summary_dir,
        eval_mode=args.eval_mode,
        python_bin=args.python_bin,
        skip_if_exists=args.skip_if_exists,
        workers=args.workers,
    )
    mode_desc = "pilot" if args.pilot else ("stress" if args.stress else (args.mode or "full"))
    print(f"Generated {len(tasks)} tasks in mode '{mode_desc}' -> '{args.output_file}'")
    if args.split_parts:
        p1 = args.output_part1 or Path(args.output_file).parent / "tasks_part1.txt"
        p2 = args.output_part2 or Path(args.output_file).parent / "tasks_part2.txt"
        print(f"  Split Part 1: {min(5000, len(tasks))} tasks -> '{p1}'")
        print(f"  Split Part 2: {max(0, len(tasks) - 5000)} tasks -> '{p2}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
