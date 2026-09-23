#!/usr/bin/env python3
"""Task generator for Extrapolation Uncertainty Quantification Sweep.

Generates reproducible task command lines across benchmark dimensions, sample sizes,
objective functions, sampling protocols, and seeds.

Full grid: 6 dimensions x 4 sample sizes x 4 functions x 2 strategies x 10 seeds = 1,920 runs.
Pilot grid: 2 dimensions ([2, 16]) x 1 sample size (112) x 1 function ('sphere') x 2 strategies x 1 seed (0) = 4 runs.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DIMENSIONS = [2, 3, 5, 8, 16, 32]
DEFAULT_N_TRAINS = [112, 224, 448, 896]
DEFAULT_FUNCTIONS = ["sphere", "rosenbrock", "rastrigin", "ackley"]
DEFAULT_STRATEGIES = ["natural", "stratified"]
DEFAULT_SEEDS = list(range(10))

PILOT_DIMENSIONS = [2, 16]
PILOT_N_TRAINS = [112]
PILOT_FUNCTIONS = ["sphere"]
PILOT_STRATEGIES = ["natural", "stratified"]
PILOT_SEEDS = [0]


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for sweep task generation."""
    parser = argparse.ArgumentParser(
        description="Generate command-line task list for extrapolation UQ experiments."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/extrapolation_sweep_tasks.txt",
        help="Target output file path for generated tasks (default: results/extrapolation_sweep_tasks.txt).",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help="Generate 4-run pilot suite: D in [2, 16], N=112, sphere, seed=0, natural & stratified.",
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
        help="List of random seeds (default: 0..9).",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="python",
        help="Python binary path or command for task execution (default: 'python').",
    )
    parser.add_argument(
        "--skip-if-exists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append --skip-if-exists flag to generated task commands (default: True).",
    )
    return parser


def generate_tasks(
    output_file: str | Path = "results/extrapolation_sweep_tasks.txt",
    pilot: bool = False,
    dimensions: Optional[List[int]] = None,
    n_trains: Optional[List[int]] = None,
    functions: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    python_bin: str = "python",
    skip_if_exists: bool = True,
) -> List[str]:
    """Generate experiment task commands and serialize to target file.

    Parameters
    ----------
    output_file : str | Path
        Path to output task text file.
    pilot : bool, default=False
        If True, overrides sweep parameters with the 4-run pilot suite.
    dimensions : list[int], optional
        Feature space dimensions.
    n_trains : list[int], optional
        Training set sizes.
    functions : list[str], optional
        Benchmark objective function names.
    strategies : list[str], optional
        Sampling strategies ('natural' or 'stratified').
    seeds : list[int], optional
        Random seeds.
    python_bin : str, default='python'
        Python interpreter binary for command lines.
    skip_if_exists : bool, default=True
        Whether to append --skip-if-exists to commands.

    Returns
    -------
    list[str]
        List of generated task command lines.
    """
    if pilot:
        dims = PILOT_DIMENSIONS
        trains = PILOT_N_TRAINS
        funcs = PILOT_FUNCTIONS
        strats = PILOT_STRATEGIES
        seed_list = PILOT_SEEDS
    else:
        dims = dimensions if dimensions is not None else DEFAULT_DIMENSIONS
        trains = n_trains if n_trains is not None else DEFAULT_N_TRAINS
        funcs = functions if functions is not None else DEFAULT_FUNCTIONS
        strats = strategies if strategies is not None else DEFAULT_STRATEGIES
        seed_list = seeds if seeds is not None else DEFAULT_SEEDS

    skip_suffix = " --skip-if-exists" if skip_if_exists else ""
    py_bin = shlex.quote(python_bin)
    tasks: List[str] = []

    for dim in dims:
        for n_train in trains:
            for func in funcs:
                for strat in strats:
                    for seed in seed_list:
                        cmd = (
                            f"{py_bin} scripts/run_extrapolation_experiment.py "
                            f"--dimension {dim} "
                            f"--n-train {n_train} "
                            f"--function {func} "
                            f"--strategy {strat} "
                            f"--seed {seed}"
                            f"{skip_suffix}"
                        )
                        tasks.append(cmd)

    target_path = Path(output_file)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        for cmd in tasks:
            f.write(cmd + "\n")

    print(f"Generated {len(tasks)} tasks in '{target_path}'.")
    return tasks


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    generate_tasks(
        output_file=args.output_file,
        pilot=args.pilot,
        dimensions=args.dimensions,
        n_trains=args.n_trains,
        functions=args.functions,
        strategies=args.strategies,
        seeds=args.seeds,
        python_bin=args.python_bin,
        skip_if_exists=args.skip_if_exists,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
