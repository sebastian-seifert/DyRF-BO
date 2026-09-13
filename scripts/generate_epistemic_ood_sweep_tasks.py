#!/usr/bin/env python3
"""
Master Task Generator for Epistemic OOD Detection & Uncertainty Quantification Benchmark Sweep.

Generates exactly 2,040 tasks:
  51 Benchmark Functions (41 normal functions across 1D-15D + 10 special named functions)
  x 2 Gap Types ('empty', 'sparse')
  x 2 Approaches ('standard_disagreement', 'distance_evidential')
  x 10 Seeds (1..10)
= 2,040 Tasks.
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from synthetic_functions import (
    get_all_normal_functions,
    get_special_functions,
)

DEFAULT_OUTPUT_DIR = "results/epistemic_ood_sweep"
TASK_FILE = "results/epistemic_ood_sweep_tasks.txt"


def get_all_sweep_functions() -> dict:
    """Returns combined master dictionary of 51 functions (41 normal + 10 special)."""
    normal = get_all_normal_functions()
    special = get_special_functions()
    return {**normal, **special}


def generate_epistemic_ood_sweep_tasks(
    output_file: str = TASK_FILE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> list[str]:
    """
    Generates all 2,040 benchmark sweep task commands and writes them to output_file.
    
    Returns:
        List of command strings.
    """
    funcs = get_all_sweep_functions()
    function_names = list(funcs.keys())
    gap_types = ["empty", "sparse"]
    approaches = ["standard_disagreement", "distance_evidential"]
    seeds = list(range(1, 11))

    # Ensure parent output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for func in function_names:
        for gap in gap_types:
            for app in approaches:
                for seed in seeds:
                    cmd = (
                        f"python scripts/run_epistemic_ood_sweep_local.py "
                        f"--func_name={func} --gap_type={gap} --approach={app} "
                        f"--seed={seed} --output_dir={output_dir}"
                    )
                    tasks.append(cmd)

    with open(output_file, "w") as f:
        for t in tasks:
            f.write(t + "\n")

    print(
        f"Generated {len(tasks)} epistemic OOD sweep tasks in '{output_file}' "
        f"across {len(function_names)} functions."
    )
    return tasks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Epistemic OOD Benchmark Sweep Tasks")
    parser.add_argument("--output_file", type=str, default=TASK_FILE, help="Path to output tasks file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for task results")
    args = parser.parse_args()

    generate_epistemic_ood_sweep_tasks(output_file=args.output_file, output_dir=args.output_dir)

