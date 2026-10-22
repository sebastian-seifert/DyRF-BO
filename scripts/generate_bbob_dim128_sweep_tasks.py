#!/usr/bin/env python3
"""Task Generator for Dedicated CARP-S BBOB Dimension 128 Benchmark Sweep.

Generates CLI task strings for strictly 5 optimizers on 128D BBOB tasks:
1. SMAC3_HPOFacade_lcb (Standard Gaussian LCB)
2. SMAC3_HPOFacade_ei (Standard Expected Improvement)
3. SMAC20_ProximityAC_LCB (Proximity AC, OOB)
4. SMAC20_ProximityBC_LCB (Proximity BC, OOB)
5. SMAC20_Entropy_LCB (Entropy-based Uncertainty, LCB)

Across:
- 72 CARP-S BBOB Tasks (Dim 128 * 24 functions * 3 instances)
- 20 strictly paired seeds (seeds 1 to 20 by default)
- 100 trials budget per run (T=100)

Total: 72 tasks * 5 optimizers * 20 seeds = 7,200 runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_bbob_full_registry import CarpsBBOBFullRegistry
from scripts.generate_bbob_full_sweep_tasks import build_bbob_command_line

BBOB_DIM128_OUTPUT_DIR = "results/sweep_bbob_dim128"
BBOB_DIM128_RUNS_DIR = "runs/sweep_bbob_dim128"

DIM128_OPTIMIZERS: List[Dict[str, str]] = [
    {
        "id": "SMAC3_HPOFacade_lcb",
        "container_id": "SMAC3_HPOFacade_lcb",
        "config_arg": "+optimizer=smac3_hpo_facade_lcb",
    },
    {
        "id": "SMAC3_HPOFacade_ei",
        "container_id": "SMAC3_HPOFacade_ei",
        "config_arg": "+optimizer=smac3_hpo_facade_ei",
    },
    {
        "id": "SMAC20_ProximityAC_LCB",
        "container_id": "SMAC20_ProximityAC_LCB",
        "config_arg": "+optimizer=smac20_proximity_ac_lcb",
    },
    {
        "id": "SMAC20_ProximityBC_LCB",
        "container_id": "SMAC20_ProximityBC_LCB",
        "config_arg": "+optimizer=smac20_proximity_bc_lcb",
    },
    {
        "id": "SMAC20_Entropy_LCB",
        "container_id": "SMAC20_Entropy_LCB",
        "config_arg": "+optimizer=smac20_entropy_lcb",
    },
]


def generate_bbob_dim128_sweep_tasks(
    seeds: Union[int, Sequence[int]] = 20,
    trials: int = 100,
    output_dir: str = BBOB_DIM128_OUTPUT_DIR,
    runs_dir: str = BBOB_DIM128_OUTPUT_DIR,
    baserundir: str = BBOB_DIM128_RUNS_DIR,
) -> List[str]:
    """Generates task command lines strictly for dimension 128 BBOB benchmarks."""
    if isinstance(seeds, int):
        seeds_list = list(range(1, seeds + 1))
    else:
        seeds_list = list(seeds)

    dim128_tasks = CarpsBBOBFullRegistry.get_tasks_by_dimension(128)

    commands: List[str] = []
    for seed in seeds_list:
        for task_arg in dim128_tasks:
            for opt_info in DIM128_OPTIMIZERS:
                cmd = build_bbob_command_line(
                    opt_info=opt_info,
                    task_arg=task_arg,
                    seed=seed,
                    trials=trials,
                    runs_dir=runs_dir,
                    baserundir=baserundir,
                )
                commands.append(cmd)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tasks_all_path = os.path.join(output_dir, "tasks_all.txt")
        tasks_txt_path = os.path.join(output_dir, "tasks.txt")

        with open(tasks_all_path, "w", encoding="utf-8") as f_all:
            for cmd in commands:
                f_all.write(f"{cmd}\n")

        with open(tasks_txt_path, "w", encoding="utf-8") as f_txt:
            for cmd in commands:
                f_txt.write(f"{cmd}\n")

        # Partition tasks into 2 halves (max 5,000 jobs per array chunk)
        midpoint = len(commands) // 2
        p1_cmds = commands[:midpoint]
        p2_cmds = commands[midpoint:]

        tasks_p1_path = os.path.join(output_dir, "tasks_part1.txt")
        tasks_p2_path = os.path.join(output_dir, "tasks_part2.txt")

        with open(tasks_p1_path, "w", encoding="utf-8") as f_p1:
            for cmd in p1_cmds:
                f_p1.write(f"{cmd}\n")

        with open(tasks_p2_path, "w", encoding="utf-8") as f_p2:
            for cmd in p2_cmds:
                f_p2.write(f"{cmd}\n")

        print(f"Generated {len(commands)} tasks in {output_dir}")
        print(f"  - {tasks_all_path} ({len(commands)} tasks)")
        print(f"  - {tasks_txt_path} ({len(commands)} tasks)")
        print(f"  - {tasks_p1_path} ({len(p1_cmds)} tasks)")
        print(f"  - {tasks_p2_path} ({len(p2_cmds)} tasks)")

    return commands


def main():
    parser = argparse.ArgumentParser(description="Generate dedicated BBOB Dimension 128 sweep tasks.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds (default: 20)")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per run (default: 100)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=BBOB_DIM128_OUTPUT_DIR,
        help="Output directory for task manifests",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=BBOB_DIM128_OUTPUT_DIR,
        help="Directory where telemetry files will be saved",
    )
    parser.add_argument(
        "--baserundir",
        type=str,
        default=BBOB_DIM128_RUNS_DIR,
        help="Baserundir for CARP-S raw trial logs",
    )

    args = parser.parse_args()
    generate_bbob_dim128_sweep_tasks(
        seeds=args.seeds,
        trials=args.trials,
        output_dir=args.output_dir,
        runs_dir=args.runs_dir,
        baserundir=args.baserundir,
    )


if __name__ == "__main__":
    main()
