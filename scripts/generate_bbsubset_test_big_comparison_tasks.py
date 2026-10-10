#!/usr/bin/env python3
"""Task Generator for CARP-S BBSubset Held-Out Test Big Comparison Sweep.

Generates 6,600 task command lines evaluating 11 optimizers:
- Baselines (2):
  1. SMAC3_HPOFacade_lcb (Standard Gaussian LCB, beta=3.8416 / kappa=1.96)
  2. SMAC3_HPOFacade_ei (Standard Expected Improvement)
- Entropy (1):
  3. SMAC20_Entropy_LCB (Shaker GMM Mutual Information in variance units, beta=3.8416)
- Proximity Family (8):
  4. SMAC20_ProximityA_LCB (Method A: k=28, lambda=0.205, eps=0.081, level=0.95)
  5. SMAC20_ProximityA_LCB_CV (Method A + CV residuals)
  6. SMAC20_ProximityB_LCB (Method B: lambda=1.273, eps=0.156, level=0.95)
  7. SMAC20_ProximityB_LCB_CV (Method B + CV residuals)
  8. SMAC20_ProximityAC_LCB (Method AC: k=7, lambda=0.328, alpha=0.863, eps=0.093, level=0.95)
  9. SMAC20_ProximityAC_LCB_CV (Method AC + CV residuals)
  10. SMAC20_ProximityBC_LCB (Method BC: lambda=1.494, alpha=1.474, eps=0.143, level=0.95)
  11. SMAC20_ProximityBC_LCB_CV (Method BC + CV residuals)

Across:
- 20 CARP-S Held-Out Test Tasks (8 BBOB + 1 HPOBench + 11 YAHPO Gym)
- 30 strictly paired seeds (seeds 1 to 30)
- 100 trials budget per run (T=100)

Total: 20 tasks * 11 optimizers * 30 seeds = 6,600 runs.
Partitioned into 2 cluster batches:
- Part 1 (3,600 runs): 6 optimizers (LCB, EI, Entropy, Proximity A, Proximity A CV, Proximity AC)
- Part 2 (3,000 runs): 5 optimizers (Proximity AC CV, Proximity B, Proximity B CV, Proximity BC, Proximity BC CV)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


OPTIMIZERS_PART1 = [
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
        "id": "SMAC20_Entropy_LCB",
        "container_id": "SMAC20_Entropy_LCB",
        "config_arg": "+optimizer=smac20_entropy_lcb",
    },
    {
        "id": "SMAC20_ProximityA_LCB",
        "container_id": "SMAC20_ProximityA_LCB",
        "config_arg": "+optimizer=smac20_proximity_a_lcb",
    },
    {
        "id": "SMAC20_ProximityA_LCB_CV",
        "container_id": "SMAC20_ProximityA_LCB_CV",
        "config_arg": "+optimizer=smac20_proximity_a_lcb_cv",
    },
    {
        "id": "SMAC20_ProximityAC_LCB",
        "container_id": "SMAC20_ProximityAC_LCB",
        "config_arg": "+optimizer=smac20_proximity_ac_lcb",
    },
]

OPTIMIZERS_PART2 = [
    {
        "id": "SMAC20_ProximityAC_LCB_CV",
        "container_id": "SMAC20_ProximityAC_LCB_CV",
        "config_arg": "+optimizer=smac20_proximity_ac_lcb_cv",
    },
    {
        "id": "SMAC20_ProximityB_LCB",
        "container_id": "SMAC20_ProximityB_LCB",
        "config_arg": "+optimizer=smac20_proximity_b_lcb",
    },
    {
        "id": "SMAC20_ProximityB_LCB_CV",
        "container_id": "SMAC20_ProximityB_LCB_CV",
        "config_arg": "+optimizer=smac20_proximity_b_lcb_cv",
    },
    {
        "id": "SMAC20_ProximityBC_LCB",
        "container_id": "SMAC20_ProximityBC_LCB",
        "config_arg": "+optimizer=smac20_proximity_bc_lcb",
    },
    {
        "id": "SMAC20_ProximityBC_LCB_CV",
        "container_id": "SMAC20_ProximityBC_LCB_CV",
        "config_arg": "+optimizer=smac20_proximity_bc_lcb_cv",
    },
]

ALL_OPTIMIZERS = OPTIMIZERS_PART1 + OPTIMIZERS_PART2


def build_command_line(
    opt_info: dict,
    task_arg: str,
    seed: int,
    trials: int,
    runs_dir: str,
    baserundir: str,
) -> str:
    task_name = task_arg.split("/")[-1]
    task_cmd = f"+{task_arg}" if not task_arg.startswith("+") else task_arg
    opt_id = opt_info["id"]
    container_id = opt_info["container_id"]
    config_arg = opt_info["config_arg"]

    telemetry_path = f"{runs_dir}/telemetry_{opt_id}_{task_name}_seed{seed}.json"

    cmd = (
        f"--config-dir carps_integration/configs "
        f"{config_arg} "
        f"{task_cmd} "
        f"task.optimization_resources.n_trials={trials} "
        f"seed={seed} "
        f"++optimizer.telemetry_path={telemetry_path} "
        f"baserundir={baserundir} "
        f"optimizer_id={opt_id} "
        f"optimizer_container_id={container_id}"
    )
    return cmd


def generate_bbsubset_test_big_comparison_tasks(
    seeds: Union[int, Sequence[int]] = 30,
    trials: int = 100,
    output_dir: str | None = "results/sweep_bbsubset_test_big_comparison",
    runs_dir: str = "results/sweep_bbsubset_test_big_comparison",
    baserundir: str = "runs/sweep_bbsubset_test_big_comparison",
) -> Tuple[List[str], List[str]]:
    tasks = CarpsBBSubsetRegistry.get_test_tasks()
    if isinstance(seeds, int):
        seeds_list = list(range(1, seeds + 1))
    else:
        seeds_list = [int(s) for s in seeds]

    if output_dir:
        os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    if runs_dir:
        os.makedirs(os.path.abspath(runs_dir), exist_ok=True)
    if baserundir:
        os.makedirs(os.path.abspath(baserundir), exist_ok=True)

    part1_lines: List[str] = []
    for opt_info in OPTIMIZERS_PART1:
        for task_arg in tasks:
            for seed in seeds_list:
                part1_lines.append(
                    build_command_line(opt_info, task_arg, seed, trials, runs_dir, baserundir)
                )

    part2_lines: List[str] = []
    for opt_info in OPTIMIZERS_PART2:
        for task_arg in tasks:
            for seed in seeds_list:
                part2_lines.append(
                    build_command_line(opt_info, task_arg, seed, trials, runs_dir, baserundir)
                )

    if output_dir:
        part1_file = os.path.join(output_dir, "tasks_part1.txt")
        part2_file = os.path.join(output_dir, "tasks_part2.txt")
        master_file = os.path.join(output_dir, "tasks_all.txt")

        with open(part1_file, "w", encoding="utf-8") as f:
            for line in part1_lines:
                f.write(line + "\n")

        with open(part2_file, "w", encoding="utf-8") as f:
            for line in part2_lines:
                f.write(line + "\n")

        with open(master_file, "w", encoding="utf-8") as f:
            for line in part1_lines + part2_lines:
                f.write(line + "\n")

        print(f"[INFO] Generated {len(part1_lines)} tasks in {part1_file}")
        print(f"[INFO] Generated {len(part2_lines)} tasks in {part2_file}")
        print(f"[INFO] Generated {len(part1_lines) + len(part2_lines)} total tasks in {master_file}")

    return part1_lines, part2_lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate CARP-S BBSubset test big comparison tasks (6,600 runs)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sweep_bbsubset_test_big_comparison",
        help="Target output directory for task manifests.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Number of independent paired seeds (default: 30).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Trial budget T per run (default: 100).",
    )
    args = parser.parse_args()

    generate_bbsubset_test_big_comparison_tasks(
        seeds=args.seeds,
        trials=args.trials,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
