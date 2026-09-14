#!/usr/bin/env python3
"""Task generator for CARP-S BBsubset Real-World ML Benchmark Suite (30 Seeds, 14 Tasks).

Generates 840 total tasks (420 proposed DA-EHRF Additive EI + 420 baseline SMAC3 HPO Facade EI)
with strict seed pairing (seeds 1 to 30) across all 14 CARP-S Real-World ML dev tasks
(11 YAHPO + 3 HPOBench ML, strictly excluding BBOB synthetic and NAS tabular benchmarks).
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Sequence, Union

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


def generate_bbsubset_realworld_30seeds_tasks(
    output_path: str = "results/bbsubset_realworld_30seeds_tasks.txt",
    seeds: Union[Sequence[int], None] = None,
    trials: int = 50,
    include_nas: bool = False,
    beta_max: float = 1.0,
    beta_min: float = 0.0,
    warmup_ratio: float = 0.20,
) -> List[str]:
    """Generates the task list with strict seed pairing across real-world ML dev tasks.

    Args:
        output_path: Path to output tasks.txt.
        seeds: List of integer seeds. Defaults to list(range(1, 31)).
        trials: Number of trials per run. Default: 50.
        include_nas: Whether to include the 2 HPOBench NAS tasks. Default: False.
        beta_max: Maximum beta exploration bonus. Default: 1.0.
        beta_min: Minimum beta exploration bonus. Default: 0.0.
        warmup_ratio: Warmup fraction of budget. Default: 0.20.

    Returns:
        List of formatted command lines.
    """
    if seeds is None:
        seeds = list(range(1, 31))
    else:
        seeds = [int(s) for s in seeds]

    tasks = CarpsBBSubsetRegistry.get_realworld_dev_tasks(include_nas=include_nas)
    lines: List[str] = []

    # 1. Proposed: DA-EHRF Additive Epistemic EI
    for task in tasks:
        task_name = task.split("/")[-1]
        task_arg = f"+{task}" if not task.startswith("+") else task
        for seed in seeds:
            telemetry = (
                f"results/bbsubset_realworld_30seeds/proposed/"
                f"telemetry_da_ehrf_additive_ei_{task_name}_seed{seed}.json"
            )
            line = (
                f"--config-dir carps_integration/configs "
                f"+optimizer=dyrf_da_ehrf_additive_ei "
                f"++optimizer.beta_max={beta_max} "
                f"++optimizer.beta_min={beta_min} "
                f"++optimizer.warmup_ratio={warmup_ratio:.2f} "
                f"{task_arg} "
                f"task.optimization_resources.n_trials={trials} "
                f"seed={seed} "
                f"++optimizer.telemetry_path={telemetry} "
                f"optimizer_id=CARPSDynamicRF_DAEHRF_AdditiveEI "
                f"optimizer_container_id=CARPSDynamicRF_DAEHRF_AdditiveEI"
            )
            lines.append(line)

    # 2. Baseline: SMAC3 HPO Facade EI
    for task in tasks:
        task_name = task.split("/")[-1]
        task_arg = f"+{task}" if not task.startswith("+") else task
        for seed in seeds:
            line = (
                f"--config-dir carps_integration/configs "
                f"+optimizer=smac3_hpo_facade_ei "
                f"{task_arg} "
                f"task.optimization_resources.n_trials={trials} "
                f"seed={seed} "
                f"optimizer_id=SMAC3_HPOFacade_ei "
                f"optimizer_container_id=SMAC3_HPOFacade_ei"
            )
            lines.append(line)

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w") as f:
            for line in lines:
                f.write(f"{line}\n")

    print(
        f"Generated {len(lines)} CARP-S Real-World ML tasks "
        f"({len(tasks)} tasks * {len(seeds)} seeds * 2 optimizers) in {output_path}"
    )
    return lines


def parse_seeds_arg(val: str) -> List[int]:
    """Parse comma-separated or integer range seed argument."""
    val = val.strip()
    if "," in val:
        return [int(s.strip()) for s in val.split(",") if s.strip()]
    if val.isdigit():
        count = int(val)
        return list(range(1, count + 1))
    return [int(val)]


def main():
    parser = argparse.ArgumentParser(
        description="Generate CARP-S Real-World ML 30-Seed Tasks (14 Tasks: 11 YAHPO + 3 HPOBench ML)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/bbsubset_realworld_30seeds_tasks.txt",
        help="Output task list file path (default: results/bbsubset_realworld_30seeds_tasks.txt)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
        help="Comma-separated seeds or single integer for count 1..N (default: 1..30)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of trials per run (default: 50)",
    )
    parser.add_argument(
        "--include-nas",
        action="store_true",
        default=False,
        help="Include the 2 HPOBench Tabular NAS tasks (NavalPropulsion, SliceLocalization)",
    )
    parser.add_argument(
        "--beta_max",
        type=float,
        default=1.0,
        help="Maximum beta exploration bonus (default: 1.0)",
    )
    parser.add_argument(
        "--beta_min",
        type=float,
        default=0.0,
        help="Minimum beta exploration bonus (default: 0.0)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.20,
        help="Warmup fraction of budget (default: 0.20)",
    )

    args = parser.parse_args()
    seed_list = parse_seeds_arg(args.seeds)

    generate_bbsubset_realworld_30seeds_tasks(
        output_path=args.output,
        seeds=seed_list,
        trials=args.trials,
        include_nas=args.include_nas,
        beta_max=args.beta_max,
        beta_min=args.beta_min,
        warmup_ratio=args.warmup_ratio,
    )


if __name__ == "__main__":
    main()
