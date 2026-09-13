#!/usr/bin/env python3
"""Task generator for CARP-S BBsubset Dev Set using DA-EHRF Additive Uncertainty BO with Warmup Cosine Annealing.

Generates 400 total tasks (200 proposed DA-EHRF Additive EI + 200 baseline SMAC3 HPO)
with strict seed pairing (seeds 1 to 10) across all 20 CARP-S BBsubset dev tasks.
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


def generate_bbsubset_da_ehrf_additive_tasks(
    output_path: str = "results/bbsubset_da_ehrf_additive_tasks.txt",
    seeds: Union[Sequence[int], None] = None,
    trials: int = 50,
    beta_max: float = 1.0,
    beta_min: float = 0.0,
    warmup_ratio: float = 0.20,
) -> List[str]:
    """Generates the task list with strict seed pairing across all 20 dev tasks.

    Args:
        output_path: Path to output tasks.txt.
        seeds: List of integer seeds. Defaults to list(range(1, 11)).
        trials: Number of trials per run. Default: 50.
        beta_max: Exploration bonus maximum. Default: 1.0.
        beta_min: Exploration bonus minimum. Default: 0.0.
        warmup_ratio: Fraction of budget spent on warmup exploration. Default: 0.20.

    Returns:
        List of formatted command lines.
    """
    if seeds is None:
        seeds = list(range(1, 11))
    else:
        seeds = [int(s) for s in seeds]

    tasks = CarpsBBSubsetRegistry.get_dev_tasks()
    lines: List[str] = []

    # 1. Proposed: DA-EHRF Additive Epistemic EI
    for task in tasks:
        task_name = task.split("/")[-1]
        task_arg = f"+{task}" if not task.startswith("+") else task
        for seed in seeds:
            telemetry = f"results/bbsubset_da_ehrf_additive/proposed/telemetry_da_ehrf_additive_ei_{task_name}_seed{seed}.json"
            line = (
                f"--config-dir carps_integration/configs "
                f"+optimizer=dyrf_da_ehrf_additive_ei "
                f"++optimizer.beta_max={beta_max} "
                f"++optimizer.beta_min={beta_min} "
                f"++optimizer.warmup_ratio={warmup_ratio} "
                f"{task_arg} "
                f"task.optimization_resources.n_trials={trials} "
                f"seed={seed} "
                f"++optimizer.telemetry_path={telemetry} "
                f"optimizer_id=CARPSDynamicRF_DAEHRF_AdditiveEI "
                f"optimizer_container_id=CARPSDynamicRF_DAEHRF_AdditiveEI"
            )
            lines.append(line)

    # 2. Baseline: Standard SMAC3 HPO EI
    for task in tasks:
        task_name = task.split("/")[-1]
        task_arg = f"+{task}" if not task.startswith("+") else task
        for seed in seeds:
            telemetry = f"results/bbsubset_da_ehrf_additive/baseline/telemetry_smac3_ei_{task_name}_seed{seed}.json"
            line = (
                f"--config-dir carps_integration/configs "
                f"+optimizer/smac20=hpo "
                f"++optimizer.acq_func_name=ei "
                f"{task_arg} "
                f"task.optimization_resources.n_trials={trials} "
                f"seed={seed} "
                f"++optimizer.telemetry_path={telemetry} "
                f"optimizer_id=SMAC3_HPOFacade_ei "
                f"optimizer_container_id=SMAC3"
            )
            lines.append(line)

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w") as f:
            for line in lines:
                f.write(f"{line}\n")

    print(f"Generated {len(lines)} CARP-S BBsubset DA-EHRF additive tasks in {output_path}")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate CARP-S BBsubset DA-EHRF Additive Uncertainty Tasks"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/bbsubset_da_ehrf_additive_tasks.txt",
        help="Output task list file path",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated seed integers (default: 1..10)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of trials per run (default: 50)",
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
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    generate_bbsubset_da_ehrf_additive_tasks(
        output_path=args.output,
        seeds=seed_list,
        trials=args.trials,
        beta_max=args.beta_max,
        beta_min=args.beta_min,
        warmup_ratio=args.warmup_ratio,
    )


if __name__ == "__main__":
    main()
