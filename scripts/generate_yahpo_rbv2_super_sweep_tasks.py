#!/usr/bin/env python3
"""Task Generator for Full CARP-S YAHPO Gym rbv2_super Benchmark Sweep.

Generates CLI task strings for all 11 evaluated optimizers:
- Baselines (2):
  1. SMAC3_HPOFacade_lcb (Standard Gaussian LCB)
  2. SMAC3_HPOFacade_ei (Standard Expected Improvement)
- Entropy (1):
  3. SMAC20_Entropy_LCB (Shaker GMM Mutual Information)
- Proximity Family (8):
  4. SMAC20_ProximityA_LCB
  5. SMAC20_ProximityA_LCB_CV
  6. SMAC20_ProximityAC_LCB
  7. SMAC20_ProximityAC_LCB_CV
  8. SMAC20_ProximityB_LCB
  9. SMAC20_ProximityB_LCB_CV
  10. SMAC20_ProximityBC_LCB
  11. SMAC20_ProximityBC_LCB_CV

Across:
- 103 YAHPO rbv2_super OpenML tasks
- 20 strictly paired seeds (seeds 1 to 20 by default)
- 100 trials budget per run (T=100)

Total: 103 tasks * 11 optimizers * 20 seeds = 22,660 runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_realworld_registry import CarpsRealworldRegistry
from scripts.generate_bbob_full_sweep_tasks import ALL_OPTIMIZERS

YAHPO_SUPER_OUTPUT_DIR = "results/sweep_yahpo_rbv2_super"
YAHPO_SUPER_RUNS_DIR = "runs/sweep_yahpo_rbv2_super"


def build_yahpo_super_command_line(
    opt_info: Dict[str, str],
    task_arg: str,
    seed: int,
    trials: int,
    runs_dir: str,
    baserundir: str,
) -> str:
    """Builds a single execution command for run_carps_patched.py."""
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


def generate_yahpo_rbv2_super_sweep_tasks(
    seeds: Union[int, Sequence[int]] = 20,
    trials: int = 100,
    output_dir: str = YAHPO_SUPER_OUTPUT_DIR,
    runs_dir: str = YAHPO_SUPER_OUTPUT_DIR,
    baserundir: str = YAHPO_SUPER_RUNS_DIR,
) -> List[str]:
    """Generates all Hydra task command lines for the YAHPO rbv2_super sweep."""
    if isinstance(seeds, int):
        seeds_list = list(range(1, seeds + 1))
    else:
        seeds_list = list(seeds)

    super_tasks = CarpsRealworldRegistry.get_super_tasks()

    commands: List[str] = []
    for opt in ALL_OPTIMIZERS:
        for task_arg in super_tasks:
            for seed in seeds_list:
                cmd = build_yahpo_super_command_line(
                    opt_info=opt,
                    task_arg=task_arg,
                    seed=seed,
                    trials=trials,
                    runs_dir=runs_dir,
                    baserundir=baserundir,
                )
                commands.append(cmd)

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "tasks.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for c in commands:
            f.write(c + "\n")

    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate task manifest for full YAHPO rbv2_super CARP-S sweep."
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
        default=YAHPO_SUPER_OUTPUT_DIR,
        help="Directory to save tasks.txt manifest.",
    )
    parser.add_argument(
        "--baserundir",
        type=str,
        default=YAHPO_SUPER_RUNS_DIR,
        help="Base rundir passed to Hydra for CARP-S logs.",
    )

    args = parser.parse_args()
    tasks = generate_yahpo_rbv2_super_sweep_tasks(
        seeds=args.seeds,
        trials=args.trials,
        output_dir=args.output_dir,
        runs_dir=args.output_dir,
        baserundir=args.baserundir,
    )
    print(f"Generated {len(tasks)} tasks to {args.output_dir}/tasks.txt")


if __name__ == "__main__":
    main()
