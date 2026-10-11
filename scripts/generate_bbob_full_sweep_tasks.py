#!/usr/bin/env python3
"""Task Generator for Full CARP-S BBOB Benchmark Sweep across Dimensions.

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
- 360 CARP-S BBOB Tasks (5 dims: 2, 4, 8, 16, 32 * 24 functions * 3 instances)
- 20 strictly paired seeds (seeds 1 to 20 by default)
- 100 trials budget per run (T=100)

Total: 360 tasks * 11 optimizers * 20 seeds = 79,200 runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_bbob_full_registry import CarpsBBOBFullRegistry


ALL_OPTIMIZERS: List[Dict[str, str]] = [
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


def build_bbob_command_line(
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


def generate_bbob_full_sweep_tasks(
    seeds: Union[int, Sequence[int]] = 20,
    trials: int = 100,
    dimensions: Optional[Sequence[int]] = None,
    output_dir: str = "results/sweep_bbob_full_comparison",
    runs_dir: str = "results/sweep_bbob_full_comparison",
    baserundir: str = "runs/sweep_bbob_full_comparison",
) -> List[str]:
    """Generates all Hydra task command lines for the BBOB full sweep."""
    if isinstance(seeds, int):
        seeds_list = list(range(1, seeds + 1))
    else:
        seeds_list = list(seeds)

    if dimensions is None:
        target_dimensions = CarpsBBOBFullRegistry.SUPPORTED_DIMENSIONS
    else:
        target_dimensions = tuple(dimensions)

    tasks_by_dim: Dict[int, List[str]] = {}
    all_selected_tasks: List[str] = []
    for dim in target_dimensions:
        dim_tasks = CarpsBBOBFullRegistry.get_tasks_by_dimension(dim)
        tasks_by_dim[dim] = dim_tasks
        all_selected_tasks.extend(dim_tasks)

    all_commands: List[str] = []
    dim_commands: Dict[int, List[str]] = {d: [] for d in target_dimensions}

    for opt_info in ALL_OPTIMIZERS:
        for seed in seeds_list:
            for dim in target_dimensions:
                for task_arg in tasks_by_dim[dim]:
                    cmd = build_bbob_command_line(
                        opt_info=opt_info,
                        task_arg=task_arg,
                        seed=seed,
                        trials=trials,
                        runs_dir=runs_dir,
                        baserundir=baserundir,
                    )
                    all_commands.append(cmd)
                    dim_commands[dim].append(cmd)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, "tasks_all.txt")
        with open(manifest_path, "w", encoding="utf-8") as f:
            for cmd in all_commands:
                f.write(f"{cmd}\n")

        for dim, cmds in dim_commands.items():
            dim_manifest = os.path.join(output_dir, f"tasks_dim{dim}.txt")
            with open(dim_manifest, "w", encoding="utf-8") as f:
                for cmd in cmds:
                    f.write(f"{cmd}\n")

    return all_commands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate full BBOB sweep tasks across dimensions.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds (default: 20)")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per run (default: 100)")
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32],
        help="Target dimensions (default: 2 4 8 16 32)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sweep_bbob_full_comparison",
        help="Output directory for task manifests",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="results/sweep_bbob_full_comparison",
        help="Directory where telemetry files will be saved",
    )
    parser.add_argument(
        "--baserundir",
        type=str,
        default="runs/sweep_bbob_full_comparison",
        help="Baserundir for CARP-S raw trial logs",
    )

    args = parser.parse_args()
    cmds = generate_bbob_full_sweep_tasks(
        seeds=args.seeds,
        trials=args.trials,
        dimensions=args.dimensions,
        output_dir=args.output_dir,
        runs_dir=args.runs_dir,
        baserundir=args.baserundir,
    )
    print(f"Generated {len(cmds)} commands successfully into {args.output_dir}.")
