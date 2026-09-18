#!/usr/bin/env python3
"""CARP-S Hydra Task Command Generator for SMAC4HPO Meta-Optimization Iterations.

Generates the exact 90 Hydra command lines (18 working dev tasks * 5 seeds) for a single
candidate configuration proposed by SMAC's ask() method.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


def generate_iteration_tasks(
    config: Dict[str, Any],
    iteration: int,
    seeds: Union[int, Sequence[int]] = 5,
    trials: int = 100,
    alpha: float = 0.05,
    output_dir: str = "results/meta_smac_proximity_hpo",
    baserundir: str = "runs/meta_smac_proximity_hpo",
    output_file: Optional[str] = None,
) -> List[str]:
    """Generates all 90 task command lines for a given iteration's configuration.

    Args:
        config: Dictionary containing 'k', 'decay_lambda' (or 'lambda'), and 'eps'.
        iteration: Iteration index (1 to 100).
        seeds: Integer count (e.g. 5) or sequence of seeds [1..5].
        trials: Number of trials per optimization run (T=100).
        alpha: Risk level alpha (default 0.05 -> level = 0.95).
        output_dir: Base directory for telemetry and results.
        baserundir: Base directory for Hydra run output.
        output_file: Optional file path to write generated tasks.

    Returns:
        List of generated command line strings.
    """
    dev_tasks = CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)

    if isinstance(seeds, int):
        seeds_list = list(range(1, seeds + 1))
    else:
        seeds_list = [int(s) for s in seeds]

    k_val = int(config["k"])
    lam_val = float(config.get("decay_lambda", config.get("lambda", 1.345)))
    eps_val = float(config["eps"])
    level_val = round(1.0 - alpha, 4)

    opt_id = f"SMAC20_ProximityLCB_iter{iteration:03d}"
    iter_run_dir = f"{output_dir}/iter_{iteration:03d}"
    iter_base_dir = f"{baserundir}/iter_{iteration:03d}"

    lines: List[str] = []

    for task_arg in dev_tasks:
        task_name = task_arg.split("/")[-1]
        task_cmd = f"+{task_arg}" if not task_arg.startswith("+") else task_arg

        for seed in seeds_list:
            telemetry = f"{iter_run_dir}/telemetry_{opt_id}_{task_name}_seed{seed}.json"
            cmd = (
                f"--config-dir carps_integration/configs "
                f"+optimizer=smac20_proximity_lcb "
                f"++optimizer.acq_func_kwargs.k={k_val} "
                f"++optimizer.acq_func_kwargs.level={level_val} "
                f"++optimizer.acq_func_kwargs.eps={eps_val} "
                f"++optimizer.smac_cfg.model_kwargs.uncertainty_func=proximity_b "
                f"++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda={lam_val} "
                f"{task_cmd} task.optimization_resources.n_trials={trials} "
                f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                f"baserundir={iter_base_dir} "
                f"optimizer_id={opt_id} optimizer_container_id=SMAC20_ProximityLCB"
            )
            lines.append(cmd)

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for line in lines:
                f.write(line + "\n")

    return lines
