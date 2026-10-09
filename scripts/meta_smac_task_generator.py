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
    method: str = "a",
    risk_alpha: Optional[float] = None,
) -> List[str]:
    """Generates all 90 task command lines for a given iteration's configuration.

    Supports Proximity methods:
    - 'a': SMAC20_ProximityLCB (k, level, eps, decay_lambda)
    - 'b': SMAC20_ProximityB_LCB (level, eps, decay_lambda)
    - 'ac': SMAC20_ProximityAC_LCB (k, k_warmup, level, eps, decay_lambda, alpha)
    - 'bc': SMAC20_ProximityBC_LCB (level, eps, decay_lambda, alpha)

    Args:
        config: Dictionary containing hyperparameters for the specified method.
        iteration: Iteration index (1 to 100).
        seeds: Integer count (e.g. 5) or sequence of seeds [1..5].
        trials: Number of trials per optimization run (T=100).
        alpha: Risk level alpha for LCB acquisition (default 0.05 -> level = 0.95).
        output_dir: Base directory for telemetry and results.
        baserundir: Base directory for Hydra run output.
        output_file: Optional file path to write generated tasks.
        method: Proximity method ('a', 'b', 'ac', 'bc'). Defaults to 'a'.
        risk_alpha: Optional override for risk level alpha.

    Returns:
        List of generated command line strings.
    """
    m = str(config.get("method", method)).lower()
    valid_methods = {"a", "b", "ac", "bc"}
    if m not in valid_methods:
        raise ValueError(
            f"Unknown method '{m}'. Supported methods are 'a', 'b', 'ac', 'bc'."
        )

    dev_tasks = CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)

    if isinstance(seeds, int):
        seeds_list = list(range(1, seeds + 1))
    else:
        seeds_list = [int(s) for s in seeds]

    effective_risk_alpha = risk_alpha if risk_alpha is not None else alpha
    level_val = round(1.0 - effective_risk_alpha, 4)
    eps_val = float(config["eps"])
    lam_val = float(config.get("decay_lambda", config.get("lambda", 1.345)))

    if m in {"a", "ac"}:
        k_val = int(config["k"])
    else:
        k_val = None

    if m in {"ac", "bc"}:
        density_alpha = float(config.get("alpha", 1.0))
    else:
        density_alpha = None

    if m == "a":
        opt_name = "smac20_proximity_lcb"
        opt_id = f"SMAC20_ProximityLCB_iter{iteration:03d}"
        container_id = "SMAC20_ProximityLCB"
        uncertainty_func = "proximity_b"
    elif m == "b":
        opt_name = "smac20_proximity_b_lcb"
        opt_id = f"SMAC20_ProximityB_LCB_iter{iteration:03d}"
        container_id = "SMAC20_ProximityB_LCB"
        uncertainty_func = "proximity_b"
    elif m == "ac":
        opt_name = "smac20_proximity_ac_lcb"
        opt_id = f"SMAC20_ProximityAC_LCB_iter{iteration:03d}"
        container_id = "SMAC20_ProximityAC_LCB"
        uncertainty_func = "proximity_ac"
    elif m == "bc":
        opt_name = "smac20_proximity_bc_lcb"
        opt_id = f"SMAC20_ProximityBC_LCB_iter{iteration:03d}"
        container_id = "SMAC20_ProximityBC_LCB"
        uncertainty_func = "proximity_bc"

    iter_run_dir = f"{output_dir}/iter_{iteration:03d}"
    iter_base_dir = f"{baserundir}/iter_{iteration:03d}"

    lines: List[str] = []

    for task_arg in dev_tasks:
        task_name = task_arg.split("/")[-1]
        task_cmd = f"+{task_arg}" if not task_arg.startswith("+") else task_arg

        for seed in seeds_list:
            telemetry = f"{iter_run_dir}/telemetry_{opt_id}_{task_name}_seed{seed}.json"

            if m == "a":
                cmd = (
                    f"--config-dir carps_integration/configs "
                    f"+optimizer={opt_name} "
                    f"++optimizer.acq_func_kwargs.k={k_val} "
                    f"++optimizer.acq_func_kwargs.level={level_val} "
                    f"++optimizer.acq_func_kwargs.eps={eps_val} "
                    f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={uncertainty_func} "
                    f"++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda={lam_val} "
                    f"{task_cmd} task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"baserundir={iter_base_dir} "
                    f"optimizer_id={opt_id} optimizer_container_id={container_id}"
                )
            elif m == "b":
                cmd = (
                    f"--config-dir carps_integration/configs "
                    f"+optimizer={opt_name} "
                    f"++optimizer.acq_func_kwargs.level={level_val} "
                    f"++optimizer.acq_func_kwargs.eps={eps_val} "
                    f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={uncertainty_func} "
                    f"++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda={lam_val} "
                    f"{task_cmd} task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"baserundir={iter_base_dir} "
                    f"optimizer_id={opt_id} optimizer_container_id={container_id}"
                )
            elif m == "ac":
                cmd = (
                    f"--config-dir carps_integration/configs "
                    f"+optimizer={opt_name} "
                    f"++optimizer.acq_func_kwargs.k={k_val} "
                    f"++optimizer.acq_func_kwargs.k_warmup={k_val} "
                    f"++optimizer.acq_func_kwargs.level={level_val} "
                    f"++optimizer.acq_func_kwargs.eps={eps_val} "
                    f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={uncertainty_func} "
                    f"++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda={lam_val} "
                    f"++optimizer.smac_cfg.model_kwargs.extractor_kwargs.alpha={density_alpha} "
                    f"{task_cmd} task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"baserundir={iter_base_dir} "
                    f"optimizer_id={opt_id} optimizer_container_id={container_id}"
                )
            elif m == "bc":
                cmd = (
                    f"--config-dir carps_integration/configs "
                    f"+optimizer={opt_name} "
                    f"++optimizer.acq_func_kwargs.level={level_val} "
                    f"++optimizer.acq_func_kwargs.eps={eps_val} "
                    f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={uncertainty_func} "
                    f"++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda={lam_val} "
                    f"++optimizer.smac_cfg.model_kwargs.extractor_kwargs.alpha={density_alpha} "
                    f"{task_cmd} task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"baserundir={iter_base_dir} "
                    f"optimizer_id={opt_id} optimizer_container_id={container_id}"
                )
            lines.append(cmd)

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for line in lines:
                f.write(line + "\n")

    return lines
