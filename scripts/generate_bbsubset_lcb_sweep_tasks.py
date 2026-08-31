#!/usr/bin/env python3
"""Task Generator for CARP-S BBsubset (Blackbox Single-Objective) Dev Set LCB Sweeps.

Evaluates 15 competitor configurations using Lower Confidence Bound (LCB) acquisition
across dual exploration schedules:
1. 'constant': beta_max=1.0, beta_min=1.0, warmup_ratio=1.0 (Fixed constant bonus)
2. 'annealed': beta_max=1.0, beta_min=0.0, warmup_ratio=0.20 (Warmup cosine decay)

Competitor matrix (15 configurations):
1. Standard SMAC3 HPOFacade Baseline (1 configuration)
2. Direct Variance Replacement (7 configurations)
3. Decoupled Additive Epistemic BO (7 configurations)

Total scale: 2 schedules * 20 dev tasks * 15 competitors * 5 seeds = 3,000 task executions.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Sequence, Union

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry

# 7 Active Epistemic Approaches (Chen variance is strictly excluded)
ACTIVE_APPROACHES: List[str] = [
    "standard_disagreement",
    "shaker_entropy",
    "likelihood_credal",
    "standard_proximity",
    "proximity_b",
    "proximity_bc",
    "proximity_auto_lambda",
]

# Dual Exploration Schedules for Additive Epistemic BO
SCHEDULE_CONFIGS: Dict[str, Dict[str, float]] = {
    "constant": {
        "beta_max": 1.0,
        "beta_min": 1.0,
        "warmup_ratio": 1.0,
    },
    "annealed": {
        "beta_max": 1.0,
        "beta_min": 0.0,
        "warmup_ratio": 0.20,
    },
}


def get_approach_configs() -> List[Dict[str, Any]]:
    """Returns the 15 competitor optimizer configurations for the LCB sweep."""
    configs: List[Dict[str, Any]] = []

    # 1. Reference SMAC3 Baseline (1 configuration)
    configs.append({
        "key": "smac3_baseline",
        "paradigm": "baseline",
        "optimizer_id": "SMAC3_HPOFacade_lcb",
        "container_id": "SMAC3",
    })

    # 2. Direct Variance Drop-in Surrogates (7 configurations)
    for approach in ACTIVE_APPROACHES:
        configs.append({
            "key": approach,
            "paradigm": "direct",
            "extractor_name": approach,
            "optimizer_id": f"SMAC20_CustomUncertainty_lcb_{approach}",
            "container_id": "SMAC20_CustomUncertainty",
        })

    # 3. Decoupled Additive Epistemic BO (7 configurations)
    for approach in ACTIVE_APPROACHES:
        configs.append({
            "key": f"{approach}_additive",
            "paradigm": "additive",
            "extractor_name": approach,
            "optimizer_id": f"CARPSDynamicRF_AdditiveEpistemic_lcb_{approach}",
            "container_id": "CARPSDynamicRF",
        })

    return configs


def generate_bbsubset_lcb_sweep_tasks(
    output_file: str = "results/sweep_bbsubset_lcb/tasks.txt",
    runs_dir: str = "results/bbsubset_lcb",
    baserundir: str = "runs/bbsubset_runs",
    seeds: Union[int, Sequence[int]] = 5,
    trials: int = 50,
    paradigm: str = "all",
    schedules: Union[str, Sequence[str]] = "both",
) -> List[str]:
    """Generates all Hydra task command lines for the CARP-S BBsubset LCB sweep."""
    tasks = CarpsBBSubsetRegistry.get_dev_tasks()
    approaches = get_approach_configs()

    if paradigm != "all":
        approaches = [a for a in approaches if a["paradigm"] == paradigm]
        if not approaches:
            raise ValueError(f"Unknown paradigm or no approaches found: {paradigm}")

    if isinstance(schedules, str):
        if schedules == "both":
            active_schedules = ["constant", "annealed"]
        elif schedules in SCHEDULE_CONFIGS:
            active_schedules = [schedules]
        else:
            raise ValueError(f"Unknown schedule: {schedules}. Supported choices: 'both', 'constant', 'annealed'")
    else:
        active_schedules = list(schedules)
        for s in active_schedules:
            if s not in SCHEDULE_CONFIGS:
                raise ValueError(f"Unknown schedule: {s}. Supported choices: 'constant', 'annealed'")

    if isinstance(seeds, int):
        seeds_list = list(range(1, seeds + 1))
    else:
        seeds_list = list(seeds)

    lines: List[str] = []

    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    for sched in active_schedules:
        sched_params = SCHEDULE_CONFIGS[sched]
        sched_runs_dir = f"{runs_dir}/{sched}" if runs_dir else sched
        sched_baserundir = f"{baserundir}/{sched}" if baserundir else sched

        if runs_dir:
            os.makedirs(os.path.abspath(sched_runs_dir), exist_ok=True)
        if baserundir:
            os.makedirs(os.path.abspath(sched_baserundir), exist_ok=True)

        for task_arg in tasks:
            task_name = task_arg.split("/")[-1]
            task_cmd = f"+{task_arg}" if not task_arg.startswith("+") else task_arg

            for approach in approaches:
                opt_id = approach["optimizer_id"]
                cont_id = approach["container_id"]

                for seed in seeds_list:
                    telemetry_path = f"{sched_runs_dir}/telemetry_{opt_id}_{task_name}_seed{seed}.json"

                    if approach["paradigm"] == "baseline":
                        cmd = (
                            f"--config-dir carps_integration/configs "
                            f"+optimizer/smac20=hpo "
                            f"++optimizer.acq_func_name=lcb "
                            f"{task_cmd} task.optimization_resources.n_trials={trials} "
                            f"seed={seed} ++optimizer.telemetry_path={telemetry_path} "
                            f"baserundir={sched_baserundir} "
                            f"optimizer_id={opt_id} optimizer_container_id={cont_id}"
                        )
                    elif approach["paradigm"] == "direct":
                        extractor = approach["extractor_name"]
                        cmd = (
                            f"--config-dir carps_integration/configs "
                            f"+optimizer=smac20_custom_uncertainty "
                            f"++optimizer.acq_func_name=lcb "
                            f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={extractor} "
                            f"{task_cmd} task.optimization_resources.n_trials={trials} "
                            f"seed={seed} ++optimizer.telemetry_path={telemetry_path} "
                            f"baserundir={sched_baserundir} "
                            f"optimizer_id={opt_id} optimizer_container_id={cont_id}"
                        )
                    elif approach["paradigm"] == "additive":
                        extractor = approach["extractor_name"]
                        b_max = sched_params["beta_max"]
                        b_min = sched_params["beta_min"]
                        w_ratio = sched_params["warmup_ratio"]
                        cmd = (
                            f"--config-dir carps_integration/configs "
                            f"+optimizer=dyrf_additive_epistemic_lcb "
                            f"++optimizer.extractor_name={extractor} "
                            f"++optimizer.beta_max={b_max} "
                            f"++optimizer.beta_min={b_min} "
                            f"++optimizer.warmup_ratio={w_ratio} "
                            f"{task_cmd} task.optimization_resources.n_trials={trials} "
                            f"seed={seed} ++optimizer.telemetry_path={telemetry_path} "
                            f"baserundir={sched_baserundir} "
                            f"optimizer_id={opt_id} optimizer_container_id={cont_id}"
                        )
                    else:
                        raise ValueError(f"Unknown paradigm: {approach['paradigm']}")

                    lines.append(cmd)

    if output_file:
        with open(output_file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CARP-S tasks for BBsubset Dual-Schedule LCB Benchmark Sweep")
    parser.add_argument("-o", "--output-file", type=str, default="results/sweep_bbsubset_lcb/tasks.txt",
                        help="Path to output tasks.txt (default: results/sweep_bbsubset_lcb/tasks.txt)")
    parser.add_argument("-r", "--runs-dir", type=str, default="results/bbsubset_lcb",
                        help="Directory for telemetry JSON files (default: results/bbsubset_lcb)")
    parser.add_argument("-b", "--baserundir", type=str, default="runs/bbsubset_runs",
                        help="Root directory for CARP-S run outputs (default: runs/bbsubset_runs)")
    parser.add_argument("-s", "--seeds", type=int, default=5,
                        help="Number of random seeds (default: 5)")
    parser.add_argument("-t", "--trials", type=int, default=50,
                        help="Number of trials per run (default: 50)")
    parser.add_argument("-p", "--paradigm", type=str, default="all",
                        choices=["all", "baseline", "direct", "additive"],
                        help="Filter optimizer paradigm (default: all)")
    parser.add_argument("--schedules", type=str, default="both",
                        choices=["both", "constant", "annealed"],
                        help="Beta exploration schedules to generate (default: both)")

    args = parser.parse_args()
    tasks = generate_bbsubset_lcb_sweep_tasks(
        output_file=args.output_file,
        runs_dir=args.runs_dir,
        baserundir=args.baserundir,
        seeds=args.seeds,
        trials=args.trials,
        paradigm=args.paradigm,
        schedules=args.schedules,
    )
    print(f"[✓] Generated {len(tasks)} tasks in '{args.output_file}'!")
    print(f"    Schedules: {args.schedules}")
    print(f"    Telemetry directory: '{args.runs_dir}'")
    print(f"    Baserundir: '{args.baserundir}'")


if __name__ == "__main__":
    main()
