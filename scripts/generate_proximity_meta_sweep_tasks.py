#!/usr/bin/env python3
"""Task Generator for Proximity Lower Bound Meta-HPO Hyperparameter Sweep.

Generates 4,500 task command lines across:
- N = 50 configurations sampled via Sobol sequence (scipy.stats.qmc.Sobol)
- 18 working CARP-S BBsubset dev tasks (excluding broken NAS tasks)
- 5 seeds per task (seeds 1 to 5)
- Budget T = 100 trials per task run

Total: 50 configs * 18 tasks * 5 seeds = 4,500 runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry
from scripts.sample_proximity_meta_configs import sample_proximity_meta_configs


def generate_proximity_meta_tasks(
    output_file: str = "results/sweep_proximity_meta_hpo/tasks.txt",
    configs_path: Optional[str] = None,
    runs_dir: str = "results/sweep_proximity_meta_hpo",
    baserundir: str = "runs/sweep_proximity_meta_hpo",
    n_configs: int = 50,
    seeds: Union[int, Sequence[int]] = 5,
    trials: int = 100,
    seed_sobol: int = 42,
    alpha: float = 0.05,
) -> List[str]:
    """Generates all 4,500 Hydra task command lines for the Meta-HPO sweep.

    Args:
        output_file: File to write generated tasks.txt.
        configs_path: Path to pre-sampled configs JSON (if None, samples dynamically).
        runs_dir: Directory where telemetry output files will be written.
        baserundir: Hydra base run directory.
        n_configs: Number of configs to sample if configs_path is not given (default: 50).
        seeds: Integer count (e.g. 5) or sequence of seeds [1..5].
        trials: Number of trials per optimization run (T=100).
        seed_sobol: Seed for Sobol sequence if generating configs dynamically.
        alpha: Nominal risk level (default: 0.05 -> level = 0.95).

    Returns:
        List of generated command line strings.
    """
    tasks = CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)

    if isinstance(seeds, int):
        seeds_list = list(range(1, seeds + 1))
    else:
        seeds_list = [int(s) for s in seeds]

    # Load or sample configs
    if configs_path and os.path.exists(configs_path):
        with open(configs_path, "r") as f:
            configs = json.load(f)
    else:
        configs = sample_proximity_meta_configs(
            n_configs=n_configs,
            seed=seed_sobol,
            output_file=configs_path,
        )

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    if runs_dir:
        Path(runs_dir).mkdir(parents=True, exist_ok=True)
    if baserundir:
        Path(baserundir).mkdir(parents=True, exist_ok=True)

    level = round(1.0 - alpha, 4)
    lines: List[str] = []

    for cfg in configs:
        cfg_id = cfg.get("config_id", "cfg")
        k_val = int(cfg["k"])
        lam_val = float(cfg.get("decay_lambda", cfg.get("lambda", 1.0)))
        eps_val = float(cfg["eps"])
        opt_id = cfg.get("optimizer_id", f"SMAC20_ProximityLCB_{cfg_id}")

        for task_arg in tasks:
            task_name = task_arg.split("/")[-1]
            task_cmd = f"+{task_arg}" if not task_arg.startswith("+") else task_arg

            for seed in seeds_list:
                telemetry = f"{runs_dir}/telemetry_{opt_id}_{task_name}_seed{seed}.json"
                cmd = (
                    f"--config-dir carps_integration/configs "
                    f"+optimizer=smac20_proximity_lcb "
                    f"++optimizer.acq_func_kwargs.k={k_val} "
                    f"++optimizer.acq_func_kwargs.level={level} "
                    f"++optimizer.acq_func_kwargs.eps={eps_val} "
                    f"++optimizer.smac_cfg.model_kwargs.uncertainty_func=proximity_b "
                    f"++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda={lam_val} "
                    f"{task_cmd} task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"baserundir={baserundir} "
                    f"optimizer_id={opt_id} optimizer_container_id=SMAC20_ProximityLCB"
                )
                lines.append(cmd)

    if output_file:
        with open(output_file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CARP-S BBsubset Proximity Meta-HPO sweep tasks."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/sweep_proximity_meta_hpo/tasks.txt",
        help="Path to output tasks.txt (default: results/sweep_proximity_meta_hpo/tasks.txt)",
    )
    parser.add_argument(
        "--configs",
        "-c",
        default="results/sweep_proximity_meta_hpo/sobol_configs.json",
        help="Path to pre-sampled sobol_configs.json (default: results/sweep_proximity_meta_hpo/sobol_configs.json)",
    )
    parser.add_argument(
        "--runs-dir",
        default="results/sweep_proximity_meta_hpo",
        help="Directory to store telemetry and run outputs (default: results/sweep_proximity_meta_hpo)",
    )
    parser.add_argument(
        "--baserundir",
        default="runs/sweep_proximity_meta_hpo",
        help="Hydra baserundir (default: runs/sweep_proximity_meta_hpo)",
    )
    parser.add_argument(
        "--n-configs",
        type=int,
        default=50,
        help="Number of Sobol configs to sample if configs file does not exist (default: 50)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds to run per task (default: 5)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Optimization budget in trials per task (default: 100)",
    )
    args = parser.parse_args()

    lines = generate_proximity_meta_tasks(
        output_file=args.output,
        configs_path=args.configs,
        runs_dir=args.runs_dir,
        baserundir=args.baserundir,
        n_configs=args.n_configs,
        seeds=args.seeds,
        trials=args.trials,
    )
    print(f"Generated {len(lines)} tasks in {args.output}")


if __name__ == "__main__":
    main()
