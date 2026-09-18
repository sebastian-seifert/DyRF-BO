#!/usr/bin/env python3
"""Master Orchestrator for SMAC4HPO Meta-Optimization Layer over Proximity LCB.

Executes sequential Bayesian Optimization across 100 iterations on CARP-S BBsubset dev tasks:
1. Iteration 1 is guaranteed to evaluate the current random search incumbent (k=25, lambda=1.345, eps=0.1678).
2. For each iteration:
   - smac.ask() proposes candidate configuration theta_i
   - Generates 90 task command lines (18 working dev tasks * 5 seeds)
   - Dispatches evaluation to SLURM job array (or mock evaluation if --dry-run)
   - Gathers telemetry files, normalizes per-task regret using empirical reference bounds
   - Feeds back aggregate meta-loss to smac.tell()
   - Checkpoints progress to meta_smac_checkpoint.json and meta_leaderboard.csv
3. Seamlessly resumes between Batch 1 (iters 1-55, 4950 jobs) and Batch 2 (iters 56-100, 4050 jobs).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory import StatusType, TrialInfo, TrialValue
from ConfigSpace import Configuration

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry
from scripts.meta_smac_proximity_space import (
    create_proximity_meta_configspace,
    build_initial_design_with_incumbent,
)
from scripts.meta_smac_task_generator import generate_iteration_tasks
from scripts.meta_smac_loss import compute_meta_loss, load_dev_reference_bounds


def parse_iteration_results(
    iter_dir: Path,
    iter_base_dir: Path,
    dev_tasks: List[str],
) -> Dict[str, List[float]]:
    """Gathers candidate evaluation results across all seeds for each dev task.

    Supports:
    1. CARP-S FileLogger: reads `trial_logs.jsonl` in `iter_base_dir/**`
    2. SMAC3 runhistory: reads `runhistory.json` in `iter_base_dir/**`
    3. Legacy telemetry JSON: reads `telemetry_*.json` in `iter_dir` or `iter_base_dir`
    """
    results: Dict[str, List[float]] = {t: [] for t in dev_tasks}

    def _match_task(target_path: Path) -> Optional[str]:
        path_str = str(target_path)
        for dt in dev_tasks:
            if dt in path_str:
                return dt
        hydra_cfg = target_path.parent / ".hydra" / "config.yaml"
        if hydra_cfg.exists():
            try:
                content = hydra_cfg.read_text()
                for dt in dev_tasks:
                    if dt in content:
                        return dt
            except Exception:
                pass
        return None

    found_runs: set = set()

    # Strategy 1: Parse trial_logs.jsonl from iter_base_dir
    if iter_base_dir.exists():
        trial_log_files = sorted(list(iter_base_dir.glob("**/trial_logs.jsonl")))
        for tfile in trial_log_files:
            dt = _match_task(tfile)
            if dt is None:
                continue
            costs = []
            try:
                with open(tfile, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        trial_val = row.get("trial_value", {})
                        status = trial_val.get("status", 1)
                        if status in (1, "SUCCESS", "StatusType.SUCCESS"):
                            cost = trial_val.get("cost")
                            if cost is not None and np.isfinite(float(cost)):
                                costs.append(float(cost))
                if costs:
                    best_c = min(costs)
                    results[dt].append(best_c)
                    found_runs.add((dt, str(tfile.parent)))
            except Exception as e:
                print(f"Warning: Failed parsing {tfile}: {e}")

        # Strategy 2: Fallback to runhistory.json in iter_base_dir
        runhistory_files = sorted(list(iter_base_dir.glob("**/runhistory.json")))
        for rh_file in runhistory_files:
            dt = _match_task(rh_file)
            if dt is None:
                continue
            run_key = (dt, str(rh_file.parent.parent))
            run_key_direct = (dt, str(rh_file.parent))
            if run_key in found_runs or run_key_direct in found_runs:
                continue
            try:
                with open(rh_file, "r") as f:
                    rh_data = json.load(f)
                costs = [
                    float(entry["cost"])
                    for entry in rh_data.get("data", [])
                    if entry.get("status") in (1, "SUCCESS", "StatusType.SUCCESS")
                    and entry.get("cost") is not None
                    and np.isfinite(float(entry["cost"]))
                ]
                if costs:
                    best_c = min(costs)
                    results[dt].append(best_c)
                    found_runs.add(run_key_direct)
            except Exception as e:
                print(f"Warning: Failed parsing {rh_file}: {e}")

    # Strategy 3: Fallback to telemetry_*.json files
    telemetry_files = []
    if iter_dir.exists():
        telemetry_files.extend(list(iter_dir.glob("telemetry_*.json")))
    if iter_base_dir.exists():
        telemetry_files.extend(list(iter_base_dir.glob("**/telemetry_*.json")))

    for tfile in sorted(telemetry_files):
        dt = _match_task(tfile)
        if dt is None:
            continue
        try:
            with open(tfile, "r") as f:
                tdata = json.load(f)
            inc_cost = tdata.get("cost_inc", tdata.get("best_cost"))
            if inc_cost is None and "trials" in tdata:
                valid_costs = [
                    float(tr.get("cost", tr.get("trial_value", {}).get("cost", np.nan)))
                    for tr in tdata["trials"]
                    if tr.get("cost", tr.get("trial_value", {}).get("cost")) is not None
                ]
                finite_costs = [c for c in valid_costs if np.isfinite(c)]
                if finite_costs:
                    inc_cost = min(finite_costs)
            if inc_cost is not None and np.isfinite(float(inc_cost)):
                results[dt].append(float(inc_cost))
        except Exception as e:
            print(f"Warning: Failed parsing {tfile}: {e}")

    return results


class MetaSmacOrchestrator:
    def __init__(
        self,
        start_iteration: int = 1,
        end_iteration: int = 100,
        seeds: int = 5,
        trials: int = 100,
        output_dir: str = "results/meta_smac_proximity_hpo",
        baserundir: str = "runs/meta_smac_proximity_hpo",
        resume: bool = False,
        dry_run: bool = False,
        sbatch_script: str = "scripts/submit_meta_smac_step.sbatch",
    ) -> None:
        self.start_iteration = start_iteration
        self.end_iteration = end_iteration
        self.seeds = seeds
        self.trials = trials
        self.output_dir = Path(output_dir)
        self.baserundir = Path(baserundir)
        self.resume = resume
        self.dry_run = dry_run
        self.sbatch_script = sbatch_script

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baserundir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.output_dir / "meta_smac_checkpoint.json"
        self.leaderboard_file = self.output_dir / "meta_leaderboard.csv"
        self.best_config_file = self.output_dir / "best_config.json"

        self.ref_bounds = load_dev_reference_bounds()
        self.dev_tasks = [t.split("/")[-1] for t in CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)]

        self.cs = create_proximity_meta_configspace()
        self.scenario = Scenario(
            configspace=self.cs,
            n_trials=self.end_iteration,
            deterministic=True,
            output_directory=self.output_dir / "smac3_internal",
        )
        self.initial_design = build_initial_design_with_incumbent(self.scenario)

        # Initialize facade
        self.smac_opt = HyperparameterOptimizationFacade(
            scenario=self.scenario,
            target_function=lambda cfg, seed=0: 0.0,
            initial_design=self.initial_design,
            overwrite=not self.resume and not self.checkpoint_file.exists(),
        )

        self.history: List[Dict[str, Any]] = []
        if self.resume or (self.start_iteration > 1 and self.checkpoint_file.exists()):
            self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Loads prior evaluated trials and replays them into SMAC's runhistory."""
        if not self.checkpoint_file.exists():
            return

        with open(self.checkpoint_file, "r") as f:
            data = json.load(f)

        self.history = data.get("history", [])

        # Replay past trials into SMAC's ask/tell state if starting clean SMAC instance
        for entry in self.history:
            cfg_dict = {
                "k": int(entry["k"]),
                "decay_lambda": float(entry["decay_lambda"]),
                "eps": float(entry["eps"]),
            }
            cfg = Configuration(self.cs, values=cfg_dict)
            trial_info = TrialInfo(config=cfg, seed=self.scenario.seed)
            trial_value = TrialValue(cost=float(entry["loss_normalized_regret"]), status=StatusType.SUCCESS)
            try:
                self.smac_opt.tell(trial_info, trial_value, save=False)
            except Exception:
                pass

    def _save_checkpoint(self, last_iter: int) -> None:
        """Saves current state, leaderboard, and best config."""
        if not self.history:
            return

        # Sort history by loss
        sorted_history = sorted(self.history, key=lambda x: x["loss_normalized_regret"])
        best_entry = sorted_history[0]

        checkpoint_data = {
            "last_iteration": last_iter,
            "total_evaluated": len(self.history),
            "best_config": best_entry,
            "history": self.history,
        }

        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        with open(self.best_config_file, "w") as f:
            json.dump(best_entry, f, indent=2)

        df = pd.DataFrame(self.history)
        df.to_csv(self.leaderboard_file, index=False)

    def _mock_evaluate(self, iteration: int, cfg: Dict[str, Any]) -> Dict[str, List[float]]:
        """Mock multi-task evaluation for dry-runs and automated tests."""
        results: Dict[str, List[float]] = {}
        # Synthetic loss function around incumbent: k=25, lambda=1.345, eps=0.1678
        k_err = ((cfg["k"] - 25) / 25.0) ** 2
        lam_err = ((cfg["decay_lambda"] - 1.345) / 1.345) ** 2
        eps_err = ((cfg["eps"] - 0.1678) / 0.1678) ** 2
        dist_factor = 0.2 + 0.5 * (k_err + lam_err + eps_err)

        for task in self.dev_tasks:
            bounds = self.ref_bounds.get(task, {"min": 0.0, "max": 1.0})
            y_min = bounds["min"]
            y_max = bounds["max"]
            spread = max(1e-5, y_max - y_min)
            # Simulated incumbent cost
            synth_cost = y_min + dist_factor * spread
            results[task] = [synth_cost] * self.seeds
        return results

    def _slurm_evaluate(self, iteration: int, task_file: str, iter_dir: Path) -> Dict[str, List[float]]:
        """Submits the iteration's 90 tasks to SLURM and blocks until completion."""
        num_tasks = 18 * self.seeds
        sbatch_cmd = [
            "sbatch",
            "--wait",
            "--parsable",
            f"--array=1-{num_tasks}%25",
            self.sbatch_script,
            str(task_file),
            str(iter_dir),
        ]
        print(f"[Iter {iteration:03d}] Submitting SLURM array: {' '.join(sbatch_cmd)}")
        subprocess.run(sbatch_cmd, check=True)

        iter_base_dir = self.baserundir / f"iter_{iteration:03d}"
        results = parse_iteration_results(iter_dir, iter_base_dir, self.dev_tasks)

        total_runs_found = sum(len(v) for v in results.values())
        expected_runs = len(self.dev_tasks) * self.seeds
        print(f"[Iter {iteration:03d}] Evaluation complete: gathered {total_runs_found}/{expected_runs} runs.")
        for dt, costs in results.items():
            if len(costs) < self.seeds:
                print(f"[Iter {iteration:03d}] Warning: Task {dt} yielded {len(costs)}/{self.seeds} valid runs.")

        return results

    def run(self) -> Dict[str, Any]:
        """Main optimization loop executing from start_iteration to end_iteration."""
        print("=" * 60)
        print(f"Starting SMAC4HPO Meta-Optimization on Proximity LCB")
        print(f"Iterations: {self.start_iteration} to {self.end_iteration}")
        print(f"Seeds: {self.seeds} | Trials per task: {self.trials}")
        print(f"Dry Run: {self.dry_run}")
        print("=" * 60)

        for i in range(self.start_iteration, self.end_iteration + 1):
            iter_dir = self.output_dir / f"iter_{i:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            task_file = iter_dir / "tasks.txt"

            trial_info = self.smac_opt.ask()
            cfg_dict = {
                "k": int(trial_info.config["k"]),
                "decay_lambda": float(trial_info.config["decay_lambda"]),
                "eps": float(trial_info.config["eps"]),
            }

            print(f"\n>>> Iteration {i:03d} / {self.end_iteration:03d}")
            print(f"Proposed: k={cfg_dict['k']}, decay_lambda={cfg_dict['decay_lambda']:.4f}, eps={cfg_dict['eps']:.4f}")

            # Generate Hydra task command lines
            generate_iteration_tasks(
                config=cfg_dict,
                iteration=i,
                seeds=self.seeds,
                trials=self.trials,
                output_dir=str(self.output_dir),
                baserundir=str(self.baserundir),
                output_file=str(task_file),
            )

            # Evaluate (Dry-run mock or SLURM cluster)
            if self.dry_run:
                task_results = self._mock_evaluate(i, cfg_dict)
            else:
                task_results = self._slurm_evaluate(i, str(task_file), iter_dir)

            meta_loss = compute_meta_loss(task_results, self.ref_bounds)
            print(f"Aggregate Meta-Loss: {meta_loss:.4f}")

            # Feedback to SMAC
            trial_value = TrialValue(cost=float(meta_loss), status=StatusType.SUCCESS)
            self.smac_opt.tell(trial_info, trial_value)

            # Record history entry
            entry = {
                "iteration": i,
                "config_id": f"cfg_{i:03d}",
                "optimizer_id": f"SMAC20_ProximityLCB_iter{i:03d}",
                "k": cfg_dict["k"],
                "decay_lambda": cfg_dict["decay_lambda"],
                "eps": cfg_dict["eps"],
                "loss_normalized_regret": float(meta_loss),
            }
            self.history.append(entry)
            self._save_checkpoint(i)

        sorted_history = sorted(self.history, key=lambda x: x["loss_normalized_regret"])
        best = sorted_history[0]
        print("\n" + "=" * 60)
        print("SMAC4HPO Meta-Optimization Batch Completed!")
        print(f"Best Configuration: k={best['k']}, decay_lambda={best['decay_lambda']:.4f}, eps={best['eps']:.4f}")
        print(f"Best Meta-Loss: {best['loss_normalized_regret']:.4f}")
        print(f"Leaderboard saved to: {self.leaderboard_file}")
        print("=" * 60)
        return best


def main() -> None:
    parser = argparse.ArgumentParser(description="SMAC4HPO Meta-Optimizer for Proximity LCB.")
    parser.add_argument("--start-iteration", type=int, default=1, help="Start iteration (default: 1)")
    parser.add_argument("--end-iteration", type=int, default=100, help="End iteration (default: 100)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (default: 5)")
    parser.add_argument("--trials", type=int, default=100, help="Trials per task (default: 100)")
    parser.add_argument("--output-dir", default="results/meta_smac_proximity_hpo", help="Output directory")
    parser.add_argument("--baserundir", default="runs/meta_smac_proximity_hpo", help="Hydra baserundir")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Execute mock evaluation for verification")
    args = parser.parse_args()

    orchestrator = MetaSmacOrchestrator(
        start_iteration=args.start_iteration,
        end_iteration=args.end_iteration,
        seeds=args.seeds,
        trials=args.trials,
        output_dir=args.output_dir,
        baserundir=args.baserundir,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
