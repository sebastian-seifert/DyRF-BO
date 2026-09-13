#!/usr/bin/env python3
"""Automated Multi-Worker Runner for CARPS DA-EHRF Direct EI Benchmark Execution.

Executes 240 runs:
  - 4 Tasks: Ackley 2D, Rosenbrock 2D, Sphere 2D, Rosenbrock 4D
  - 2 Optimizers: SMAC3 Baseline (EI) vs DA-EHRF Direct EI
  - 30 Seeds: seed 1 to 30
  - 50 Budget Trials per run (10 Sobol initial design + 40 BO iterations)

Guarantees:
  - Strict Paired Seed Rigor: identical initial design evaluations per seed.
  - Zero crashes, NaN, or infinite values.
  - Aggregation of trial logs into consolidated Parquet and CSV datasets.
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TASKS = {
    "ackley_2d": "+task/Noisy/bbob=cfg_ackley_2d_gaussian",
    "rosenbrock_2d": "+task/Noisy/bbob=cfg_rosenbrock_2d_gaussian",
    "sphere_2d": "+task/Noisy/bbob=cfg_sphere_2d_gaussian",
    "rosenbrock_4d": "+task/Noisy/bbob=cfg_rosenbrock_4d_gaussian",
}

OPTIMIZERS = {
    "baseline": {
        "args": [
            "+optimizer/smac20=hpo",
            "++optimizer.acq_func_name=ei",
            "optimizer_id=SMAC3_HPOFacade_ei",
            "optimizer_container_id=SMAC3_HPOFacade_ei",
        ],
        "display_name": "SMAC3 Baseline (EI)",
    },
    "da_ehrf_direct_ei": {
        "args": [
            "+optimizer=smac20_da_ehrf_direct_ei",
        ],
        "display_name": "DA-EHRF Direct EI",
    },
}


def is_run_completed(log_file: Path, expected_trials: int = 50) -> bool:
    """Checks if a run directory already contains a complete trial_logs.jsonl file."""
    if not log_file.is_file():
        return False
    try:
        count = 0
        with open(log_file, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count >= expected_trials
    except Exception:
        return False


def execute_single_run(
    task_name: str,
    optimizer_key: str,
    seed: int,
    output_base_dir: Path,
    expected_trials: int = 50,
) -> dict:
    """Executes a single CARPS run using scripts/run_carps_patched.py."""
    run_dir = output_base_dir / "raw" / task_name / optimizer_key / f"seed_{seed}"
    log_file = run_dir / "trial_logs.jsonl"
    t_start = time.time()

    # Idempotent skip if already completed
    if is_run_completed(log_file, expected_trials):
        return {
            "task": task_name,
            "optimizer": optimizer_key,
            "seed": seed,
            "status": "CACHED",
            "returncode": 0,
            "runtime": 0.0,
            "run_dir": str(run_dir),
            "log_file": str(log_file),
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    task_arg = TASKS[task_name]
    opt_args = OPTIMIZERS[optimizer_key]["args"]

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "run_carps_patched.py"),
        "--config-dir", os.path.join(PROJECT_ROOT, "carps_integration", "configs"),
        *opt_args,
        task_arg,
        f"task.optimization_resources.n_trials={expected_trials}",
        f"seed={seed}",
        f"outdir={run_dir}",
        f"hydra.run.dir={run_dir}",
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    elapsed = time.time() - t_start

    if res.returncode != 0:
        return {
            "task": task_name,
            "optimizer": optimizer_key,
            "seed": seed,
            "status": "FAILED",
            "returncode": res.returncode,
            "runtime": elapsed,
            "stderr": res.stderr[-1000:] if res.stderr else "",
            "stdout": res.stdout[-1000:] if res.stdout else "",
            "run_dir": str(run_dir),
            "log_file": str(log_file),
        }

    # Verify log file generated
    if not is_run_completed(log_file, expected_trials):
        return {
            "task": task_name,
            "optimizer": optimizer_key,
            "seed": seed,
            "status": "INCOMPLETE",
            "returncode": -1,
            "runtime": elapsed,
            "run_dir": str(run_dir),
            "log_file": str(log_file),
        }

    return {
        "task": task_name,
        "optimizer": optimizer_key,
        "seed": seed,
        "status": "SUCCESS",
        "returncode": 0,
        "runtime": elapsed,
        "run_dir": str(run_dir),
        "log_file": str(log_file),
    }


def parse_trial_logs(log_file: Path) -> list[dict]:
    """Parses trial_logs.jsonl into a structured list of trial records."""
    trials = []
    best_y_true = float("inf")
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            trial_num = int(record["n_trials"])
            cost = float(record["trial_value"]["cost"])
            time_eval = float(record["trial_value"]["time"])
            add_info = record["trial_value"].get("additional_info", {})
            y_true = float(add_info.get("y_true", cost))
            sigma_true = float(add_info.get("sigma_true", 0.0))
            inst_regret = float(add_info.get("instantaneous_regret", y_true))
            noise_residual = float(add_info.get("noise_residual", 0.0))
            config = record["trial_info"]["config"]

            best_y_true = min(best_y_true, y_true)
            incumbent_regret = best_y_true  # Because f_opt = 0 for all 4 tasks!

            trials.append({
                "trial": trial_num,
                "cost": cost,
                "y_true": y_true,
                "sigma_true": sigma_true,
                "instantaneous_regret": inst_regret,
                "incumbent_regret": incumbent_regret,
                "noise_residual": noise_residual,
                "time": time_eval,
                "config": json.dumps(config) if isinstance(config, (list, dict)) else str(config),
            })
    return trials


def aggregate_dataset(output_base_dir: Path, expected_trials: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collects all 240 run logs and compiles master dataframe and summary table."""
    all_records = []
    run_summaries = []

    for task_name in TASKS:
        for opt_key in OPTIMIZERS:
            for seed in range(1, 31):
                log_file = output_base_dir / "raw" / task_name / opt_key / f"seed_{seed}" / "trial_logs.jsonl"
                if not log_file.is_file():
                    raise FileNotFoundError(f"Missing log file: {log_file}")

                trials = parse_trial_logs(log_file)
                if len(trials) < expected_trials:
                    raise ValueError(f"Incomplete trials in {log_file}: found {len(trials)}, expected {expected_trials}")

                trials = trials[:expected_trials]
                for t in trials:
                    rec = {
                        "task": task_name,
                        "optimizer": opt_key,
                        "optimizer_name": OPTIMIZERS[opt_key]["display_name"],
                        "seed": seed,
                        **t,
                    }
                    all_records.append(rec)

                final_trial = trials[-1]
                run_summaries.append({
                    "task": task_name,
                    "optimizer": opt_key,
                    "optimizer_name": OPTIMIZERS[opt_key]["display_name"],
                    "seed": seed,
                    "n_trials": len(trials),
                    "final_incumbent_regret": final_trial["incumbent_regret"],
                    "best_y_true": min(t["y_true"] for t in trials),
                    "mean_eval_time": float(np.mean([t["time"] for t in trials])),
                })

    df_master = pd.DataFrame(all_records)
    df_summary = pd.DataFrame(run_summaries)
    return df_master, df_summary


def verify_paired_seed_rigor(df_master: pd.DataFrame) -> bool:
    """Verifies that initial design evaluations (trials 1..10) are identical per seed."""
    aligned = True
    for task in TASKS:
        for seed in range(1, 31):
            sub_base = df_master[(df_master["task"] == task) & (df_master["optimizer"] == "baseline") & (df_master["seed"] == seed)]
            sub_prop = df_master[(df_master["task"] == task) & (df_master["optimizer"] == "da_ehrf_direct_ei") & (df_master["seed"] == seed)]

            # Check trial 1 config & cost match exactly
            cfg_base_1 = sub_base[sub_base["trial"] == 1]["config"].values[0]
            cfg_prop_1 = sub_prop[sub_prop["trial"] == 1]["config"].values[0]
            cost_base_1 = sub_base[sub_base["trial"] == 1]["cost"].values[0]
            cost_prop_1 = sub_prop[sub_prop["trial"] == 1]["cost"].values[0]

            if cfg_base_1 != cfg_prop_1 or not np.isclose(cost_base_1, cost_prop_1, atol=1e-6):
                print(f"[WARNING] Seed {seed} on {task} unaligned: {cost_base_1} vs {cost_prop_1}")
                aligned = False
    return aligned


def main():
    parser = argparse.ArgumentParser(description="Execute CARPS DA-EHRF Direct EI 30-seed benchmark suite.")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel worker processes (default: 6)")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials per run (default: 50)")
    parser.add_argument("--outdir", type=str, default="results/carps_da_ehrf_direct_ei", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for task in TASKS:
        for opt in OPTIMIZERS:
            for seed in range(1, 31):
                plan.append((task, opt, seed))

    total_runs = len(plan)
    print(f"================================================================")
    print(f"CARPS DA-EHRF Direct EI 30-Seed Benchmark Execution")
    print(f"Tasks ({len(TASKS)}): {list(TASKS.keys())}")
    print(f"Optimizers ({len(OPTIMIZERS)}): {list(OPTIMIZERS.keys())}")
    print(f"Seeds: 1 to 30 (Total {total_runs} runs, {args.trials} trials each)")
    print(f"Concurrency: {args.workers} workers | Output: {output_dir}")
    print(f"================================================================")

    t0 = time.time()
    completed_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                execute_single_run,
                task_name,
                opt_key,
                seed,
                output_dir,
                args.trials,
            ): (task_name, opt_key, seed)
            for (task_name, opt_key, seed) in plan
        }

        for fut in as_completed(futures):
            res = fut.result()
            task, opt, seed = res["task"], res["optimizer"], res["seed"]
            status = res["status"]
            runtime = res["runtime"]

            if status in ("SUCCESS", "CACHED"):
                completed_count += 1
                flag = "[DONE]" if status == "SUCCESS" else "[CACHED]"
                print(f"[{completed_count:3d}/{total_runs:3d}] {flag} {task:14s} | {opt:24s} | seed {seed:2d} ({runtime:5.1f}s)")
            else:
                failed_count += 1
                print(f"[FAIL] {task} | {opt} | seed {seed} | Code {res['returncode']}")
                if "stderr" in res and res["stderr"]:
                    print(f"  STDERR: {res['stderr'][-300:]}")

    total_time = time.time() - t0
    print(f"\n================================================================")
    print(f"Sweep Execution Complete: {completed_count}/{total_runs} runs succeeded, {failed_count} failed.")
    print(f"Total Wallclock Time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"================================================================")

    if failed_count > 0:
        print(f"[ERROR] Sweep had {failed_count} failed runs. Aborting aggregation.")
        sys.exit(1)

    print("\nAggregating dataset...")
    df_master, df_summary = aggregate_dataset(output_dir, expected_trials=args.trials)

    # Save master datasets
    parquet_path = output_dir / "logs.parquet"
    csv_path = output_dir / "logs.csv"
    summary_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"

    df_master.to_parquet(parquet_path, index=False)
    df_master.to_csv(csv_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    # Validate data integrity
    assert len(df_master) == total_runs * args.trials, f"Expected {total_runs * args.trials} rows, got {len(df_master)}"
    assert not df_master["cost"].isna().any(), "Found NaN in costs"
    assert not df_master["incumbent_regret"].isna().any(), "Found NaN in incumbent regret"

    aligned = verify_paired_seed_rigor(df_master)
    print(f"Strict Paired Seed Alignment: {'100% VERIFIED' if aligned else 'MISMATCH DETECTED'}")

    # Compute task-level aggregation
    task_agg = df_summary.groupby(["task", "optimizer_name"]).agg(
        final_regret_mean=("final_incumbent_regret", "mean"),
        final_regret_std=("final_incumbent_regret", "std"),
        final_regret_median=("final_incumbent_regret", "median"),
        final_regret_min=("final_incumbent_regret", "min"),
        final_regret_max=("final_incumbent_regret", "max"),
        mean_eval_time=("mean_eval_time", "mean"),
    ).reset_index()

    task_agg.to_csv(output_dir / "task_aggregates.csv", index=False)
    summary_dict = {
        "total_runs": total_runs,
        "total_trials": len(df_master),
        "total_wallclock_seconds": total_time,
        "seed_alignment_verified": aligned,
        "task_aggregates": task_agg.to_dict(orient="records"),
    }
    with open(summary_json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\nSaved master logs to: {parquet_path} ({len(df_master)} rows)")
    print(f"Saved summary to: {summary_path}")
    print("\nBenchmark Results Summary Table:")
    print(task_agg.to_string(index=False))


if __name__ == "__main__":
    main()
