#!/usr/bin/env python3
"""
Local Execution Harness & Single-Task Runner for Epistemic OOD Benchmark Sweep.

Supports:
1. Single-task CLI and programmatic execution (compatible with SLURM array tasks).
2. Multi-core parallel execution using ProcessPoolExecutor for local benchmarking.
3. Fast dry-run / smoke testing modes.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

from data_generator import generate_data
from synthetic_functions import (
    get_all_normal_functions,
    get_special_functions,
)
from ep_extractors import UQExtractorRegistry
from metrics import (
    calculate_roc_metrics,
    calculate_aupr,
    calculate_aurc_exact,
    calculate_jensen_shannon_divergence,
    calculate_mutual_information,
    calculate_nlpd,
)

DEFAULT_OUTPUT_DIR = "results/epistemic_ood_sweep"


def get_all_benchmark_functions() -> Dict[str, Any]:
    """Returns combined master dictionary of all 51 benchmark functions."""
    return {**get_all_normal_functions(), **get_special_functions()}


def run_epistemic_ood_task(
    func_name: str,
    gap_type: str = "empty",
    approach: str = "standard_disagreement",
    seed: int = 1,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    n_estimators: int = 100,
    noise_std: float = 0.1,
    id_split: float = 0.7,
) -> Dict[str, Any]:
    """
    Executes a single epistemic OOD detection benchmark evaluation task.

    Parameters:
        func_name: Name of synthetic benchmark function (out of 51).
        gap_type: 'empty' or 'sparse'.
        approach: 'standard_disagreement' or 'distance_evidential'.
        seed: Random seed for reproducibility.
        output_dir: Optional path to save result JSON.
        dry_run: If True, returns mock evaluation result without heavy RF fitting.
        n_estimators: Number of trees for Random Forest Regressor.
        noise_std: Gaussian observation noise standard deviation.
        id_split: Fraction of test points reserved for In-Distribution (ID).

    Returns:
        Dictionary containing benchmark configuration, timings, and all 8 evaluation metrics.
    """
    funcs = get_all_benchmark_functions()
    if func_name not in funcs:
        raise ValueError(f"Unknown benchmark function '{func_name}'. Available: {list(funcs.keys())}")

    cfg = funcs[func_name]
    dim = cfg.get("dim", 1)

    t_start = time.time()

    if dry_run:
        results = {
            "func_name": func_name,
            "dim": dim,
            "gap_type": gap_type,
            "approach": approach,
            "seed": seed,
            "auroc": 0.85,
            "fpr95": 0.20,
            "aupr": 0.80,
            "spearman": 0.50,
            "aurc": 0.15,
            "oracle_aurc": 0.05,
            "jsd": 0.40,
            "mi": 0.35,
            "nlpd": 1.00,
            "brier": 0.10,
            "n_train": 100,
            "n_test": 50,
            "elapsed_seconds": 0.001,
            "status": "success",
            "dry_run": True,
        }
    else:
        # 1. Generate Dataset (ID Train, ID Test, OOD Test)
        X_train, y_train, X_test, y_test, y_true_binary = generate_data(
            func_dict=funcs,
            func_name=func_name,
            seed=seed,
            gap_type=gap_type,
            noise_std=noise_std,
            id_split=id_split,
        )

        # 2. Fit Random Forest Regressor Surrogate with OOB scoring enabled
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            oob_score=True,
            random_state=seed,
            min_samples_leaf=5,
            n_jobs=1,
        )
        rf.fit(X_train, y_train)

        # Predict mean and test residuals
        y_pred = rf.predict(X_test)
        abs_error = np.abs(y_test - y_pred)

        # 3. Extract Epistemic Uncertainty Signal U(x)
        extractor = UQExtractorRegistry.get(approach, rf)
        extractor.fit(X_train, y_train)
        unc_signal = extractor.extract_epistemic_signal(X_test)

        # Ensure numerical safety (no NaN or Inf)
        unc_signal = np.nan_to_num(unc_signal, nan=0.0, posinf=1e6, neginf=0.0)

        # 4. Compute All Evaluation Metrics
        # A. AUROC & FPR@95
        auroc, fpr95 = calculate_roc_metrics(y_true_binary, unc_signal)

        # B. AUPR
        aupr = calculate_aupr(y_true_binary, unc_signal)

        # C. Spearman Rank Correlation
        spearman_corr, _ = spearmanr(unc_signal, abs_error)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0

        # D. AURC & Oracle AURC
        aurc = calculate_aurc_exact(unc_signal, y_pred, y_test, loss_type="MAE")
        oracle_aurc = calculate_aurc_exact(abs_error, y_pred, y_test, loss_type="MAE")

        # E. Information-Theoretic Metrics (JSD & MI)
        jsd = calculate_jensen_shannon_divergence(unc_signal, y_true_binary)
        mi = calculate_mutual_information(unc_signal, y_true_binary)

        # F. Brier Score & NLPD
        nlpd = calculate_nlpd(y_test, y_pred, unc_signal)
        max_unc = np.max(unc_signal)
        brier_score = float(np.mean((y_true_binary - unc_signal / (max_unc + 1e-8)) ** 2))

        elapsed = time.time() - t_start

        results = {
            "func_name": func_name,
            "dim": dim,
            "gap_type": gap_type,
            "approach": approach,
            "seed": seed,
            "auroc": float(auroc),
            "fpr95": float(fpr95),
            "aupr": float(aupr),
            "spearman": float(spearman_corr),
            "aurc": float(aurc),
            "oracle_aurc": float(oracle_aurc),
            "jsd": float(jsd),
            "mi": float(mi),
            "nlpd": float(nlpd),
            "brier": float(brier_score),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "elapsed_seconds": float(elapsed),
            "status": "success",
            "dry_run": False,
        }

    # Save to disk if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(
            output_dir,
            f"ood_result_{func_name}_{gap_type}_{approach}_seed{seed}.json",
        )
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


def _task_worker(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Helper worker function for ProcessPoolExecutor."""
    try:
        return run_epistemic_ood_task(**task_dict)
    except Exception as e:
        return {
            "func_name": task_dict.get("func_name", "unknown"),
            "gap_type": task_dict.get("gap_type", "unknown"),
            "approach": task_dict.get("approach", "unknown"),
            "seed": task_dict.get("seed", 0),
            "status": "error",
            "error_message": str(e),
        }


def parse_task_command(cmd: str, default_output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    """Parses command string e.g. python scripts/run_epistemic_ood_sweep_local.py --func_name=sin ... into dict."""
    parts = cmd.strip().split()
    task = {"output_dir": default_output_dir}
    for p in parts:
        if p.startswith("--func_name="):
            task["func_name"] = p.split("=")[1]
        elif p.startswith("--gap_type="):
            task["gap_type"] = p.split("=")[1]
        elif p.startswith("--approach="):
            task["approach"] = p.split("=")[1]
        elif p.startswith("--seed="):
            task["seed"] = int(p.split("=")[1])
        elif p.startswith("--output_dir="):
            task["output_dir"] = p.split("=")[1]
        elif p == "--dry_run":
            task["dry_run"] = True
    return task


def run_epistemic_ood_sweep_local(
    tasks: Union[List[Dict[str, Any]], List[str], str],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_workers: Optional[int] = None,
    dry_run: bool = False,
    max_tasks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Executes a list of sweep tasks in parallel on local multi-core machine.

    Parameters:
        tasks: List of task dicts, list of command lines, or path to task text file.
        output_dir: Directory where JSON results will be written.
        max_workers: Concurrency level (defaults to CPU count).
        dry_run: Override all tasks to run in dry_run mode.
        max_tasks: Optional limit on number of tasks to execute (useful for testing).

    Returns:
        List of task execution result dictionaries.
    """
    task_dicts: List[Dict[str, Any]] = []

    if isinstance(tasks, str):
        # File path
        if not os.path.isfile(tasks):
            raise FileNotFoundError(f"Task file '{tasks}' not found.")
        with open(tasks, "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        for line in lines:
            t = parse_task_command(line, default_output_dir=output_dir)
            if dry_run:
                t["dry_run"] = True
            task_dicts.append(t)
    elif isinstance(tasks, list):
        for item in tasks:
            if isinstance(item, str):
                t = parse_task_command(item, default_output_dir=output_dir)
                if dry_run:
                    t["dry_run"] = True
                task_dicts.append(t)
            elif isinstance(item, dict):
                t = dict(item)
                if output_dir and "output_dir" not in t:
                    t["output_dir"] = output_dir
                if dry_run:
                    t["dry_run"] = True
                task_dicts.append(t)

    if max_tasks is not None:
        task_dicts = task_dicts[:max_tasks]

    n_tasks = len(task_dicts)
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 16)

    print(f"Starting local epistemic OOD sweep: {n_tasks} tasks across {max_workers} worker processes.")
    results: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_task_worker, t): t for t in task_dicts}
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % 50 == 0 or completed == n_tasks:
                print(f"Progress: [{completed}/{n_tasks}] ({completed / n_tasks * 100:.1f}%) completed.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Epistemic OOD Benchmark Runner")
    # Single-task arguments
    parser.add_argument("--func_name", type=str, help="Function name for single task execution")
    parser.add_argument("--gap_type", type=str, default="empty", choices=["empty", "sparse"])
    parser.add_argument(
        "--approach",
        type=str,
        default="standard_disagreement",
        choices=["standard_disagreement", "distance_evidential"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry_run", action="store_true", help="Execute in dry-run mode without RF fitting")
    parser.add_argument("--n_estimators", type=int, default=100)

    # Multi-task batch arguments
    parser.add_argument("--tasks_file", type=str, help="Path to task file to execute locally")
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--max_tasks", type=int, default=None)

    args = parser.parse_args()

    if args.tasks_file:
        run_epistemic_ood_sweep_local(
            tasks=args.tasks_file,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
            max_tasks=args.max_tasks,
        )
    elif args.func_name:
        res = run_epistemic_ood_task(
            func_name=args.func_name,
            gap_type=args.gap_type,
            approach=args.approach,
            seed=args.seed,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            n_estimators=args.n_estimators,
        )
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
