#!/usr/bin/env python3
"""Compute Task Median of Paired Differences and One-Sided Wilcoxon Signed-Rank Test.

Methodology:
1. For each task, extract all paired seed runs (e.g. seeds 1-30).
2. For each seed, calculate the performance difference (proposed - baseline).
   To prevent tasks with large dynamic ranges (e.g. 10^5) from drowning out
   tasks with small ranges (e.g. 10^-2), differences are min-max normalized per task:
       delta_{k, s} = (cost_prop - cost_base) / (y_max^{(k)} - y_min^{(k)}) in [-1, 1]
3. For each task, compute the median of these 30 paired differences:
       m_k = median_{s}(delta_{k, s})
4. Run a one-sided Wilcoxon signed-rank test across the vector of task medians:
       wilcoxon([m_1, ..., m_K], alternative='less')
   Testing H_1: proposed significantly reduces regret/cost compared to baseline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_SWEEPS = {
    "bbsubset_test_proximity": "results/sweep_bbsubset_test_proximity/logs.csv",
    "bbsubset_test_proximity_vs_ei": "results/sweep_bbsubset_test_proximity_vs_ei/logs.csv",
    "bbob_highdim_proximity": "results/sweep_bbob_highdim_proximity/logs.csv",
    "bbob_highdim_proximity_vs_ei": "results/sweep_bbob_highdim_proximity_vs_ei/logs.csv",
}


def auto_detect_optimizers(unique_optimizers: Sequence[str]) -> Tuple[str, str]:
    """Auto-detects proposed (Proximity) and baseline optimizers from unique IDs."""
    proposed: Optional[str] = None
    baseline: Optional[str] = None

    for opt in unique_optimizers:
        opt_str = str(opt)
        if "proximity" in opt_str.lower():
            proposed = opt_str
        elif "hpofacade" in opt_str.lower() or "customuncertainty" in opt_str.lower() or "smac3" in opt_str.lower():
            baseline = opt_str

    if proposed is None or baseline is None:
        if len(unique_optimizers) >= 2:
            return str(unique_optimizers[0]), str(unique_optimizers[1])
        raise ValueError(f"Could not auto-detect 2 distinct optimizers from: {unique_optimizers}")

    return proposed, baseline


def compute_task_median_differences(
    df: pd.DataFrame,
    proposed_id: str,
    baseline_id: str,
    normalize: bool = True,
    cost_col: Optional[str] = None,
    tie_tolerance: float = 1e-9,
) -> Tuple[Dict[str, float], Dict[str, List[float]], pd.DataFrame]:
    """Computes the median of paired seed differences per task.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory or trial log dataframe.
    proposed_id : str
        Optimizer ID of proposed method.
    baseline_id : str
        Optimizer ID of baseline method.
    normalize : bool
        Whether to min-max normalize costs per task into [0, 1] before differencing.
    cost_col : str, optional
        Cost column name. If None, auto-detects incumbent or cost column.
    tie_tolerance : float
        Threshold below which a difference is considered a tie.

    Returns
    -------
    task_medians : Dict[str, float]
        Mapping of task_id to its median paired seed difference.
    seed_diffs : Dict[str, List[float]]
        Mapping of task_id to list of raw/normalized differences across seeds.
    merged : pd.DataFrame
        Joined dataframe of paired runs.
    """
    task_col = "task_id" if "task_id" in df.columns else ("task" if "task" in df.columns else None)
    if task_col is None:
        raise ValueError("Neither 'task_id' nor 'task' found in DataFrame columns.")

    if cost_col is None:
        for c in ["trial_value__cost_inc", "cost_inc", "trial_value__cost", "cost", "value"]:
            if c in df.columns:
                cost_col = c
                break
    if cost_col is None:
        raise ValueError("Could not find a valid cost column in DataFrame.")

    # Sort and reduce to the final incumbent per (task, optimizer, seed)
    trial_col = None
    for c in ["n_trials", "n_function_calls", "trial", "trial_number", "trial_idx", "step"]:
        if c in df.columns:
            trial_col = c
            break

    grouper = [task_col, "optimizer_id", "seed"]
    if trial_col:
        reduced_df = df.sort_values(by=grouper + [trial_col]).drop_duplicates(subset=grouper, keep="last")
    else:
        reduced_df = df.drop_duplicates(subset=grouper, keep="last")

    p_df = reduced_df[reduced_df["optimizer_id"] == proposed_id]
    b_df = reduced_df[reduced_df["optimizer_id"] == baseline_id]

    if p_df.empty:
        raise ValueError(f"No records found for proposed optimizer: '{proposed_id}'")
    if b_df.empty:
        raise ValueError(f"No records found for baseline optimizer: '{baseline_id}'")

    merged = pd.merge(
        p_df[[task_col, "seed", cost_col]],
        b_df[[task_col, "seed", cost_col]],
        on=[task_col, "seed"],
        suffixes=("_proposed", "_baseline"),
    )

    if merged.empty:
        raise ValueError("No strictly paired seeds found between proposed and baseline.")

    task_medians: Dict[str, float] = {}
    seed_diffs: Dict[str, List[float]] = {}

    for task, group in merged.groupby(task_col):
        c_p = group[f"{cost_col}_proposed"].to_numpy(dtype=float)
        c_b = group[f"{cost_col}_baseline"].to_numpy(dtype=float)
        valid = np.isfinite(c_p) & np.isfinite(c_b)
        c_p, c_b = c_p[valid], c_b[valid]

        if len(c_p) == 0:
            continue

        if normalize:
            all_vals = np.concatenate([c_p, c_b])
            y_min = float(np.min(all_vals))
            y_max = float(np.max(all_vals))
            spread = y_max - y_min
            if spread > 1e-12:
                diffs = (c_p - c_b) / spread
            else:
                diffs = np.zeros_like(c_p)
        else:
            diffs = c_p - c_b

        # Filter negligible differences near floating zero
        diffs = np.where(np.abs(diffs) < tie_tolerance, 0.0, diffs)

        task_medians[str(task)] = float(np.median(diffs))
        seed_diffs[str(task)] = diffs.tolist()

    return task_medians, seed_diffs, merged


def run_one_sided_wilcoxon(task_medians: Sequence[float] | np.ndarray) -> Tuple[float, float]:
    """Executes a one-sided Wilcoxon signed-rank test on task medians.

    H_0: median difference >= 0 (proposed does not reduce cost/regret)
    H_1: median difference < 0  (proposed reduces cost/regret compared to baseline)

    Returns
    -------
    w_stat : float
        Wilcoxon test statistic.
    p_value : float
        One-sided p-value.
    """
    medians = np.asarray(task_medians, dtype=float)
    non_zero = medians[np.abs(medians) > 1e-12]

    if len(non_zero) == 0:
        return 0.0, 1.0

    try:
        res = wilcoxon(non_zero, alternative="less")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def process_sweep_logs(
    log_path: str,
    proposed_id: Optional[str] = None,
    baseline_id: Optional[str] = None,
    normalize: bool = True,
    cost_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Processes a single sweep log file or directory and computes statistical summary."""
    p = Path(log_path)
    if p.is_dir():
        csv_p = p / "logs.csv"
        parquet_p = p / "logs.parquet"
        if csv_p.exists():
            target = csv_p
        elif parquet_p.exists():
            target = parquet_p
        else:
            raise FileNotFoundError(f"Neither logs.csv nor logs.parquet found in {log_path}")
    else:
        target = p

    if target.suffix == ".parquet":
        df = pd.read_parquet(target)
    else:
        df = pd.read_csv(target)

    # Auto-detect optimizers if not supplied
    if proposed_id is None or baseline_id is None:
        opts = df["optimizer_id"].dropna().unique().tolist()
        detected_p, detected_b = auto_detect_optimizers(opts)
        proposed_id = proposed_id or detected_p
        baseline_id = baseline_id or detected_b

    task_medians, seed_diffs, merged = compute_task_median_differences(
        df=df,
        proposed_id=proposed_id,
        baseline_id=baseline_id,
        normalize=normalize,
        cost_col=cost_col,
    )

    medians_array = np.array(list(task_medians.values()))
    w_stat, p_val = run_one_sided_wilcoxon(medians_array)

    # Overall task-level win/loss/tie based on task median
    w_tasks = int(np.sum(medians_array < -1e-9))
    l_tasks = int(np.sum(medians_array > 1e-9))
    t_tasks = int(np.sum(np.abs(medians_array) <= 1e-9))

    # Overall run-level counts
    all_diffs = []
    for diffs in seed_diffs.values():
        all_diffs.extend(diffs)
    all_diffs_arr = np.array(all_diffs)
    w_runs = int(np.sum(all_diffs_arr < -1e-9))
    l_runs = int(np.sum(all_diffs_arr > 1e-9))
    t_runs = int(np.sum(np.abs(all_diffs_arr) <= 1e-9))

    return {
        "log_path": str(target),
        "proposed_id": proposed_id,
        "baseline_id": baseline_id,
        "n_tasks": len(task_medians),
        "n_paired_runs": len(merged),
        "normalized": normalize,
        "task_wins": w_tasks,
        "task_losses": l_tasks,
        "task_ties": t_tasks,
        "task_win_rate": w_tasks / max(len(task_medians), 1),
        "run_wins": w_runs,
        "run_losses": l_runs,
        "run_ties": t_runs,
        "run_win_rate": w_runs / max(len(all_diffs_arr), 1),
        "mean_task_median": float(np.mean(medians_array)) if len(medians_array) else 0.0,
        "median_task_median": float(np.median(medians_array)) if len(medians_array) else 0.0,
        "w_stat": w_stat,
        "p_value": p_val,
        "task_medians": task_medians,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute task median differences and one-sided Wilcoxon signed-rank test across sweeps."
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Named preset or path to logs.csv. Supported presets: " + ", ".join(DEFAULT_SWEEPS.keys()),
    )
    parser.add_argument(
        "--all-sweeps",
        action="store_true",
        help="Evaluate all 4 canonical sweeps and display comparative scorecard.",
    )
    parser.add_argument(
        "--raw-cost",
        action="store_true",
        help="Use raw cost differences instead of per-task min-max normalization.",
    )
    parser.add_argument(
        "--proposed-id",
        type=str,
        default=None,
        help="Override proposed optimizer ID.",
    )
    parser.add_argument(
        "--baseline-id",
        type=str,
        default=None,
        help="Override baseline optimizer ID.",
    )
    return parser.parse_args()


def format_p_value(p: float) -> str:
    if p < 1e-4:
        return f"{p:.4e}"
    return f"{p:.4f}"


def main() -> None:
    args = parse_args()
    normalize = not args.raw_cost

    targets: List[Tuple[str, str]] = []

    if args.all_sweeps or args.sweep is None:
        for name, path in DEFAULT_SWEEPS.items():
            if os.path.exists(path):
                targets.append((name, path))
            else:
                print(f"[WARNING] Skipping {name}: {path} not found.")
    else:
        path = DEFAULT_SWEEPS.get(args.sweep, args.sweep)
        targets.append((args.sweep, path))

    if not targets:
        print("No valid sweep log files found to evaluate.")
        sys.exit(1)

    results_table = []
    for name, path in targets:
        print(f"\nProcessing sweep: {name} ({path})...")
        res = process_sweep_logs(
            path,
            proposed_id=args.proposed_id,
            baseline_id=args.baseline_id,
            normalize=normalize,
        )
        results_table.append((name, res))

    print("\n" + "=" * 115)
    metric_label = "Per-Task Min-Max Normalized Regret" if normalize else "Raw Cost"
    print(f"SWEEP TASK-MEDIAN WILCOXON ANALYSIS (Metric: {metric_label}, One-Sided: Proposed < Baseline)")
    print("=" * 115)
    header = (
        f"{'Sweep Name':<32} | {'Tasks':<5} | {'Runs':<6} | {'Task Record (W/L/T)':<19} | "
        f"{'Run WR':<7} | {'Mean Task Med':<13} | {'W-Stat':<8} | {'p-value (one-sided)':<18}"
    )
    print(header)
    print("-" * 115)

    for name, r in results_table:
        rec = f"{r['task_wins']} / {r['task_losses']} / {r['task_ties']}"
        sig = " ***" if r['p_value'] < 0.001 else (" **" if r['p_value'] < 0.01 else (" *" if r['p_value'] < 0.05 else ""))
        p_str = f"{format_p_value(r['p_value'])}{sig}"
        row = (
            f"{name:<32} | {r['n_tasks']:<5} | {r['n_paired_runs']:<6} | {rec:<19} | "
            f"{r['run_win_rate']*100:>5.1f}% | {r['mean_task_median']:>+13.4f} | {r['w_stat']:>8.1f} | {p_str:<18}"
        )
        print(row)
    print("=" * 115)
    print("Significance markers: * p < 0.05, ** p < 0.01, *** p < 0.001\n")


if __name__ == "__main__":
    main()
