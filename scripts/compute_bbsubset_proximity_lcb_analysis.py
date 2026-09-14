#!/usr/bin/env python3
"""Statistical Analysis Suite for CARP-S BBsubset Proximity LCB Benchmark Sweep.

Evaluates performance at dual budget horizons:
1. Early/Mid-convergence: t = 50
2. Final/Asymptotic convergence: t = 100

Computes for each checkpoint:
- Paired Wilcoxon signed-rank test (W-statistic, p-value, significance at alpha=0.05).
- Cliff's delta non-parametric effect size.
- Win / Tie / Loss counts (overall and per task).
- Mean incumbent cost and SEM per benchmark task.
- Formats results into GitHub-flavored Markdown, CSV, and LaTeX tables.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame to a clean GitHub-flavored markdown table."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        row_str = [str(val) for val in row.values]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)


def calculate_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cliff's delta non-parametric effect size between two paired vectors.

    delta = (sum(x_i > y_j) - sum(x_i < y_j)) / (n_x * n_y)
    Negative delta means x values are systematically smaller (better for cost minimization).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    greater = 0
    less = 0
    for val_x in x:
        greater += int(np.sum(val_x > y))
        less += int(np.sum(val_x < y))
    return float((greater - less) / (n_x * n_y))


def find_input_file(input_file: str) -> str:
    """Resolves input file path with fallback search candidates."""
    input_path = Path(input_file)
    if input_path.exists():
        return str(input_path)

    candidates = [
        input_path.parent / "logs.parquet",
        input_path.parent / "logs_normalized.parquet",
        input_path.parent / "logs.csv",
        Path("results/bbsubset_proximity_lcb/logs.parquet"),
        Path("results/bbsubset_proximity_lcb/logs.csv"),
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)

    raise FileNotFoundError(
        f"Input file not found: {input_file}. Also checked candidates: {[str(c) for c in candidates]}"
    )


def load_logs_dataframe(input_path: str) -> pd.DataFrame:
    """Loads dataframe from parquet, CSV, or directory of JSON/JSONL files."""
    path = Path(input_path)
    if path.is_file():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)

    # If directory, scan for trial_logs.jsonl or telemetry JSON files
    if path.is_dir():
        trial_files = list(path.glob("**/trial_logs.jsonl"))
        if trial_files:
            rows = []
            for f in trial_files:
                with open(f, "r") as fp:
                    for line in fp:
                        if line.strip():
                            rows.append(json.loads(line))
            if rows:
                return pd.DataFrame(rows)

        # Fallback to telemetry JSON files
        telem_files = list(path.glob("**/telemetry_*.json"))
        if telem_files:
            rows = []
            for f in telem_files:
                with open(f, "r") as fp:
                    data = json.load(fp)
                    trials = data.get("trials", [])
                    opt_id = data.get("optimizer_id", "unknown")
                    task_id = data.get("task_id", "unknown")
                    seed = data.get("seed", 1)
                    for t_idx, tr in enumerate(trials, 1):
                        rows.append({
                            "trial": t_idx,
                            "optimizer_id": opt_id,
                            "task_id": task_id,
                            "seed": seed,
                            "trial_value__cost_inc": tr.get("cost_inc", tr.get("cost", np.nan)),
                        })
            if rows:
                return pd.DataFrame(rows)

    raise ValueError(f"Could not load benchmark logs from: {input_path}")


def compute_checkpoint_stats(
    df_checkpoint: pd.DataFrame,
    checkpoint: int,
    proposed_id: str,
    baseline_id: str,
    alpha_test: float = 0.05,
    cost_col: str = "trial_value__cost_inc",
    tie_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Computes paired statistical test and win/tie/loss counts at a single checkpoint."""
    p_df = df_checkpoint[df_checkpoint["optimizer_id"] == proposed_id]
    b_df = df_checkpoint[df_checkpoint["optimizer_id"] == baseline_id]

    task_col = "task_id" if "task_id" in df_checkpoint.columns else "task"

    # Merge on (task, seed)
    merged = pd.merge(
        p_df[[task_col, "seed", cost_col]],
        b_df[[task_col, "seed", cost_col]],
        on=[task_col, "seed"],
        suffixes=("_proposed", "_baseline"),
    )

    if merged.empty:
        raise ValueError(
            f"No paired runs found between proposed '{proposed_id}' and baseline '{baseline_id}' at t={checkpoint}"
        )

    p_costs = merged[f"{cost_col}_proposed"].to_numpy(dtype=float)
    b_costs = merged[f"{cost_col}_baseline"].to_numpy(dtype=float)

    # Valid finite values
    valid_mask = np.isfinite(p_costs) & np.isfinite(b_costs)
    p_valid = p_costs[valid_mask]
    b_valid = b_costs[valid_mask]

    diffs = p_valid - b_valid
    wins = int(np.sum(diffs < -tie_tolerance))
    losses = int(np.sum(diffs > tie_tolerance))
    ties = int(np.sum(np.abs(diffs) <= tie_tolerance))

    delta = calculate_cliffs_delta(p_valid, b_valid)

    # Paired Wilcoxon Signed-Rank Test
    non_zero_diffs = diffs[np.abs(diffs) > tie_tolerance]
    if len(non_zero_diffs) >= 5:
        try:
            stat, p_val = wilcoxon(p_valid, b_valid, alternative="two-sided")
        except Exception:
            stat, p_val = np.nan, 1.0
    else:
        stat, p_val = np.nan, 1.0

    # Per-task breakdown
    task_records = []
    for task_name, group in merged.groupby(task_col):
        grp_p = group[f"{cost_col}_proposed"].to_numpy(dtype=float)
        grp_b = group[f"{cost_col}_baseline"].to_numpy(dtype=float)
        grp_mask = np.isfinite(grp_p) & np.isfinite(grp_b)
        gp = grp_p[grp_mask]
        gb = grp_b[grp_mask]
        g_diff = gp - gb

        g_wins = int(np.sum(g_diff < -tie_tolerance))
        g_losses = int(np.sum(g_diff > tie_tolerance))
        g_ties = int(np.sum(np.abs(g_diff) <= tie_tolerance))
        g_delta = calculate_cliffs_delta(gp, gb)

        task_records.append({
            "task": task_name,
            "n_seeds": len(gp),
            "mean_proposed": float(np.mean(gp)) if len(gp) > 0 else np.nan,
            "sem_proposed": float(np.std(gp, ddof=1) / np.sqrt(len(gp))) if len(gp) > 1 else 0.0,
            "mean_baseline": float(np.mean(gb)) if len(gb) > 0 else np.nan,
            "sem_baseline": float(np.std(gb, ddof=1) / np.sqrt(len(gb))) if len(gb) > 1 else 0.0,
            "cliffs_delta": g_delta,
            "wins": g_wins,
            "ties": g_ties,
            "losses": g_losses,
        })

    task_summary_df = pd.DataFrame(task_records)

    return {
        "checkpoint": checkpoint,
        "n_pairs": len(p_valid),
        "wilcoxon_stat": stat,
        "wilcoxon_p": p_val,
        "significant": p_val < alpha_test if np.isfinite(p_val) else False,
        "cliffs_delta": delta,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / len(p_valid) if len(p_valid) > 0 else 0.0,
        "task_summary_df": task_summary_df,
    }


def compute_dual_checkpoint_analysis(
    input_file: str = "results/bbsubset_proximity_lcb/logs.parquet",
    output_dir: str = "results/bbsubset_proximity_lcb_analysis",
    checkpoints: Sequence[int] = (50, 100),
    proposed_id: str = "SMAC20_ProximityLCB_k10",
    baseline_id: str = "SMAC3_HPOFacade_lcb",
    alpha_test: float = 0.05,
    cost_col: str = "trial_value__cost_inc",
) -> Dict[int, Dict[str, Any]]:
    """Analyzes benchmark results at both t=50 and t=100 checkpoints."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_logs_dataframe(input_file)

    # Standardize column names if needed
    trial_col = "trial"
    if trial_col not in df.columns:
        for c in ["trial_number", "trial_idx", "iteration", "step"]:
            if c in df.columns:
                trial_col = c
                break

    results: Dict[int, Dict[str, Any]] = {}
    unified_summary_rows = []

    for t in checkpoints:
        # Filter trials up to t, take the incumbent or latest cost at trial <= t
        sub_df = df[df[trial_col] <= t].copy()
        # Get the row corresponding to max trial <= t for each run
        task_col = "task_id" if "task_id" in sub_df.columns else "task"
        idx = sub_df.groupby(["optimizer_id", task_col, "seed"])[trial_col].idxmax()
        latest_df = sub_df.loc[idx]

        stats = compute_checkpoint_stats(
            latest_df,
            checkpoint=t,
            proposed_id=proposed_id,
            baseline_id=baseline_id,
            alpha_test=alpha_test,
            cost_col=cost_col,
        )
        results[t] = stats

        # Write per-checkpoint markdown report
        md_lines = [
            f"# Benchmark Evaluation at Checkpoint Horizon t = {t} Trials",
            "",
            f"**Proposed Method**: `{proposed_id}`  ",
            f"**Baseline Method**: `{baseline_id}`  ",
            f"**Significance Level (alpha)**: `{alpha_test}`  ",
            f"**Total Paired Runs**: `{stats['n_pairs']}`  ",
            f"**Overall Record (W / T / L)**: `{stats['wins']} / {stats['ties']} / {stats['losses']}` ({stats['win_rate']*100:.1f}% win rate)  ",
            f"**Wilcoxon Signed-Rank p-value**: `{stats['wilcoxon_p']:.4e}` ({'Significant (p < 0.05)' if stats['significant'] else 'Not Significant'})  ",
            f"**Cliff's delta**: `{stats['cliffs_delta']:+.4f}`  ",
            "",
            "## Per-Task Breakdown",
            "",
        ]

        task_df = stats["task_summary_df"]
        md_lines.append(dataframe_to_markdown(task_df))

        report_md_path = Path(output_dir) / f"proximity_lcb_analysis_t{t}.md"
        report_csv_path = Path(output_dir) / f"proximity_lcb_per_task_t{t}.csv"
        report_md_path.write_text("\n".join(md_lines))
        task_df.to_csv(report_csv_path, index=False)

        unified_summary_rows.append({
            "Checkpoint": f"t = {t}",
            "Pairs": stats["n_pairs"],
            "Wins": stats["wins"],
            "Ties": stats["ties"],
            "Losses": stats["losses"],
            "Win Rate": f"{stats['win_rate']*100:.1f}%",
            "Wilcoxon p-val": f"{stats['wilcoxon_p']:.4e}",
            "Significance": "p < 0.05" if stats["significant"] else "n.s.",
            "Cliff's delta": f"{stats['cliffs_delta']:+.4f}",
        })

    # Generate unified dual-checkpoint summary
    unified_df = pd.DataFrame(unified_summary_rows)
    unified_md = [
        "# Head-to-Head Benchmark: SMAC3 LCB Baseline vs. Proximity Lower Bound LCB",
        "",
        "## Dual-Checkpoint Comparison Summary",
        "",
        dataframe_to_markdown(unified_df),
        "",
    ]
    summary_path = Path(output_dir) / "proximity_lcb_dual_checkpoint_summary.md"
    summary_path.write_text("\n".join(unified_md))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute dual-checkpoint (t=50 and t=100) statistical analysis for Proximity LCB sweep."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="results/bbsubset_proximity_lcb/logs.parquet",
        help="Input parquet, CSV, or directory of run logs (default: results/bbsubset_proximity_lcb/logs.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="results/bbsubset_proximity_lcb_analysis",
        help="Output directory for reports (default: results/bbsubset_proximity_lcb_analysis)",
    )
    parser.add_argument(
        "--proposed-id",
        default="SMAC20_ProximityLCB_k10",
        help="Optimizer ID of proposed method (default: SMAC20_ProximityLCB_k10)",
    )
    parser.add_argument(
        "--baseline-id",
        default="SMAC3_HPOFacade_lcb",
        help="Optimizer ID of baseline method (default: SMAC3_HPOFacade_lcb)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for hypothesis testing (default: 0.05)",
    )
    args = parser.parse_args()

    results = compute_dual_checkpoint_analysis(
        input_file=args.input,
        output_dir=args.output_dir,
        checkpoints=[50, 100],
        proposed_id=args.proposed_id,
        baseline_id=args.baseline_id,
        alpha_test=args.alpha,
    )

    print("Dual-checkpoint analysis complete:")
    for t, res in results.items():
        print(f"  t={t}: W/T/L={res['wins']}/{res['ties']}/{res['losses']}, p={res['wilcoxon_p']:.4e}, delta={res['cliffs_delta']:+.4f}")


if __name__ == "__main__":
    main()
