#!/usr/bin/env python3
"""Statistical Analysis Suite for CARP-S BBsubset Held-Out Test Evaluation Suite.

Compares Tuned Proximity LCB (SMAC20_ProximityLCB_tuned) vs. SMAC3 Baseline (SMAC3_HPOFacade_lcb)
across 20 held-out test tasks and 30 seeds.

Computes:
1. Paired Wilcoxon signed-rank test across seeds per task.
2. Holm-Bonferroni family-wise error rate correction (alpha=0.05).
3. Win / Tie / Loss summary table (at seed-level and task-level).
4. Normalized regret comparison between Proximity LCB and SMAC3.
5. Exports comprehensive GitHub-flavored Markdown and CSV scorecards.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KNOWN_TASK_DIMS: Dict[str, int] = {
    "svm_12": 2,
    "lcbench": 7,
    "glmnet": 3,
    "ranger": 8,
    "rpart": 5,
    "rbv2_svm": 6,
    "xgboost": 14,
}


def get_task_dimension(task_str: str) -> int:
    """Infers problem dimensionality from task name or CARP-S path."""
    m = re.search(r"bbob[/_](\d+)", task_str)
    if m:
        return int(m.group(1))
    for k, v in KNOWN_TASK_DIMS.items():
        if k in task_str:
            return v
    return 0


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


def apply_holm_bonferroni(p_vals: Sequence[float]) -> List[float]:
    """Computes Holm-Bonferroni step-down adjusted p-values ensuring FWER control and monotonicity.

    Args:
        p_vals: List or array of unadjusted p-values.

    Returns:
        List of Holm-Bonferroni adjusted p-values in the original input order.
    """
    p = np.array(p_vals, dtype=float)
    n = len(p)
    if n == 0:
        return []

    # Replace any nan with 1.0
    p = np.where(np.isnan(p), 1.0, p)

    # Sort p-values ascending
    sort_idx = np.argsort(p)
    sorted_p = p[sort_idx]

    # Step-down adjustment: p_adj_i = (n - i) * p_i
    adj_sorted = np.zeros(n, dtype=float)
    for i in range(n):
        adj_sorted[i] = min(1.0, sorted_p[i] * (n - i))

    # Enforce monotonicity: p_adj_(i+1) >= p_adj_i
    for i in range(1, n):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])

    # Invert sorting back to original order
    adj_original = np.zeros(n, dtype=float)
    adj_original[sort_idx] = adj_sorted
    return [float(val) for val in adj_original]


def find_input_file(input_file: str) -> str:
    """Resolves input file path with fallback candidates."""
    input_path = Path(input_file)
    if input_path.exists():
        return str(input_path)

    candidates = [
        input_path.parent / "logs.parquet",
        input_path.parent / "logs_normalized.parquet",
        input_path.parent / "logs.csv",
        Path("results/sweep_bbsubset_test_proximity/logs.parquet"),
        Path("results/sweep_bbsubset_test_proximity/logs.csv"),
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


def compute_bbsubset_test_proximity_analysis(
    input_file: str = "results/sweep_bbsubset_test_proximity/logs.parquet",
    output_dir: str = "results/sweep_bbsubset_test_proximity",
    proposed_id: str = "SMAC20_ProximityLCB_tuned",
    baseline_id: str = "SMAC3_HPOFacade_lcb",
    alpha: float = 0.05,
    cost_col: str = "trial_value__cost_inc",
    tie_tolerance: float = 1e-6,
    min_dim: Optional[int] = None,
    out_prefix: str = "test_proximity_scorecard",
) -> Dict[str, Any]:
    """Computes full test-set statistical analysis, Holm-Bonferroni correction, and scorecards."""
    os.makedirs(output_dir, exist_ok=True)
    resolved_input = find_input_file(input_file)
    df = load_logs_dataframe(resolved_input)

    # Standardize column names
    trial_col = None
    for candidate in ["n_trials", "n_function_calls", "trial", "trial_number", "trial_idx", "iteration", "step"]:
        if candidate in df.columns:
            trial_col = candidate
            break

    if cost_col not in df.columns:
        for c in ["trial_value__cost_inc", "cost_inc", "trial_value__cost", "cost", "value"]:
            if c in df.columns:
                cost_col = c
                break

    task_col = "task_id" if "task_id" in df.columns else "task"

    if min_dim is not None:
        dims = df[task_col].apply(get_task_dimension)
        df = df[dims >= min_dim].copy()
        if df.empty:
            raise ValueError(f"No tasks match min_dim >= {min_dim}")

    if cost_col in ["trial_value__cost", "cost", "value"] and "trial_value__cost_inc" not in df.columns:
        sort_cols = ["optimizer_id", task_col, "seed"]
        if trial_col:
            sort_cols.append(trial_col)
        df = df.sort_values(by=sort_cols).reset_index(drop=True)
        df["trial_value__cost_inc"] = df.groupby(["optimizer_id", task_col, "seed"])[cost_col].cummin()
        cost_col = "trial_value__cost_inc"

    # Take the latest / incumbent trial per run
    if trial_col is not None:
        idx = df.groupby(["optimizer_id", task_col, "seed"])[trial_col].idxmax()
        latest_df = df.loc[idx].copy()
    else:
        latest_df = df.copy()

    p_df = latest_df[latest_df["optimizer_id"] == proposed_id]
    b_df = latest_df[latest_df["optimizer_id"] == baseline_id]

    if p_df.empty or b_df.empty:
        raise ValueError(
            f"Missing optimizer data: found {p_df['optimizer_id'].nunique()} proposed and {b_df['optimizer_id'].nunique()} baseline rows."
        )

    # Merge paired runs
    merged = pd.merge(
        p_df[[task_col, "seed", cost_col]],
        b_df[[task_col, "seed", cost_col]],
        on=[task_col, "seed"],
        suffixes=("_proposed", "_baseline"),
    )

    if merged.empty:
        raise ValueError(f"No paired runs found between '{proposed_id}' and '{baseline_id}'.")

    # Global paired arrays
    p_all = merged[f"{cost_col}_proposed"].to_numpy(dtype=float)
    b_all = merged[f"{cost_col}_baseline"].to_numpy(dtype=float)
    valid_mask = np.isfinite(p_all) & np.isfinite(b_all)
    p_all_val = p_all[valid_mask]
    b_all_val = b_all[valid_mask]

    diff_all = p_all_val - b_all_val
    overall_wins = int(np.sum(diff_all < -tie_tolerance))
    overall_losses = int(np.sum(diff_all > tie_tolerance))
    overall_ties = int(np.sum(np.abs(diff_all) <= tie_tolerance))

    overall_delta = calculate_cliffs_delta(p_all_val, b_all_val)
    try:
        macro_stat, macro_p = wilcoxon(p_all_val, b_all_val, alternative="two-sided")
    except Exception:
        macro_stat, macro_p = np.nan, 1.0

    # Per-task analysis
    task_rows: List[Dict[str, Any]] = []
    unique_tasks = sorted(merged[task_col].unique())

    for task in unique_tasks:
        sub = merged[merged[task_col] == task]
        gp = sub[f"{cost_col}_proposed"].to_numpy(dtype=float)
        gb = sub[f"{cost_col}_baseline"].to_numpy(dtype=float)
        g_mask = np.isfinite(gp) & np.isfinite(gb)
        gp_v = gp[g_mask]
        gb_v = gb[g_mask]

        n_seeds = len(gp_v)
        diff = gp_v - gb_v
        w = int(np.sum(diff < -tie_tolerance))
        l = int(np.sum(diff > tie_tolerance))
        t = int(np.sum(np.abs(diff) <= tie_tolerance))

        # Wilcoxon per task
        nonzero = diff[np.abs(diff) > tie_tolerance]
        if len(nonzero) >= 5:
            try:
                stat, p_val = wilcoxon(gp_v, gb_v, alternative="two-sided")
            except Exception:
                stat, p_val = np.nan, 1.0
        else:
            stat, p_val = np.nan, 1.0

        delta = calculate_cliffs_delta(gp_v, gb_v)

        # Normalized Regret per task
        all_costs = np.concatenate([gp_v, gb_v])
        if len(all_costs) > 0:
            y_min = float(np.min(all_costs))
            y_max = float(np.max(all_costs))
            spread = y_max - y_min
        else:
            y_min, y_max, spread = 0.0, 0.0, 0.0

        if spread > 1e-12:
            norm_p = float(np.mean((gp_v - y_min) / spread))
            norm_b = float(np.mean((gb_v - y_min) / spread))
        else:
            norm_p = 0.0
            norm_b = 0.0

        mean_p = float(np.mean(gp_v)) if n_seeds > 0 else np.nan
        sem_p = float(np.std(gp_v, ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0
        mean_b = float(np.mean(gb_v)) if n_seeds > 0 else np.nan
        sem_b = float(np.std(gb_v, ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0

        task_rows.append({
            "task": task,
            "n_seeds": n_seeds,
            "mean_proposed": mean_p,
            "sem_proposed": sem_p,
            "mean_baseline": mean_b,
            "sem_baseline": sem_b,
            "norm_regret_proposed": norm_p,
            "norm_regret_baseline": norm_b,
            "mean_diff": float(np.mean(diff)) if n_seeds > 0 else 0.0,
            "wins": w,
            "ties": t,
            "losses": l,
            "cliffs_delta": delta,
            "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
            "p_raw": float(p_val) if np.isfinite(p_val) else 1.0,
        })

    # Apply Holm-Bonferroni correction across tasks
    raw_p_values = [row["p_raw"] for row in task_rows]
    adj_p_values = apply_holm_bonferroni(raw_p_values)

    task_wins = 0
    task_ties = 0
    task_losses = 0

    for i, row in enumerate(task_rows):
        p_h = adj_p_values[i]
        row["p_holm"] = p_h
        row["significant"] = bool(p_h < alpha)

        # Decision classification
        diff = row["mean_proposed"] - row["mean_baseline"]
        if diff < -tie_tolerance:
            row["decision"] = "WIN"
            task_wins += 1
        elif diff > tie_tolerance:
            row["decision"] = "LOSS"
            task_losses += 1
        else:
            row["decision"] = "TIE"
            task_ties += 1

    scorecard_df = pd.DataFrame(task_rows)
    scorecard_df["dim"] = scorecard_df["task"].apply(get_task_dimension)

    # Save CSV scorecard
    csv_path = Path(output_dir) / f"{out_prefix}.csv"
    scorecard_df.to_csv(csv_path, index=False)

    # Task-level Wilcoxon signed-rank test on normalized regret (Demšar, 2006)
    task_norm_p = scorecard_df["norm_regret_proposed"].to_numpy(dtype=float)
    task_norm_b = scorecard_df["norm_regret_baseline"].to_numpy(dtype=float)
    diff_norm = task_norm_p - task_norm_b
    nonzero_norm = int(np.sum(np.abs(diff_norm) > 1e-12))
    if nonzero_norm >= 5:
        try:
            task_w_stat, task_w_p_two = wilcoxon(task_norm_p, task_norm_b, alternative="two-sided")
            _, task_w_p_one = wilcoxon(task_norm_p, task_norm_b, alternative="less")
        except Exception:
            task_w_stat, task_w_p_two, task_w_p_one = np.nan, 1.0, 1.0
    else:
        task_w_stat, task_w_p_two, task_w_p_one = np.nan, 1.0, 1.0

    # Non-parametric effect sizes across tasks
    mean_cliffs_delta = float(scorecard_df["cliffs_delta"].mean())
    task_level_cliffs_delta = calculate_cliffs_delta(task_norm_p, task_norm_b)

    # High-Dimensional Stratification
    strata_definitions = [
        ("bbob_high_d_16", "BBOB High-D (D >= 16)", scorecard_df[scorecard_df["task"].str.contains("bbob") & (scorecard_df["dim"] >= 16)]),
        ("bbob_high_d_8", "BBOB High-D (D >= 8)", scorecard_df[scorecard_df["task"].str.contains("bbob") & (scorecard_df["dim"] >= 8)]),
        ("suite_high_d_8", "All High-D (D >= 8)", scorecard_df[scorecard_df["dim"] >= 8]),
        ("suite_low_d_3", "Low-D (D <= 3)", scorecard_df[scorecard_df["dim"] <= 3]),
    ]

    stratified_results: Dict[str, Dict[str, Any]] = {}
    strat_table_rows = []
    for key, label, sub_df in strata_definitions:
        n_sub = len(sub_df)
        if n_sub == 0:
            continue
        m_p = float(sub_df["norm_regret_proposed"].mean())
        m_b = float(sub_df["norm_regret_baseline"].mean())
        rel_red = float((m_b - m_p) / m_b * 100.0) if m_b > 1e-12 else 0.0
        m_cd = float(sub_df["cliffs_delta"].mean())
        s_p = sub_df["norm_regret_proposed"].to_numpy(dtype=float)
        s_b = sub_df["norm_regret_baseline"].to_numpy(dtype=float)
        s_cd = calculate_cliffs_delta(s_p, s_b)
        s_diff = s_p - s_b
        nz = int(np.sum(np.abs(s_diff) > 1e-12))
        if nz >= 4:
            try:
                s_stat, s_p_two = wilcoxon(s_p, s_b, alternative="two-sided")
                _, s_p_one = wilcoxon(s_p, s_b, alternative="less")
            except Exception:
                s_stat, s_p_two, s_p_one = np.nan, 1.0, 1.0
        else:
            s_stat, s_p_two, s_p_one = np.nan, 1.0, 1.0

        stratified_results[key] = {
            "label": label,
            "n_tasks": n_sub,
            "mean_norm_regret_proposed": m_p,
            "mean_norm_regret_baseline": m_b,
            "relative_reduction_pct": rel_red,
            "mean_cliffs_delta": m_cd,
            "task_level_cliffs_delta": s_cd,
            "wilcoxon_stat": float(s_stat) if np.isfinite(s_stat) else np.nan,
            "p_twosided": float(s_p_two),
            "p_onesided": float(s_p_one),
        }
        strat_table_rows.append({
            "Stratum": label,
            "N Tasks": n_sub,
            "Mean Regret Proposed": f"{m_p:.4f}",
            "Mean Regret Baseline": f"{m_b:.4f}",
            "Rel. Reduction": f"{rel_red:+.1f}%",
            "Mean Cliff's Delta": f"{m_cd:+.3f}",
            "Wilcoxon p (1-sided)": f"{s_p_one:.4f}",
            "Wilcoxon p (2-sided)": f"{s_p_two:.4f}",
        })

    strat_df = pd.DataFrame(strat_table_rows)

    # Build Markdown scorecard
    mean_norm_p = float(scorecard_df["norm_regret_proposed"].mean())
    mean_norm_b = float(scorecard_df["norm_regret_baseline"].mean())
    rel_red_all = float((mean_norm_b - mean_norm_p) / mean_norm_b * 100.0) if mean_norm_b > 1e-12 else 0.0

    sig_task_wins = int(np.sum((scorecard_df["decision"] == "WIN") & scorecard_df["significant"]))
    sig_task_losses = int(np.sum((scorecard_df["decision"] == "LOSS") & scorecard_df["significant"]))
    sig_task_ties = len(task_rows) - sig_task_wins - sig_task_losses

    md_lines = [
        "# CARP-S BBsubset Held-Out Test Evaluation Scorecard",
        "",
        "## Executive Summary",
        "",
        f"- **Proposed Method**: `{proposed_id}` (Proximity LCB, level=0.95)",
        f"- **Baseline Method**: `{baseline_id}` (SMAC3 native LCB: kappa=1.96 / beta=3.8416)",
        f"- **Benchmark Suite**: CARP-S BBsubset Held-Out Test Set ({len(unique_tasks)} tasks, 30 seeds, T=100 budget)",
        f"- **Total Paired Runs**: {len(p_all_val)}",
        f"- **Seed-level Record (W / T / L)**: **{overall_wins} / {overall_ties} / {overall_losses}** ({overall_wins / len(p_all_val) * 100:.1f}% win rate)",
        f"- **Task-level Empirical Record (W / T / L)**: **{task_wins} / {task_ties} / {task_losses}**",
        f"- **Task-level Statistically Significant Record (alpha={alpha}) (W / T / L)**: **{sig_task_wins} / {sig_task_ties} / {sig_task_losses}**",
        f"- **Mean Normalized Regret**: Proposed = **{mean_norm_p:.4f}** vs Baseline = **{mean_norm_b:.4f}** ({rel_red_all:+.1f}% relative reduction)",
        f"- **Task-level Wilcoxon (Demšar) p-value (two-sided)**: `{task_w_p_two:.4f}` ({'Significant (p < 0.05)' if task_w_p_two < alpha else 'Not Significant (p >= 0.05)'})",
        f"- **Task-level Wilcoxon (Demšar) p-value (one-sided, proposed < baseline)**: `{task_w_p_one:.4f}`",
        f"- **Mean Per-Task Cliff's Delta**: `{mean_cliffs_delta:+.4f}` ({'Favors Proposed' if mean_cliffs_delta < 0 else 'Favors Baseline'})",
        f"- **Task-Level Cliff's Delta (on normalized regret)**: `{task_level_cliffs_delta:+.4f}`",
        f"- **Legacy Pooled Wilcoxon p-value (unnormalized scale-sensitive)**: `{macro_p:.4e}`",
        "",
        "## High-Dimensional Stratification Analysis",
        "",
        dataframe_to_markdown(strat_df),
        "",
        "## Per-Task Test Set Scorecard (Holm-Bonferroni FWER alpha = 0.05)",
        "",
    ]

    # Display clean table
    display_df = scorecard_df[[
        "task", "n_seeds", "mean_proposed", "mean_baseline",
        "norm_regret_proposed", "norm_regret_baseline",
        "wins", "ties", "losses", "cliffs_delta", "p_raw", "p_holm", "decision"
    ]].copy()

    display_df["mean_proposed"] = display_df["mean_proposed"].apply(lambda v: f"{v:.4f}")
    display_df["mean_baseline"] = display_df["mean_baseline"].apply(lambda v: f"{v:.4f}")
    display_df["norm_regret_proposed"] = display_df["norm_regret_proposed"].apply(lambda v: f"{v:.4f}")
    display_df["norm_regret_baseline"] = display_df["norm_regret_baseline"].apply(lambda v: f"{v:.4f}")
    display_df["cliffs_delta"] = display_df["cliffs_delta"].apply(lambda v: f"{v:+.3f}")
    display_df["p_raw"] = display_df["p_raw"].apply(lambda v: f"{v:.3e}")
    display_df["p_holm"] = display_df["p_holm"].apply(lambda v: f"{v:.3e}")

    md_lines.append(dataframe_to_markdown(display_df))
    md_lines.append("")

    md_path = Path(output_dir) / f"{out_prefix}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Scorecards generated successfully in {output_dir}:")
    print(f"  - Markdown: {md_path}")
    print(f"  - CSV: {csv_path}")

    return {
        "n_tasks": len(unique_tasks),
        "total_paired_runs": len(p_all_val),
        "overall_wins": overall_wins,
        "overall_ties": overall_ties,
        "overall_losses": overall_losses,
        "task_wins": task_wins,
        "task_ties": task_ties,
        "task_losses": task_losses,
        "mean_norm_regret_proposed": mean_norm_p,
        "mean_norm_regret_baseline": mean_norm_b,
        "overall_cliffs_delta": overall_delta,
        "mean_cliffs_delta": mean_cliffs_delta,
        "task_level_cliffs_delta": task_level_cliffs_delta,
        "task_wilcoxon_stat": float(task_w_stat) if np.isfinite(task_w_stat) else np.nan,
        "task_wilcoxon_p_twosided": float(task_w_p_two),
        "task_wilcoxon_p_onesided": float(task_w_p_one),
        "macro_wilcoxon_p": macro_p,
        "stratified_analysis": stratified_results,
        "scorecard_df": scorecard_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Statistical Analysis & Scorecards for BBsubset Held-Out Test Evaluation"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="results/sweep_bbsubset_test_proximity/logs.parquet",
        help="Path to CARP-S logs parquet or CSV (default: results/sweep_bbsubset_test_proximity/logs.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="results/sweep_bbsubset_test_proximity",
        help="Directory to save scorecards (default: results/sweep_bbsubset_test_proximity)",
    )
    parser.add_argument(
        "--proposed-id",
        default="SMAC20_ProximityLCB_tuned",
        help="Optimizer ID for proposed tuned Proximity LCB (default: SMAC20_ProximityLCB_tuned)",
    )
    parser.add_argument(
        "--baseline-id",
        default="SMAC3_HPOFacade_lcb",
        help="Optimizer ID for reference baseline (default: SMAC3_HPOFacade_lcb)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise error rate significance threshold (default: 0.05)",
    )
    parser.add_argument(
        "--min-dim",
        type=int,
        default=None,
        help="Filter to tasks with minimum dimensionality (e.g. 8 for high-D tasks)",
    )
    parser.add_argument(
        "--out-prefix",
        default="test_proximity_scorecard",
        help="Prefix for output scorecard files (default: test_proximity_scorecard)",
    )

    args = parser.parse_args()
    compute_bbsubset_test_proximity_analysis(
        input_file=args.input,
        output_dir=args.output_dir,
        proposed_id=args.proposed_id,
        baseline_id=args.baseline_id,
        alpha=args.alpha,
        min_dim=args.min_dim,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()
