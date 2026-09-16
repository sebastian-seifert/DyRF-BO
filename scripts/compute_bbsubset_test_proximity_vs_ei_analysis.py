#!/usr/bin/env python3
"""Statistical Analysis Suite for CARP-S BBsubset Held-Out Test Evaluation (Proximity LCB vs. SMAC3 EI).

Compares Tuned Proximity LCB (SMAC20_ProximityLCB_tuned) vs. SMAC3 EI Baseline (SMAC3_HPOFacade_ei)
across 20 held-out test tasks and 30 seeds.

Computes:
1. Paired Wilcoxon signed-rank test across seeds per task.
2. Step-down Holm-Bonferroni family-wise error rate correction (alpha=0.05).
3. Scale-invariant normalized regret comparison between Proximity LCB and SMAC3 EI.
4. Demšar task-level Wilcoxon signed-rank test (two-sided and one-sided) on normalized regret.
5. Exact Binomial sign test on task-level win/loss counts.
6. Non-parametric effect sizes (Cliff's delta per-task and across tasks).
7. High-D (D >= 8), Low-D (D < 8), and Domain (BBOB vs Real-World ML) stratified scorecards.
8. Exports comprehensive GitHub-flavored Markdown and CSV scorecards.
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
from scipy.stats import binomtest, wilcoxon

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
    m = re.search(r"bbob[/_](\d+)", str(task_str))
    if m:
        return int(m.group(1))
    for k, v in KNOWN_TASK_DIMS.items():
        if k in str(task_str):
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

    # Step-down adjustment: p_adj_i = min(1.0, (n - i) * p_i)
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


def load_logs_dataframe(input_path: str) -> pd.DataFrame:
    """Loads dataframe from parquet or CSV file."""
    path = Path(input_path)
    if path.is_file():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)

    candidates = [
        path / "logs.parquet",
        path / "logs.csv",
        Path("results/sweep_bbsubset_test_proximity_vs_ei/logs.parquet"),
        Path("results/sweep_bbsubset_test_proximity_vs_ei/logs.csv"),
    ]
    for cand in candidates:
        if cand.exists():
            if cand.suffix == ".parquet":
                return pd.read_parquet(cand)
            return pd.read_csv(cand)

    raise FileNotFoundError(f"Input file not found: {input_path}")


def compute_bbsubset_test_proximity_vs_ei_analysis(
    input_file: str = "results/sweep_bbsubset_test_proximity_vs_ei/logs.csv",
    output_dir: str = "results/sweep_bbsubset_test_proximity_vs_ei/analysis",
    out_prefix: str = "test_proximity_vs_ei_scorecard",
    proposed_id: str = "SMAC20_ProximityLCB_tuned",
    baseline_id: str = "SMAC3_HPOFacade_ei",
    cost_col: Optional[str] = None,
    alpha: float = 0.05,
    tie_tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """Performs rigorous statistical comparison and exports scorecards."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_logs_dataframe(input_file)

    # Standardize column names
    trial_col = None
    for candidate in ["n_trials", "n_function_calls", "trial", "trial_number", "trial_idx", "iteration", "step"]:
        if candidate in df.columns:
            trial_col = candidate
            break

    if cost_col is None or cost_col not in df.columns:
        for c in ["trial_value__cost_inc", "cost_inc", "trial_value__cost", "cost", "value"]:
            if c in df.columns:
                cost_col = c
                break

    task_col = "task_id" if "task_id" in df.columns else "task"

    if cost_col in ["trial_value__cost", "cost", "value"] and "trial_value__cost_inc" not in df.columns:
        sort_cols = ["optimizer_id", task_col, "seed"]
        if trial_col:
            sort_cols.append(trial_col)
        df = df.sort_values(by=sort_cols).reset_index(drop=True)
        df["trial_value__cost_inc"] = df.groupby(["optimizer_id", task_col, "seed"])[cost_col].cummin()
        cost_col = "trial_value__cost_inc"

    # Take the latest trial per run
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

    # Seed-level record
    p_all = merged[f"{cost_col}_proposed"].to_numpy(dtype=float)
    b_all = merged[f"{cost_col}_baseline"].to_numpy(dtype=float)
    valid_mask = np.isfinite(p_all) & np.isfinite(b_all)
    p_all_val = p_all[valid_mask]
    b_all_val = b_all[valid_mask]

    diff_all = p_all_val - b_all_val
    overall_wins = int(np.sum(diff_all < -tie_tolerance))
    overall_losses = int(np.sum(diff_all > tie_tolerance))
    overall_ties = int(np.sum(np.abs(diff_all) <= tie_tolerance))

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

        # Seed-level Wilcoxon per task
        nonzero = diff[np.abs(diff) > tie_tolerance]
        if len(nonzero) >= 5:
            try:
                stat, p_val = wilcoxon(gp_v, gb_v, alternative="two-sided")
            except Exception:
                stat, p_val = np.nan, 1.0
        else:
            stat, p_val = np.nan, 1.0

        delta = calculate_cliffs_delta(gp_v, gb_v)

        # Normalized Regret per task: scale to [0, 1] relative to joint extremes
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

        diff_mean = row["norm_regret_proposed"] - row["norm_regret_baseline"]
        if diff_mean < -tie_tolerance:
            row["decision"] = "WIN"
            task_wins += 1
        elif diff_mean > tie_tolerance:
            row["decision"] = "LOSS"
            task_losses += 1
        else:
            row["decision"] = "TIE"
            task_ties += 1

    scorecard_df = pd.DataFrame(task_rows)
    scorecard_df["dim"] = scorecard_df["task"].apply(get_task_dimension)

    # Save detailed CSV scorecard
    csv_path = Path(output_dir) / f"{out_prefix}.csv"
    scorecard_df.to_csv(csv_path, index=False)

    # Demšar task-level Wilcoxon test on normalized regrets across tasks
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

    # Exact Binomial Sign Test across tasks
    non_tied_tasks = task_wins + task_losses
    if non_tied_tasks > 0:
        b_res = binomtest(k=task_wins, n=non_tied_tasks, p=0.5, alternative="greater")
        task_binom_p = float(b_res.pvalue)
    else:
        task_binom_p = 1.0

    mean_cliffs_delta = float(scorecard_df["cliffs_delta"].mean())
    task_level_cliffs_delta = calculate_cliffs_delta(task_norm_p, task_norm_b)

    # Stratification Subsets
    strata_definitions = [
        ("all_tasks", "All Tasks (Full Suite)", scorecard_df),
        ("suite_high_d_8", "All High-D (D >= 8)", scorecard_df[scorecard_df["dim"] >= 8]),
        ("suite_low_d_7", "All Low-D (D < 8)", scorecard_df[scorecard_df["dim"] < 8]),
        ("bbob_subset", "BBOB Continuous Subset", scorecard_df[scorecard_df["task"].str.contains("bbob")]),
        ("realworld_ml_subset", "Real-World ML (YAHPO + HPOBench)", scorecard_df[~scorecard_df["task"].str.contains("bbob")]),
    ]

    strat_table_rows = []
    for s_key, s_label, sub_df in strata_definitions:
        n_sub = len(sub_df)
        if n_sub == 0:
            continue
        m_p = float(sub_df["norm_regret_proposed"].mean())
        m_b = float(sub_df["norm_regret_baseline"].mean())
        rel_red = float((m_b - m_p) / m_b * 100.0) if m_b > 1e-12 else 0.0
        m_cd = float(sub_df["cliffs_delta"].mean())

        sub_p = sub_df["norm_regret_proposed"].to_numpy(dtype=float)
        sub_b = sub_df["norm_regret_baseline"].to_numpy(dtype=float)
        non_zero = int(np.sum(np.abs(sub_p - sub_b) > 1e-12))
        if non_zero >= 5:
            try:
                _, s_p_two = wilcoxon(sub_p, sub_b, alternative="two-sided")
                _, s_p_one = wilcoxon(sub_p, sub_b, alternative="less")
            except Exception:
                s_p_two, s_p_one = 1.0, 1.0
        else:
            s_p_two, s_p_one = 1.0, 1.0

        sub_w = int(np.sum(sub_df["decision"] == "WIN"))
        sub_l = int(np.sum(sub_df["decision"] == "LOSS"))
        sub_t = int(np.sum(sub_df["decision"] == "TIE"))

        strat_table_rows.append({
            "Stratum": s_label,
            "N Tasks": n_sub,
            "Record (W/T/L)": f"{sub_w}/{sub_t}/{sub_l}",
            "Mean Regret Proposed": f"{m_p:.4f}",
            "Mean Regret Baseline": f"{m_b:.4f}",
            "Rel. Reduction": f"{rel_red:+.1f}%",
            "Mean Cliff's Delta": f"{m_cd:+.3f}",
            "Demšar Wilcoxon p (2-sided)": f"{s_p_two:.4f}",
            "Demšar Wilcoxon p (1-sided)": f"{s_p_one:.4f}",
        })

    strat_df = pd.DataFrame(strat_table_rows)
    strat_csv_path = Path(output_dir) / "stratified_scorecard.csv"
    strat_df.to_csv(strat_csv_path, index=False)

    # Build Markdown scorecard
    mean_norm_p = float(scorecard_df["norm_regret_proposed"].mean())
    mean_norm_b = float(scorecard_df["norm_regret_baseline"].mean())
    rel_red_all = float((mean_norm_b - mean_norm_p) / mean_norm_b * 100.0) if mean_norm_b > 1e-12 else 0.0

    sig_task_wins = int(np.sum((scorecard_df["decision"] == "WIN") & scorecard_df["significant"]))
    sig_task_losses = int(np.sum((scorecard_df["decision"] == "LOSS") & scorecard_df["significant"]))
    sig_task_ties = len(task_rows) - sig_task_wins - sig_task_losses

    md_lines = [
        "# CARP-S BBsubset Held-Out Test Evaluation Scorecard: Proximity LCB vs SMAC3 EI",
        "",
        "## Executive Summary",
        "",
        f"- **Proposed Method**: `{proposed_id}` (Tuned Proximity LCB: k=25, decay_lambda=1.345, eps=0.1678, level=0.95)",
        f"- **Baseline Method**: `{baseline_id}` (Standard SMAC3 with native Expected Improvement)",
        f"- **Benchmark Suite**: CARP-S BBsubset Held-Out Test Set ({len(unique_tasks)} tasks, 30 seeds, T=100 budget)",
        f"- **Total Paired Runs**: {len(p_all_val)}",
        f"- **Seed-level Record (W / T / L)**: **{overall_wins} / {overall_ties} / {overall_losses}** ({overall_wins / len(p_all_val) * 100:.1f}% win rate)",
        f"- **Task-level Empirical Record (W / T / L)**: **{task_wins} / {task_ties} / {task_losses}**",
        f"- **Task-level Statistically Significant Record (Holm alpha={alpha}) (W / T / L)**: **{sig_task_wins} / {sig_task_ties} / {sig_task_losses}**",
        f"- **Mean Normalized Regret**: Proposed = **{mean_norm_p:.4f}** vs Baseline = **{mean_norm_b:.4f}** ({rel_red_all:+.1f}% relative reduction)",
        f"- **Demšar Task-Level Wilcoxon p-value (two-sided)**: `{task_w_p_two:.4e}` ({'Significant (p < 0.05)' if task_w_p_two < alpha else 'Not Significant (p >= 0.05)'})",
        f"- **Demšar Task-Level Wilcoxon p-value (one-sided, proposed < baseline)**: `{task_w_p_one:.4e}`",
        f"- **Exact Binomial Sign Test on Task Wins**: `{task_binom_p:.4e}` ({task_wins}/{non_tied_tasks} non-tied tasks won)",
        f"- **Mean Within-Task Cliff's Delta**: `{mean_cliffs_delta:+.4f}` ({'Favors Proposed' if mean_cliffs_delta < 0 else 'Favors Baseline'})",
        f"- **Task-Level Cliff's Delta (on normalized regret)**: `{task_level_cliffs_delta:+.4f}`",
        "",
        "## Stratified Scorecard Breakdown",
        "",
        dataframe_to_markdown(strat_df),
        "",
        "## Per-Task Test Set Scorecard (Holm-Bonferroni FWER alpha = 0.05)",
        "",
    ]

    display_df = scorecard_df[[
        "task", "dim", "n_seeds", "mean_proposed", "mean_baseline",
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
    print(f"  - Stratified CSV: {strat_csv_path}")

    return {
        "n_tasks": len(unique_tasks),
        "total_paired_runs": len(p_all_val),
        "overall_wins": overall_wins,
        "overall_ties": overall_ties,
        "overall_losses": overall_losses,
        "task_wins": task_wins,
        "task_ties": task_ties,
        "task_losses": task_losses,
        "mean_norm_proposed": mean_norm_p,
        "mean_norm_baseline": mean_norm_b,
        "demsar_wilcoxon_p_two": task_w_p_two,
        "demsar_wilcoxon_p_one": task_w_p_one,
        "task_binom_p": task_binom_p,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute statistical comparison between Proximity LCB and SMAC3 EI on BBsubset test set."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="results/sweep_bbsubset_test_proximity_vs_ei/logs.csv",
        help="Path to aggregated logs (csv or parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sweep_bbsubset_test_proximity_vs_ei/analysis",
        help="Directory to save scorecard artifacts.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="test_proximity_vs_ei_scorecard",
        help="Prefix for generated scorecard files.",
    )
    parser.add_argument(
        "--proposed-id",
        type=str,
        default="SMAC20_ProximityLCB_tuned",
        help="Optimizer ID for proposed method.",
    )
    parser.add_argument(
        "--baseline-id",
        type=str,
        default="SMAC3_HPOFacade_ei",
        help="Optimizer ID for baseline method.",
    )
    parser.add_argument(
        "--cost-col",
        type=str,
        default=None,
        help="Cost column name to compare.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for Holm-Bonferroni correction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_bbsubset_test_proximity_vs_ei_analysis(
        input_file=args.input_file,
        output_dir=args.output_dir,
        out_prefix=args.out_prefix,
        proposed_id=args.proposed_id,
        baseline_id=args.baseline_id,
        cost_col=args.cost_col,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
