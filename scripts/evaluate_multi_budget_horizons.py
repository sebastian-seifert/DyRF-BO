#!/usr/bin/env python3
"""Multi-Budget Horizon Evaluation & Statistical Rank Testing Tool for CARP-S Benchmark Runs.

Evaluates optimizer rankings, omnibus Friedman / Iman-Davenport tests,
and Holm-Bonferroni corrected paired Wilcoxon signed-rank tests across multiple budget horizons
(e.g., T=11, 15, 20, 25, 35, 50).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def slice_logs_at_budget(df: pd.DataFrame, max_trial: int) -> pd.DataFrame:
    """Filters logs up to max_trial and recalculates incumbent cost trajectory."""
    df_sub = df[df["n_trials"] <= max_trial].copy()
    
    # Sort to ensure proper sequential cumulative min
    df_sub = df_sub.sort_values(by=["task_id", "optimizer_id", "seed", "n_trials"])
    
    cost_col = "trial_value__cost"
    if cost_col not in df_sub.columns and "trial_value__cost_inc" in df_sub.columns:
        cost_col = "trial_value__cost_inc"
        
    df_sub["trial_value__cost_inc"] = df_sub.groupby(
        ["task_id", "optimizer_id", "seed"]
    )[cost_col].cummin()
    
    return df_sub


def compute_ranks_at_budget(
    df_sliced: pd.DataFrame,
    perf_col: str = "trial_value__cost_inc"
) -> pd.DataFrame:
    """Extracts final performance at sliced budget, averages across seeds, and computes mean ranks."""
    # Group to get final trial per (task_id, optimizer_id, seed)
    final_trials = df_sliced.groupby(["task_id", "optimizer_id", "seed"]).last().reset_index()
    
    # Average across seeds for each (task_id, optimizer_id)
    mean_per_task = final_trials.groupby(["task_id", "optimizer_id"])[perf_col].mean().reset_index()
    
    # Compute rank per task (1 = best / lowest cost)
    mean_per_task["task_rank"] = mean_per_task.groupby("task_id")[perf_col].rank(ascending=True, method="average")
    
    # Compute overall mean rank across all tasks
    optimizer_ranks = mean_per_task.groupby("optimizer_id")["task_rank"].agg(
        mean_rank="mean",
        std_rank="std",
        median_rank="median"
    ).reset_index()
    
    return optimizer_ranks.sort_values(by="mean_rank", ascending=True).reset_index(drop=True)


def evaluate_multi_budget_horizons(
    df: pd.DataFrame,
    budgets: List[int] = [15, 20, 25, 35, 50]
) -> pd.DataFrame:
    """Evaluates optimizer mean ranks across all specified budget horizons."""
    all_ranks = {}
    
    for b in budgets:
        df_b = slice_logs_at_budget(df, max_trial=b)
        ranks_b = compute_ranks_at_budget(df_b)
        all_ranks[b] = dict(zip(ranks_b["optimizer_id"], ranks_b["mean_rank"]))
        
    optimizers = sorted(list(next(iter(all_ranks.values())).keys()))
    
    rows = []
    for opt in optimizers:
        row = {"optimizer_id": opt}
        for b in budgets:
            row[f"Rank (T={b})"] = all_ranks[b].get(opt, np.nan)
        rows.append(row)
        
    summary_df = pd.DataFrame(rows)
    last_col = f"Rank (T={budgets[-1]})"
    return summary_df.sort_values(by=last_col, ascending=True).reset_index(drop=True)


def apply_holm_bonferroni(p_values: List[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    if m == 0:
        return []
        
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    
    cum_max = 0.0
    for rank_idx, (orig_idx, p_val) in enumerate(indexed):
        # Multiplier is (m - rank_idx)
        adj_p = min(1.0, (m - rank_idx) * p_val)
        cum_max = max(cum_max, adj_p)
        adjusted[orig_idx] = min(1.0, cum_max)
        
    return adjusted


def compute_rank_biserial(diffs: np.ndarray) -> float:
    """Computes matched-pairs rank-biserial correlation r_rb in [-1, 1]."""
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
        
    ranks = stats.rankdata(np.abs(nonzero))
    pos_ranks = np.sum(ranks[nonzero > 0])
    neg_ranks = np.sum(ranks[nonzero < 0])
    total_ranks = pos_ranks + neg_ranks
    
    if total_ranks == 0:
        return 0.0
    return float((pos_ranks - neg_ranks) / total_ranks)


def run_statistical_tests_at_budget(
    df_sliced: pd.DataFrame,
    baseline_id: str = "SMAC3_HPOFacade_ei",
    alpha: float = 0.05,
    perf_col: str = "trial_value__cost_inc"
) -> Dict[str, Any]:
    """Computes Friedman omnibus test and Holm-corrected pairwise Wilcoxon tests against baseline."""
    # Group to get final trial per (task_id, optimizer_id, seed)
    final_trials = df_sliced.groupby(["task_id", "optimizer_id", "seed"]).last().reset_index()
    
    # Average across seeds per task
    task_means = final_trials.groupby(["task_id", "optimizer_id"])[perf_col].mean().unstack(level="optimizer_id")
    
    # Drop any tasks with missing values for any optimizer
    task_means = task_means.dropna()
    
    n_tasks, k_opts = task_means.shape
    
    # 1. Friedman Test
    opt_vectors = [task_means[col].values for col in task_means.columns]
    friedman_stat, friedman_p = stats.friedmanchisquare(*opt_vectors)
    
    # Iman-Davenport correction
    # F_F = (N - 1) * chi2 / (N * (k - 1) - chi2)
    denom = (n_tasks * (k_opts - 1) - friedman_stat)
    if denom > 0:
        iman_davenport_f = ((n_tasks - 1) * friedman_stat) / denom
        df1 = k_opts - 1
        df2 = (k_opts - 1) * (n_tasks - 1)
        iman_davenport_p = 1.0 - stats.f.cdf(iman_davenport_f, df1, df2)
    else:
        iman_davenport_f = np.inf
        iman_davenport_p = 0.0

    # Nemenyi Critical Difference at alpha=0.05
    # Standard studentized range statistic approximation for alpha=0.05
    # For common k in [2..10], lookup table:
    q_alpha_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q_alpha = q_alpha_table.get(k_opts, 3.031)
    critical_difference = q_alpha * np.sqrt((k_opts * (k_opts + 1)) / (6.0 * max(1, n_tasks)))
    
    # 2. Pairwise Wilcoxon against Baseline
    pairwise_rows = []
    if baseline_id in task_means.columns:
        base_series = task_means[baseline_id].values
        candidates = [col for col in task_means.columns if col != baseline_id]
        
        raw_p_list = []
        raw_rows = []
        
        for cand in candidates:
            cand_series = task_means[cand].values
            diff = cand_series - base_series  # cand - base: negative means cand is better (lower cost)
            
            # Wilcoxon signed-rank test
            nonzero = diff[diff != 0]
            if len(nonzero) == 0:
                w_stat = 0.0
                p_raw = 1.0
            else:
                try:
                    res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                    w_stat = float(res.statistic)
                    p_raw = float(res.pvalue)
                except Exception:
                    w_stat = 0.0
                    p_raw = 1.0
                    
            r_rb = compute_rank_biserial(diff)
            raw_p_list.append(p_raw)
            raw_rows.append({
                "optimizer_id": cand,
                "w_stat": w_stat,
                "p_raw": p_raw,
                "r_rb": r_rb,
                "cand_mean": float(np.mean(cand_series)),
                "base_mean": float(np.mean(base_series)),
            })
            
        holm_p_list = apply_holm_bonferroni(raw_p_list)
        
        for r_dict, p_h in zip(raw_rows, holm_p_list):
            r_dict["p_holm"] = p_h
            r_dict["is_significant"] = p_h < alpha
            if p_h < alpha:
                r_dict["verdict"] = "WIN" if r_dict["r_rb"] < 0 else "LOSS"
            else:
                r_dict["verdict"] = "TIE"
            pairwise_rows.append(r_dict)

    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df = pairwise_df.sort_values(by="p_holm", ascending=True).reset_index(drop=True)
        
    return {
        "n_tasks": n_tasks,
        "k_optimizers": k_opts,
        "friedman_stat": float(friedman_stat),
        "friedman_p": float(friedman_p),
        "iman_davenport_f": float(iman_davenport_f),
        "iman_davenport_p": float(iman_davenport_p),
        "critical_difference": float(critical_difference),
        "pairwise_results": pairwise_df
    }


def evaluate_multi_budget_statistical_matrix(
    df: pd.DataFrame,
    budgets: List[int] = [11, 15, 20, 25, 35, 50],
    baseline_id: str = "SMAC3_HPOFacade_ei",
    alpha: float = 0.05
) -> Dict[int, Dict[str, Any]]:
    """Evaluates statistical rank tests across all specified budget cutoffs."""
    matrix = {}
    for b in budgets:
        df_b = slice_logs_at_budget(df, max_trial=b)
        matrix[b] = run_statistical_tests_at_budget(df_b, baseline_id=baseline_id, alpha=alpha)
    return matrix


def format_multi_budget_markdown(
    summary_df: pd.DataFrame,
    budgets: List[int],
    stat_matrix: Optional[Dict[int, Dict[str, Any]]] = None,
    baseline_id: str = "SMAC3_HPOFacade_ei"
) -> str:
    """Generates detailed GitHub Flavored Markdown report with statistical tests."""
    lines = [
        "# Multi-Budget Horizon Ranking & Statistical Analysis",
        "",
        "## 1. Cross-Horizon Mean Rank Evolution",
        "",
        "| Optimizer | " + " | ".join([f"Rank (T={b})" for b in budgets]) + " | Δ (Early T={} → Final T={}) |".format(budgets[0], budgets[-1]),
        "| :--- | " + " | ".join([":---:" for _ in budgets]) + " | :---: |"
    ]
    
    for _, row in summary_df.iterrows():
        opt = f"`{row['optimizer_id']}`"
        ranks_str = " | ".join([f"{row[f'Rank (T={b})']:.2f}" for b in budgets])
        delta = row[f"Rank (T={budgets[-1]})"] - row[f"Rank (T={budgets[0]})"]
        delta_str = f"{delta:+.2f}"
        if delta > 0.3:
            delta_tag = f"🔴 {delta_str} (Degraded)"
        elif delta < -0.3:
            delta_tag = f"🟢 {delta_str} (Improved)"
        else:
            delta_tag = f"⚪ {delta_str} (Stable)"
        lines.append(f"| {opt} | {ranks_str} | {delta_tag} |")
        
    lines.append("")
    
    if stat_matrix is not None:
        lines.extend([
            "## 2. Omnibus Friedman & Iman-Davenport Tests per Budget Horizon",
            "",
            "| Budget Horizon | Tasks | Optimizers | Friedman $\\chi_F^2$ | $p_{\\text{Friedman}}$ | Iman-Davenport $F_F$ | $p_{\\text{Iman-Davenport}}$ | Critical Difference ($CD$) | Global Significance (α=0.05) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ])
        
        for b in budgets:
            res = stat_matrix[b]
            sig_tag = "✓ **YES**" if res["iman_davenport_p"] < 0.05 else "✗ NO"
            lines.append(
                f"| **T={b}** | {res['n_tasks']} | {res['k_optimizers']} | {res['friedman_stat']:.2f} | {res['friedman_p']:.4e} | {res['iman_davenport_f']:.2f} | {res['iman_davenport_p']:.4e} | {res['critical_difference']:.3f} | {sig_tag} |"
            )
            
        lines.extend([
            "",
            f"## 3. Pairwise Holm-Bonferroni Corrected Wilcoxon Tests (vs. `{baseline_id}`)",
            ""
        ])
        
        for b in budgets:
            res = stat_matrix[b]
            lines.append(f"### Horizon $T = {b}$ (Iman-Davenport $p = {res['iman_davenport_p']:.4e}$)")
            pw_df = res["pairwise_results"]
            if pw_df.empty:
                lines.append("No pairwise results available.\n")
                continue
                
            lines.extend([
                "| Candidate Optimizer | $r_{\\text{rb}}$ (Effect Size) | $W$ Stat | $p_{\\text{raw}}$ | $p_{\\text{Holm}}$ | Significant (α=0.05) | Verdict |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
            ])
            for _, r in pw_df.iterrows():
                opt = f"`{r['optimizer_id']}`"
                sig_str = "✓ **YES**" if r["is_significant"] else "✗ NO"
                if r["verdict"] == "WIN":
                    v_str = "**WIN (Superior)** 🏆"
                elif r["verdict"] == "LOSS":
                    v_str = "*LOSS (Inferior)* ❌"
                else:
                    v_str = "TIE (Equivalent) ⚪"
                lines.append(f"| {opt} | {r['r_rb']:+.2f} | {r['w_stat']:.1f} | {r['p_raw']:.4f} | {r['p_holm']:.4f} | {sig_str} | {v_str} |")
            lines.append("")
            
    return "\n".join(lines)


import argparse


def positive_int(value: str) -> int:
    """Type validator ensuring integer value is strictly positive (> 0)."""
    try:
        ivalue = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value!r}")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Value must be a positive integer (> 0), got {ivalue}")
    return ivalue


def non_empty_path(value: str) -> str:
    """Type validator ensuring path argument is non-empty."""
    if not value or not str(value).strip():
        raise argparse.ArgumentTypeError("Path argument cannot be empty.")
    return str(value).strip()


def non_empty_str(value: str) -> str:
    """Type validator ensuring string argument is non-empty."""
    if not value or not str(value).strip():
        raise argparse.ArgumentTypeError("String argument cannot be empty.")
    return str(value).strip()


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the configured argument parser with type validation and usage examples."""
    parser = argparse.ArgumentParser(
        description="Multi-budget horizon evaluator & statistical rank testing tool for CARP-S benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate default parquet logs across standard horizons:
  python scripts/evaluate_multi_budget_horizons.py results/noisy_sweep_analysis/logs.parquet

  # Custom budget horizons and baseline:
  python scripts/evaluate_multi_budget_horizons.py logs.csv --budgets 10 20 30 50 --baseline SMAC3_HPOFacade_ei

  # Custom output report directory:
  python scripts/evaluate_multi_budget_horizons.py logs.parquet --outdir results/my_budget_eval
        """
    )
    parser.add_argument(
        "logs_path",
        nargs="?",
        type=non_empty_path,
        default="results/noisy_sweep_analysis/logs.parquet",
        help="Path to logs.parquet or logs.csv (default: results/noisy_sweep_analysis/logs.parquet)"
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=positive_int,
        default=[11, 15, 20, 25, 35, 50],
        help="List of positive budget trial cutoffs (> 0, default: 11 15 20 25 35 50)"
    )
    parser.add_argument(
        "--baseline",
        type=non_empty_str,
        default="SMAC3_HPOFacade_ei",
        help="Baseline optimizer ID for pairwise tests (default: SMAC3_HPOFacade_ei)"
    )
    parser.add_argument(
        "--outdir",
        type=non_empty_path,
        default="results/multi_budget_analysis",
        help="Output directory for reports (default: results/multi_budget_analysis)"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    logs_file = Path(args.logs_path)
    if not logs_file.exists():
        print(f"Error: {logs_file} not found.")
        sys.exit(1)
        
    print(f"Loading logs from {logs_file}...")
    if logs_file.suffix == ".parquet":
        df = pd.read_parquet(logs_file)
    else:
        df = pd.read_csv(logs_file)
        
    print(f"Loaded {len(df)} rows across {df['task_id'].nunique()} tasks and {df['optimizer_id'].nunique()} optimizers.")
    print(f"Evaluating statistical tests across budget horizons: {args.budgets} against baseline '{args.baseline}'...")
    
    summary_df = evaluate_multi_budget_horizons(df, budgets=args.budgets)
    stat_matrix = evaluate_multi_budget_statistical_matrix(df, budgets=args.budgets, baseline_id=args.baseline)
    md_report = format_multi_budget_markdown(summary_df, args.budgets, stat_matrix=stat_matrix, baseline_id=args.baseline)
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    csv_out = outdir / "multi_budget_ranks.csv"
    md_out = outdir / "multi_budget_report.md"
    
    summary_df.to_csv(csv_out, index=False)
    with open(md_out, "w") as f:
        f.write(md_report)
        
    print("\n" + md_report)
    print(f"Saved results to {csv_out} and {md_out}")


if __name__ == "__main__":
    main()
