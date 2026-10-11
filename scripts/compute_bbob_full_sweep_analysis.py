#!/usr/bin/env python3
"""Statistical Analysis and Dimensionality-Stratified Scorecards for BBOB Sweeps.

Computes:
1. Global per-task Min-Max scaled regret across all evaluated approaches.
2. Demšar-compliant omnibus Friedman test.
3. Paired Wilcoxon signed-rank tests vs Control 1 (LCB) and Control 2 (EI).
4. Multiplicity correction via step-down Holm-Bonferroni method.
5. Non-parametric effect sizes: Cliff's Delta and paired Win/Loss/Tie dominance.
6. Dimensionality-stratified scorecards across individual dimensions (2, 4, 8, 16, 32)
   and dimensional tiers (Low-D <= 4D, Mid-D 8D, High-D >= 16D).
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
from scipy.stats import friedmanchisquare, wilcoxon


def apply_holm_bonferroni(p_vals: Sequence[float]) -> List[float]:
    """Computes Holm-Bonferroni step-down adjusted p-values ensuring FWER control."""
    p = np.array(p_vals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    p = np.where(np.isnan(p), 1.0, p)
    sort_idx = np.argsort(p)
    sorted_p = p[sort_idx]

    adj_sorted = np.zeros(n, dtype=float)
    for i in range(n):
        adj_sorted[i] = min(1.0, sorted_p[i] * (n - i))

    for i in range(1, n):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])

    adj_original = np.zeros(n, dtype=float)
    adj_original[sort_idx] = adj_sorted
    return [float(val) for val in adj_original]


def calculate_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cliff's delta non-parametric effect size between two vectors.

    Negative delta indicates x is systematically lower (better for loss minimization).
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


def compute_min_max_scaled_regret(df: pd.DataFrame, cost_col: str = "trial_value__cost_inc") -> pd.DataFrame:
    """Applies global per-task Min-Max scaling across all approaches and seeds."""
    df_scaled = df.copy()
    task_bounds = df.groupby("task_id")[cost_col].agg(["min", "max"]).reset_index()

    task_bounds["spread"] = task_bounds["max"] - task_bounds["min"]
    merged = pd.merge(df_scaled, task_bounds, on="task_id", how="left")

    def _scale(row):
        spread = row["spread"]
        if spread > 1e-12:
            return (row[cost_col] - row["min"]) / spread
        return 0.0

    merged["normalized_regret"] = merged.apply(_scale, axis=1)
    return merged


def infer_dimension(task_id: str) -> int:
    m = re.search(r"cfg_(\d+)_", str(task_id))
    if m:
        return int(m.group(1))
    m2 = re.search(r"bbob[/-](\d+)", str(task_id), re.IGNORECASE)
    if m2:
        return int(m2.group(1))
    return -1


def _generate_scorecard_for_subset(
    subset_df: pd.DataFrame,
    all_optimizers: List[str],
    controls: Sequence[str],
    cost_col: str,
) -> pd.DataFrame:
    """Generates statistical comparison rows for a filtered subset of tasks."""
    rows = []
    task_ids = sorted(subset_df["task_id"].unique())

    for opt_id in all_optimizers:
        opt_data = subset_df[subset_df["optimizer_id"] == opt_id]
        if opt_data.empty:
            continue

        mean_norm = float(opt_data["normalized_regret"].mean())
        median_norm = float(opt_data["normalized_regret"].median())
        mean_raw = float(opt_data[cost_col].mean())

        row: Dict[str, Any] = {
            "optimizer_id": opt_id,
            "n_tasks": len(opt_data["task_id"].unique()),
            "n_runs": len(opt_data),
            "mean_norm_regret": mean_norm,
            "median_norm_regret": median_norm,
            "mean_raw_cost": mean_raw,
        }

        # Compare against controls
        for ctrl in controls:
            ctrl_data = subset_df[subset_df["optimizer_id"] == ctrl]
            ctrl_prefix = "ctrl_lcb" if "lcb" in ctrl.lower() else "ctrl_ei"

            if ctrl_data.empty or opt_id == ctrl:
                row[f"{ctrl_prefix}_win_rate"] = 0.5 if opt_id == ctrl else np.nan
                row[f"{ctrl_prefix}_cliffs_delta"] = 0.0 if opt_id == ctrl else np.nan
                row[f"{ctrl_prefix}_wilcoxon_p"] = 1.0 if opt_id == ctrl else np.nan
                continue

            # Task-level paired aggregation
            opt_medians = opt_data.groupby("task_id")["normalized_regret"].median()
            ctrl_medians = ctrl_data.groupby("task_id")["normalized_regret"].median()

            common_tasks = sorted(set(opt_medians.index).intersection(set(ctrl_medians.index)))
            if not common_tasks:
                row[f"{ctrl_prefix}_win_rate"] = np.nan
                row[f"{ctrl_prefix}_cliffs_delta"] = np.nan
                row[f"{ctrl_prefix}_wilcoxon_p"] = np.nan
                continue

            vec_opt = np.array([opt_medians[t] for t in common_tasks])
            vec_ctrl = np.array([ctrl_medians[t] for t in common_tasks])

            wins = int(np.sum(vec_opt < vec_ctrl - 1e-6))
            losses = int(np.sum(vec_opt > vec_ctrl + 1e-6))
            ties = len(common_tasks) - wins - losses
            win_rate = (wins + 0.5 * ties) / len(common_tasks)

            cd = calculate_cliffs_delta(vec_opt, vec_ctrl)

            diff = vec_opt - vec_ctrl
            nonzeros = np.sum(np.abs(diff) > 1e-8)
            if nonzeros >= 5:
                try:
                    _, p_val = wilcoxon(vec_opt, vec_ctrl, alternative="two-sided")
                except Exception:
                    p_val = 1.0
            else:
                p_val = 1.0

            row[f"{ctrl_prefix}_win_rate"] = float(win_rate)
            row[f"{ctrl_prefix}_cliffs_delta"] = float(cd)
            row[f"{ctrl_prefix}_wilcoxon_p"] = float(p_val)

        rows.append(row)

    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        # Apply Holm-Bonferroni correction
        for ctrl in controls:
            ctrl_prefix = "ctrl_lcb" if "lcb" in ctrl.lower() else "ctrl_ei"
            p_col = f"{ctrl_prefix}_wilcoxon_p"
            if p_col in res_df.columns:
                adj = apply_holm_bonferroni(res_df[p_col].fillna(1.0).tolist())
                res_df[f"{ctrl_prefix}_wilcoxon_adj_p"] = adj

        res_df = res_df.sort_values("mean_norm_regret")
    return res_df


def run_bbob_statistical_analysis(
    df: pd.DataFrame,
    cost_col: str = "trial_value__cost_inc",
    controls: Sequence[str] = ("SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"),
) -> Dict[str, Any]:
    """Runs global, dimension-stratified, and tier-stratified statistical analysis."""
    working_df = df.copy()
    if "dimension" not in working_df.columns or working_df["dimension"].isna().all():
        working_df["dimension"] = working_df["task_id"].apply(infer_dimension)

    # Final incumbent extraction
    grouper = ["task_id", "optimizer_id", "seed"]
    trial_col = "trial" if "trial" in working_df.columns else "n_trials"
    if trial_col in working_df.columns:
        idx = working_df.groupby(grouper)[trial_col].idxmax()
        final_df = working_df.loc[idx].copy()
    else:
        final_df = working_df.copy()

    scaled_df = compute_min_max_scaled_regret(final_df, cost_col=cost_col)
    all_optimizers = sorted(scaled_df["optimizer_id"].unique())

    # 1. Omnibus Friedman test
    task_medians = scaled_df.groupby(["task_id", "optimizer_id"])["normalized_regret"].median().unstack()
    clean_medians = task_medians.dropna()
    friedman_stat, friedman_p = np.nan, np.nan
    if clean_medians.shape[0] >= 3 and clean_medians.shape[1] >= 3:
        try:
            samples = [clean_medians[col].values for col in clean_medians.columns]
            friedman_stat, friedman_p = friedmanchisquare(*samples)
        except Exception:
            pass

    # 2. Global Scorecard
    global_scorecard = _generate_scorecard_for_subset(scaled_df, all_optimizers, controls, cost_col)

    # 3. Stratified by exact dimension
    by_dimension: Dict[int, pd.DataFrame] = {}
    unique_dims = sorted([d for d in scaled_df["dimension"].unique() if d > 0])
    for dim in unique_dims:
        dim_sub = scaled_df[scaled_df["dimension"] == dim]
        by_dimension[dim] = _generate_scorecard_for_subset(dim_sub, all_optimizers, controls, cost_col)

    # 4. Stratified by dimensional tier
    by_tier: Dict[str, pd.DataFrame] = {}
    tier_defs = {
        "low_d (<=4D)": scaled_df[scaled_df["dimension"] <= 4],
        "mid_d (8D)": scaled_df[scaled_df["dimension"] == 8],
        "high_d (>=16D)": scaled_df[scaled_df["dimension"] >= 16],
    }
    for tier_name, tier_sub in tier_defs.items():
        if not tier_sub.empty:
            by_tier[tier_name] = _generate_scorecard_for_subset(tier_sub, all_optimizers, controls, cost_col)

    return {
        "friedman_stat": float(friedman_stat) if not np.isnan(friedman_stat) else None,
        "friedman_p": float(friedman_p) if not np.isnan(friedman_p) else None,
        "global_scorecard": global_scorecard,
        "by_dimension": by_dimension,
        "by_tier": by_tier,
    }


def export_analysis_results(results: Dict[str, Any], output_dir: str) -> None:
    """Exports analysis tables to CSV and formatted Markdown."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Export global scorecard
    results["global_scorecard"].to_csv(out_path / "global_scorecard.csv", index=False)

    # Export dimension scorecards
    for dim, card in results["by_dimension"].items():
        card.to_csv(out_path / f"scorecard_dim_{dim}.csv", index=False)

    # Export tier scorecards
    for tier, card in results["by_tier"].items():
        tier_slug = tier.split()[0].replace("-", "_")
        card.to_csv(out_path / f"scorecard_tier_{tier_slug}.csv", index=False)

    # Generate Markdown Summary Report
    md_report = [
        "# CARP-S Full BBOB Benchmark Scorecard & Dimensionality Analysis",
        "",
        f"**Omnibus Friedman Test**: $\\chi^2 = {results['friedman_stat']}$, $p = {results['friedman_p']}$",
        "",
        "## 1. Global Scorecard (All Dimensions)",
        "",
        results["global_scorecard"].to_markdown(index=False),
        "",
        "## 2. Dimensional Tier Stratification",
        "",
    ]

    for tier, card in results["by_tier"].items():
        md_report.append(f"### Tier: {tier}")
        md_report.append("")
        md_report.append(card.to_markdown(index=False))
        md_report.append("")

    report_path = out_path / "bbob_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_report))
    print(f"[SUCCESS] Exported report and scorecards to {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BBOB full sweep statistical analysis.")
    parser.add_argument("--input", type=str, default="results/sweep_bbob_full_comparison/logs.parquet")
    parser.add_argument("--output-dir", type=str, default="results/sweep_bbob_full_comparison/analysis")
    parser.add_argument("--cost-col", type=str, default="trial_value__cost_inc")
    args = parser.parse_args()

    in_path = Path(args.input)
    if in_path.suffix == ".parquet":
        data = pd.read_parquet(in_path)
    else:
        data = pd.read_csv(in_path)

    res = run_bbob_statistical_analysis(data, cost_col=args.cost_col)
    export_analysis_results(res, args.output_dir)
