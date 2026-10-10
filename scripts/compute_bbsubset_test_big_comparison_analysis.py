#!/usr/bin/env python3
"""Statistical Analysis and Scorecard Generator for CARP-S BBSubset Big Comparison Sweep.

Computes:
1. Global per-task Min-Max Scaling across all 11 evaluated approaches:
   norm_regret = (cost - min_task_cost) / (max_task_cost - min_task_cost)
2. Demšar-compliant omnibus Friedman test across all candidate approaches.
3. Paired Wilcoxon signed-rank tests against both control baselines:
   - Control 1: SMAC3_HPOFacade_lcb (Tier 1: Locked LCB family)
   - Control 2: SMAC3_HPOFacade_ei (Tier 2: Production Default)
4. Multiplicity correction via step-down Holm-Bonferroni method.
5. Non-parametric effect sizes: Cliff's Delta and Paired Win/Loss/Tie Dominance.
6. Markdown and CSV scorecards.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    """Computes Cliff's delta non-parametric effect size between two paired vectors.

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
    """Applies rigorous min-max scaling per task across all approaches and seeds."""
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


def run_statistical_analysis(
    df: pd.DataFrame,
    cost_col: str = "trial_value__cost_inc",
    tie_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Runs omnibus Friedman test and paired Wilcoxon signed-rank tests."""
    # 1. Take the final incumbent per run
    grouper = ["task_id", "optimizer_id", "seed"]
    trial_col = "n_trials" if "n_trials" in df.columns else "trial"
    if trial_col in df.columns:
        idx = df.groupby(grouper)[trial_col].idxmax()
        final_df = df.loc[idx].copy()
    else:
        final_df = df.copy()

    # Apply global per-task Min-Max scaling
    scaled_df = compute_min_max_scaled_regret(final_df, cost_col=cost_col)

    # 2. Compute task-level medians for the omnibus Friedman test
    task_medians = scaled_df.groupby(["task_id", "optimizer_id"])["normalized_regret"].median().unstack()

    friedman_stat, friedman_p = np.nan, np.nan
    clean_medians = task_medians.dropna()
    if clean_medians.shape[0] >= 3 and clean_medians.shape[1] >= 3:
        try:
            samples = [clean_medians[col].values for col in clean_medians.columns]
            friedman_stat, friedman_p = friedmanchisquare(*samples)
        except Exception:
            pass

    # 3. Paired analysis against Control 1 (SMAC3_HPOFacade_lcb) and Control 2 (SMAC3_HPOFacade_ei)
    controls = ["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"]
    all_optimizers = sorted(scaled_df["optimizer_id"].unique())

    comparison_results = {}
    for ctrl in controls:
        if ctrl not in all_optimizers:
            continue
        ctrl_df = scaled_df[scaled_df["optimizer_id"] == ctrl]

        pairwise_rows = []
        raw_p_values = []
        for opt in all_optimizers:
            if opt == ctrl:
                continue
            cand_df = scaled_df[scaled_df["optimizer_id"] == opt]
            paired = pd.merge(
                cand_df[["task_id", "seed", "normalized_regret", cost_col]],
                ctrl_df[["task_id", "seed", "normalized_regret", cost_col]],
                on=["task_id", "seed"],
                suffixes=("_cand", "_ctrl"),
            )
            if paired.empty:
                continue

            diff = paired["normalized_regret_cand"].values - paired["normalized_regret_ctrl"].values
            n_pairs = len(diff)
            wins = int(np.sum(diff < -tie_tolerance))
            losses = int(np.sum(diff > tie_tolerance))
            ties = int(np.sum(np.abs(diff) <= tie_tolerance))

            try:
                stat, p_val = wilcoxon(
                    paired["normalized_regret_cand"].values,
                    paired["normalized_regret_ctrl"].values,
                    alternative="two-sided",
                )
            except Exception:
                stat, p_val = np.nan, 1.0

            delta = calculate_cliffs_delta(
                paired["normalized_regret_cand"].values,
                paired["normalized_regret_ctrl"].values,
            )

            raw_p_values.append(p_val)
            pairwise_rows.append({
                "optimizer": opt,
                "control": ctrl,
                "n_paired_runs": n_pairs,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate": float(wins / n_pairs) if n_pairs > 0 else 0.0,
                "mean_norm_regret_cand": float(np.mean(paired["normalized_regret_cand"])),
                "mean_norm_regret_ctrl": float(np.mean(paired["normalized_regret_ctrl"])),
                "regret_diff": float(np.mean(diff)),
                "cliffs_delta": delta,
                "raw_p_value": p_val,
            })

        # Apply Holm-Bonferroni correction
        adj_p_values = apply_holm_bonferroni(raw_p_values)
        for r, adj_p in zip(pairwise_rows, adj_p_values):
            r["holm_bonferroni_p"] = adj_p

        comparison_results[ctrl] = pairwise_rows

    return {
        "friedman_stat": float(friedman_stat) if np.isfinite(friedman_stat) else np.nan,
        "friedman_p": float(friedman_p) if np.isfinite(friedman_p) else np.nan,
        "comparisons": comparison_results,
        "scaled_df": scaled_df,
    }


def generate_markdown_scorecard(analysis: Dict[str, Any]) -> str:
    """Builds a publication-quality Markdown scorecard."""
    lines = [
        "# CARP-S BBSubset Held-Out Test Big Comparison Scorecard",
        "",
        "## 1. Omnibus Friedman Test (Demšar Multi-Task Protocol)",
        f"- **Friedman Statistic $\\chi^2_F$**: {analysis['friedman_stat']:.4f}",
        f"- **Asymptotic p-value**: {analysis['friedman_p']:.4e}",
        "",
    ]

    comparisons = analysis.get("comparisons", {})
    for ctrl, rows in comparisons.items():
        lines.append(f"## 2. Paired Evaluation vs. Control: `{ctrl}`")
        lines.append("| Optimizer | Paired Runs | Record (W / L / T) | Win Rate | Mean Norm Regret (Cand vs Ctrl) | Cliff's $\\delta$ | Raw $p$ | Holm-Bonferroni $p$ |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in rows:
            record_str = f"{r['wins']} / {r['losses']} / {r['ties']}"
            regret_str = f"{r['mean_norm_regret_cand']:.4f} vs {r['mean_norm_regret_ctrl']:.4f} ($\\Delta={r['regret_diff']:+.4f}$)"
            lines.append(
                f"| `{r['optimizer']}` | {r['n_paired_runs']} | {record_str} | {r['win_rate']*100:.1f}% | {regret_str} | {r['cliffs_delta']:+.3f} | {r['raw_p_value']:.3e} | **{r['holm_bonferroni_p']:.3e}** |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run statistical analysis and produce scorecards for CARP-S BBSubset Big Comparison sweep."
    )
    parser.add_argument(
        "--input-parquet",
        type=str,
        default="results/sweep_bbsubset_test_big_comparison/logs.parquet",
        help="Input logs.parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sweep_bbsubset_test_big_comparison/analysis",
        help="Target directory for scorecards and analysis tables.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_parquet)
    if not input_path.exists():
        print(f"[ERROR] Input parquet file {input_path} does not exist.")
        sys.exit(1)

    df = pd.read_parquet(input_path)
    analysis = run_statistical_analysis(df)
    scorecard_md = generate_markdown_scorecard(analysis)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scorecard.md", "w", encoding="utf-8") as f:
        f.write(scorecard_md)

    print(f"[SUCCESS] Scorecard generated at {out_dir / 'scorecard.md'}")


if __name__ == "__main__":
    main()
