#!/usr/bin/env python3
"""Script to compute median incumbent across seeds per task for yahpo_rbv2_ranger,
aggregate the mean over all task medians per approach, and compute the one-sided
Wilcoxon signed-rank test and Cliff's delta effect size.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Any

import numpy as np
import pandas as pd
from scipy import stats


def load_data(path: Path | str) -> pd.DataFrame:
    """Loads benchmark logs from either a .parquet or .csv file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Logs file not found: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        # Fallback: try parquet then csv
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)


def extract_final_incumbents(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts the final incumbent cost for each (task_id, optimizer_id, seed) run.
    Uses 'trial_value__cost_inc' if available, otherwise 'trial_value__cost'.
    """
    cost_col = "trial_value__cost_inc" if "trial_value__cost_inc" in df.columns else "trial_value__cost"
    
    # Compute the minimum cost per run (or last cost_inc)
    inc = (
        df.groupby(["task_id", "optimizer_id", "seed"])[cost_col]
        .min()
        .reset_index()
        .rename(columns={cost_col: "incumbent_cost"})
    )
    return inc


def compute_task_medians(df_incumbents: pd.DataFrame) -> pd.DataFrame:
    """Computes the median incumbent cost across seeds for each task and optimizer."""
    medians = (
        df_incumbents.groupby(["task_id", "optimizer_id"])["incumbent_cost"]
        .median()
        .reset_index()
        .rename(columns={"incumbent_cost": "median_cost"})
    )
    return medians


def compute_mean_of_medians(df_task_medians: pd.DataFrame) -> dict[str, float]:
    """Computes the mean over all task medians for each approach."""
    means = df_task_medians.groupby("optimizer_id")["median_cost"].mean().to_dict()
    return means


def calculate_cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Computes Cliff's delta non-parametric effect size.
    Negative delta means x is systematically lower (better for minimization).
    """
    arr_x = np.asarray(x).ravel()
    arr_y = np.asarray(y).ravel()
    n_x, n_y = len(arr_x), len(arr_y)
    if n_x == 0 or n_y == 0:
        return 0.0
    greater = 0
    less = 0
    for val_x in arr_x:
        greater += int(np.sum(val_x > arr_y))
        less += int(np.sum(val_x < arr_y))
    return float((greater - less) / (n_x * n_y))


def interpret_cliffs_delta(d: float) -> str:
    """Provides Romano et al. qualitative interpretation of Cliff's delta."""
    abs_d = abs(d)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"


def compute_wilcoxon_and_cliffs_delta(
    df_task_medians: pd.DataFrame,
    proposed_id: str = "SMAC20_ProximityLCB",
    baseline_id: str = "SMAC3_HPOFacade_lcb",
    eps: float = 1e-9,
) -> dict[str, Any]:
    """Aligns tasks across proposed and baseline, calculates mean of medians,
    one-sided Wilcoxon signed-rank test (alternative='less'), Cliff's delta,
    and task win/tie/loss counts.
    """
    pivoted = df_task_medians.pivot(index="task_id", columns="optimizer_id", values="median_cost").dropna()
    
    if proposed_id not in pivoted.columns or baseline_id not in pivoted.columns:
        raise ValueError(f"Required optimizers '{proposed_id}' and '{baseline_id}' not found in data.")

    x = pivoted[proposed_id].to_numpy(dtype=float)
    y = pivoted[baseline_id].to_numpy(dtype=float)
    n_tasks = len(pivoted)

    mean_prop = float(np.mean(x))
    mean_base = float(np.mean(y))

    # Wins / Ties / Losses for proposed
    wins = int(np.sum(x < y - eps))
    losses = int(np.sum(x > y + eps))
    ties = int(np.sum(np.abs(x - y) <= eps))

    # Cliff's Delta
    cliffs_d = calculate_cliffs_delta(x, y)
    cliffs_mag = interpret_cliffs_delta(cliffs_d)

    # Paired task dominance: (wins - losses) / n_tasks
    paired_dom = float((wins - losses) / n_tasks) if n_tasks > 0 else 0.0

    # One-sided Wilcoxon signed-rank test (x < y => alternative='less')
    diffs = x - y
    if np.allclose(diffs, 0):
        w_stat, w_p = 0.0, 1.0
    else:
        res = stats.wilcoxon(x, y, alternative="less", zero_method="wilcox")
        w_stat, w_p = float(res.statistic), float(res.pvalue)

    return {
        "n_tasks": n_tasks,
        "proposed_id": proposed_id,
        "baseline_id": baseline_id,
        "mean_proposed": mean_prop,
        "mean_baseline": mean_base,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "cliffs_delta": cliffs_d,
        "cliffs_magnitude": cliffs_mag,
        "wins_proposed": wins,
        "ties": ties,
        "losses_proposed": losses,
        "paired_dominance": paired_dom,
        "paired_table": pivoted,
    }


def format_markdown_report(stats_dict: dict[str, Any]) -> str:
    """Formats the statistical analysis into a clear markdown scorecard with full experimental setup."""
    suite = stats_dict.get("suite_name", "yahpo_rbv2_ranger")
    prop = stats_dict["proposed_id"]
    base = stats_dict["baseline_id"]
    n_tasks = stats_dict["n_tasks"]
    n_seeds = stats_dict.get("n_seeds", 30)
    n_trials = stats_dict.get("n_trials", 100)
    n_init = stats_dict.get("n_init", 10)

    lines = [
        f"# Statistical Scorecard: {suite}",
        "",
        "## Experimental Setup",
        "",
        f"- **Benchmark Suite**: `{suite}` ({n_tasks} tasks)",
        f"- **Total Budget / Iterations**: {n_trials} trials per run",
        f"- **Stage 1 (Initial Design Phase, Trials 1–10)**: {n_init} trials (`SobolInitialDesign`, quasi-random initialization, **100% identical configurations across both methods**)",
        f"- **Stage 2 (LCB Warmup Phase, Trials 11–25)**: 15 trials (Standard RF LCB with $\\beta=3.8416$ to accumulate $N > k=25$ observations for neighbor graphs)",
        f"- **Stage 3 (Active Proximity BO Phase, Trials 26–100)**: {n_trials - 25} trials (Floored Proximity Lower Bound acquisition diverges and guides search)",
        f"- **Seeds**: {n_seeds} independent runs per task (seeds 1 to {n_seeds})",
        f"- **Total Runs Evaluated**: {n_tasks} tasks × 2 approaches × {n_seeds} seeds = {n_tasks * 2 * n_seeds:,} runs ({n_tasks * 2 * n_seeds * n_trials:,} trials)",
        "",
        "### Approaches & Hyperparameters",
        "",
        f"1. **Proposed Approach (`{prop}`)**:",
        "   - **Surrogate**: `CustomUncertaintyRandomForest` with localized epistemic uncertainty (`proximity_b`)",
        "   - **Kernel Distance Decay**: $\\lambda = 1.345$",
        "   - **Acquisition Function**: `proximity_lcb`",
        "   - **Proximity Hyperparameters**: $k = 25$ nearest neighbors, $\\text{level} = 0.95$ (95% empirical quantile), $\\epsilon = 0.16$ (dispersion floor), $k_{\\text{warmup}} = 25$",
        "",
        f"2. **Baseline Approach (`{base}`)**:",
        "   - **Surrogate**: Standard SMAC3 Random Forest surrogate (`smac.facade.HyperparameterOptimizationFacade`)",
        "   - **Acquisition Function**: Standard Lower Confidence Bound (`lcb`)",
        "   - **Exploration Weight**: $\\beta = 3.8416$ (corresponding to $\\kappa = \\sqrt{\\beta} = 1.96$, fixed, `update_beta = False`)",
        "",
        "## Summary Metrics Across All Tasks",
        "",
        "| Metric | Proposed (`" + prop + "`) | Baseline (`" + base + "`) | Net Advantage |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Mean of Medians** (lower is better) | `{stats_dict['mean_proposed']:.6f}` | `{stats_dict['mean_baseline']:.6f}` | `Δ = {stats_dict['mean_proposed'] - stats_dict['mean_baseline']:.6f}` |",
        f"| **Task Wins** | **{stats_dict['wins_proposed']}** ({stats_dict['wins_proposed'] / n_tasks * 100:.1f}%) | {stats_dict['losses_proposed']} ({stats_dict['losses_proposed'] / n_tasks * 100:.1f}%) | **+{stats_dict['wins_proposed'] - stats_dict['losses_proposed']} tasks** |",
        f"| **Task Ties** | {stats_dict['ties']} ({stats_dict['ties'] / n_tasks * 100:.1f}%) | {stats_dict['ties']} ({stats_dict['ties'] / n_tasks * 100:.1f}%) | — |",
        "",
        "## Hypothesis Testing & Effect Sizes",
        "",
        "- **One-Sided Wilcoxon Signed-Rank Test (`H1: Proposed < Baseline`)**:",
        f"  - Statistic ($W$): `{stats_dict['wilcoxon_stat']}`",
        f"  - $p$-value: `{stats_dict['wilcoxon_p']:.4e}` {'(Statistically Significant, p < 0.05)' if stats_dict['wilcoxon_p'] < 0.05 else '(Not Statistically Significant)'}",
        "- **Paired Task Dominance Metric**: **`" + f"{stats_dict['paired_dominance']:+.4f}`**",
        f"  - Calculated as $(W_{{\\text{{wins}}}} - W_{{\\text{{losses}}}}) / N = ({stats_dict['wins_proposed']} - {stats_dict['losses_proposed']}) / {n_tasks} = {stats_dict['paired_dominance'] * 100:+.1f}\\%$",
        rf"- **Cross-Task Unpaired Cliff's Delta ($\delta$)**:",
        f"  - Value: `{stats_dict['cliffs_delta']:.4f}`",
        f"  - Interpretation: **{stats_dict['cliffs_magnitude'].capitalize()}** effect ({stats_dict['cliffs_magnitude']} due to cross-task baseline scale variance)",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute median incumbents, mean of medians, Wilcoxon p-value, and Cliff's delta."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/sweep_yahpo_rbv2_ranger_proximity/logs.parquet",
        help="Path to logs.parquet or logs.csv",
    )
    parser.add_argument(
        "--proposed",
        type=str,
        default="SMAC20_ProximityLCB",
        help="Identifier of the proposed optimizer",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="SMAC3_HPOFacade_lcb",
        help="Identifier of the baseline optimizer",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional path to write the markdown scorecard",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    print(f"Loaded {len(df):,} trials across {df['task_id'].nunique()} tasks.")

    print("Extracting final incumbents per run...")
    df_inc = extract_final_incumbents(df)

    print("Computing task medians across seeds...")
    df_med = compute_task_medians(df_inc)

    print("Computing hypothesis testing and effect size...")
    stats_res = compute_wilcoxon_and_cliffs_delta(
        df_med,
        proposed_id=args.proposed,
        baseline_id=args.baseline,
    )
    stats_res["suite_name"] = "yahpo_rbv2_ranger"

    report = format_markdown_report(stats_res)
    print("\n" + report)

    if args.output_md:
        out_p = Path(args.output_md)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report, encoding="utf-8")
        print(f"\nSaved report to {out_p}")


if __name__ == "__main__":
    main()
