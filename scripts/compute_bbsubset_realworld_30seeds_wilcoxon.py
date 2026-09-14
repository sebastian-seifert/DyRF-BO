#!/usr/bin/env python3
"""Statistical Analysis Suite: Pairwise Wilcoxon Signed-Rank Test & Effect Sizes for 30-Seed Real-World ML Benchmark.

Computes:
1. Wilcoxon Signed-Rank Test paired across tasks using final incumbent costs (`trial_value__cost_inc`).
2. Cliff's delta non-parametric effect size.
3. Head-to-head Win / Loss / Tie counts.
4. Summary and per-task breakdown tables in Markdown, CSV, and LaTeX.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame to GitHub-flavored markdown table without external tabulate dependency."""
    headers = [str(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_str = [str(val) for val in row.values]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)


def calculate_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cliff's delta non-parametric effect size between two paired vectors.

    delta = (sum(x_i > y_j) - sum(x_i < y_j)) / (n_x * n_y)
    Negative delta means x values are systematically smaller (better for minimization).
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
    """Resolves input file path with fallback candidates."""
    input_path = Path(input_file)
    if input_path.exists():
        return str(input_path)

    candidates = [
        input_path.parent / "logs_normalized.parquet",
        input_path.parent / "logs.parquet",
        input_path.parent / "logs_normalized.csv",
        input_path.parent / "logs.csv",
        Path("results/bbsubset_realworld_30seeds_analysis/logs_normalized.parquet"),
        Path("results/bbsubset_realworld_30seeds_analysis/logs.parquet"),
        Path("results/bbsubset_realworld_30seeds_analysis/logs_normalized.csv"),
        Path("results/bbsubset_realworld_30seeds_analysis/logs.csv"),
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)

    raise FileNotFoundError(
        f"Input file not found: {input_file}. Also checked candidates: {[str(c) for c in candidates]}"
    )


def compute_bbsubset_realworld_30seeds_wilcoxon(
    input_file: str = "results/bbsubset_realworld_30seeds_analysis/logs_normalized.parquet",
    output_dir: str = "results/bbsubset_realworld_30seeds_analysis/statistical_tables",
    baseline_id: str = "SMAC3_HPOFacade_ei",
    proposed_id: str = "CARPSDynamicRF_DAEHRF_AdditiveEI",
) -> pd.DataFrame:
    """Computes Wilcoxon Signed-Rank Test and Cliff's delta between proposed DA-EHRF Additive EI and baseline SMAC3.

    Args:
        input_file: Path to parquet or CSV containing CARP-S logs.
        output_dir: Directory where markdown, CSV, and LaTeX tables are written.
        baseline_id: Optimizer ID for the reference baseline.
        proposed_id: Optimizer ID for the proposed method.

    Returns:
        pd.DataFrame containing summary statistics.
    """
    resolved_input = find_input_file(input_file)
    print(f"Reading CARP-S logs from: {resolved_input}")

    if resolved_input.endswith(".csv"):
        df = pd.read_csv(resolved_input)
    else:
        df = pd.read_parquet(resolved_input)

    os.makedirs(output_dir, exist_ok=True)

    # Determine column names flexibly, prioritizing incumbent values
    task_col = next((c for c in ["task_id", "task", "benchmark_id"] if c in df.columns), "task_id")
    opt_col = next((c for c in ["optimizer_id", "optimizer", "optimizer_name"] if c in df.columns), "optimizer_id")
    seed_col = next((c for c in ["seed", "trial_info__seed"] if c in df.columns), "seed")

    # Metric priority: trial_value__cost_inc -> trial_value__cost_inc_norm -> cost_inc -> trial_value__cost -> cost
    cost_candidates = [
        "trial_value__cost_inc",
        "trial_value__cost_inc_norm",
        "cost_inc",
        "trial_value__cost",
        "cost",
    ]
    cost_col = next((c for c in cost_candidates if c in df.columns), None)
    if cost_col is None:
        raise ValueError(f"Could not find a valid cost column in {df.columns}. Checked: {cost_candidates}")

    print(f"Using metric column: '{cost_col}' for statistical evaluation.")

    trials_col = next((c for c in ["n_trials", "trial_id", "trial"] if c in df.columns), None)

    if trials_col:
        final_df = df.sort_values(trials_col).groupby([task_col, seed_col, opt_col]).last().reset_index()
    else:
        final_df = df.groupby([task_col, seed_col, opt_col]).last().reset_index()

    # Aggregate mean final incumbent cost per task across seeds
    task_matrix = final_df.groupby([task_col, opt_col])[cost_col].mean().unstack()

    # Resolve column names if exact match is not found but prefix matches
    if baseline_id not in task_matrix.columns:
        matching = [c for c in task_matrix.columns if c.startswith(baseline_id) or baseline_id.startswith(c)]
        if matching:
            baseline_id = matching[0]
        else:
            raise ValueError(f"Baseline optimizer '{baseline_id}' not found in data columns: {list(task_matrix.columns)}")

    if proposed_id not in task_matrix.columns:
        matching = [c for c in task_matrix.columns if c.startswith(proposed_id) or proposed_id.startswith(c)]
        if matching:
            proposed_id = matching[0]
        else:
            raise ValueError(f"Proposed optimizer '{proposed_id}' not found in data columns: {list(task_matrix.columns)}")

    # Clean rows with complete paired evaluations
    valid_tasks = task_matrix.dropna(subset=[baseline_id, proposed_id])
    base_vals = valid_tasks[baseline_id].values
    prop_vals = valid_tasks[proposed_id].values
    n_tasks = len(valid_tasks)

    diff = prop_vals - base_vals
    wins = int(np.sum(diff < 0))    # Proposed strictly lower regret/cost (better)
    losses = int(np.sum(diff > 0))  # Baseline strictly lower regret/cost
    ties = int(np.sum(diff == 0))

    try:
        res = wilcoxon(prop_vals, base_vals, zero_method="wilcox")
        stat, p_val = float(res.statistic), float(res.pvalue)
    except Exception:
        stat, p_val = np.nan, 1.0

    cliffs_d = calculate_cliffs_delta(prop_vals, base_vals)
    mean_prop = float(np.mean(prop_vals))
    mean_base = float(np.mean(base_vals))
    mean_diff = mean_prop - mean_base
    rel_reduction = (mean_diff / mean_base * 100.0) if mean_base != 0 else 0.0

    summary_records = [{
        "Proposed Optimizer": proposed_id,
        "Baseline Optimizer": baseline_id,
        "N Tasks": n_tasks,
        "Proposed Mean Incumbent": mean_prop,
        "Baseline Mean Incumbent": mean_base,
        "Mean Diff vs Base": mean_diff,
        "Rel Reduction (%)": rel_reduction,
        "Win / Loss / Tie": f"{wins} / {losses} / {ties}",
        "Wilcoxon W": stat,
        "p_val": p_val,
        "Cliff's delta": cliffs_d,
        "Significance (p < 0.05)": "Significant (*)" if p_val < 0.05 else "Non-significant",
    }]

    res_df = pd.DataFrame(summary_records)

    # Format dataframe for display
    formatted_df = res_df.copy()
    formatted_df["Proposed Mean Incumbent"] = formatted_df["Proposed Mean Incumbent"].apply(lambda v: f"{v:.4f}")
    formatted_df["Baseline Mean Incumbent"] = formatted_df["Baseline Mean Incumbent"].apply(lambda v: f"{v:.4f}")
    formatted_df["Mean Diff vs Base"] = formatted_df["Mean Diff vs Base"].apply(lambda v: f"{v:+.4f}")
    formatted_df["Rel Reduction (%)"] = formatted_df["Rel Reduction (%)"].apply(lambda v: f"{v:+.1f}%")
    formatted_df["Wilcoxon W"] = formatted_df["Wilcoxon W"].apply(lambda v: f"{v:.1f}")
    formatted_df["p_val"] = formatted_df["p_val"].apply(lambda v: f"{v:.4f}")
    formatted_df["Cliff's delta"] = formatted_df["Cliff's delta"].apply(lambda v: f"{v:+.3f}")

    # Build per-task breakdown
    task_breakdown = valid_tasks.copy()
    task_breakdown["Diff (Prop - Base)"] = task_breakdown[proposed_id] - task_breakdown[baseline_id]
    task_breakdown["Outcome"] = np.where(
        task_breakdown["Diff (Prop - Base)"] < 0,
        "Win (Proposed)",
        np.where(task_breakdown["Diff (Prop - Base)"] > 0, "Loss (Baseline)", "Tie"),
    )
    task_breakdown = task_breakdown.reset_index()

    # Write files
    md_file = os.path.join(output_dir, "wilcoxon_results.md")
    csv_file = os.path.join(output_dir, "wilcoxon_results.csv")
    tex_file = os.path.join(output_dir, "wilcoxon_results.tex")
    task_csv = os.path.join(output_dir, "task_breakdown.csv")

    with open(md_file, "w") as f:
        f.write("# Wilcoxon Signed-Rank Test: Real-World ML Benchmark Suite (30 Seeds)\n\n")
        f.write(f"- Reference Baseline: `{baseline_id}`\n")
        f.write(f"- Proposed Model: `{proposed_id}`\n")
        f.write(f"- Evaluated Real-World Tasks: {n_tasks}\n")
        f.write(f"- Metric: `{cost_col}` (Final Incumbent Cost, Minimization)\n\n")
        f.write("### Summary Statistics\n\n")
        f.write(dataframe_to_markdown(formatted_df))
        f.write("\n\n### Per-Task Breakdown\n\n")
        f.write(dataframe_to_markdown(task_breakdown.round(5)))
        f.write("\n")

    res_df.to_csv(csv_file, index=False)
    task_breakdown.to_csv(task_csv, index=False)
    formatted_df.to_latex(tex_file, index=False)

    print(f"Wilcoxon statistical tables successfully written to {output_dir}")
    return res_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute Wilcoxon Signed-Rank Test for 30-Seed Real-World ML Benchmark Suite"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/bbsubset_realworld_30seeds_analysis/logs_normalized.parquet",
        help="Input normalized CARP-S logs (.parquet or .csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/bbsubset_realworld_30seeds_analysis/statistical_tables",
        help="Output directory for statistical tables",
    )
    parser.add_argument(
        "--baseline_id",
        type=str,
        default="SMAC3_HPOFacade_ei",
        help="Optimizer ID for baseline (default: SMAC3_HPOFacade_ei)",
    )
    parser.add_argument(
        "--proposed_id",
        type=str,
        default="CARPSDynamicRF_DAEHRF_AdditiveEI",
        help="Optimizer ID for proposed method (default: CARPSDynamicRF_DAEHRF_AdditiveEI)",
    )

    args = parser.parse_args()
    compute_bbsubset_realworld_30seeds_wilcoxon(
        input_file=args.input,
        output_dir=args.output_dir,
        baseline_id=args.baseline_id,
        proposed_id=args.proposed_id,
    )


if __name__ == "__main__":
    main()
