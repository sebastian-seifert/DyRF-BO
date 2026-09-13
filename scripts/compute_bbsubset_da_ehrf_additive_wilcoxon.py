#!/usr/bin/env python3
"""Statistical Analysis Suite: Pairwise Wilcoxon Signed-Rank Test & Effect Sizes for DA-EHRF Additive BO.

Processes normalized CARP-S logs and computes:
1. Wilcoxon Signed-Rank Test (paired across tasks)
2. Cliff's delta non-parametric effect size
3. Win / Loss / Tie head-to-head counts
4. Outputs formatted Markdown, CSV, and LaTeX tables.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame to GitHub-flavored markdown without external tabulate dependency."""
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


def compute_bbsubset_da_ehrf_additive_wilcoxon(
    input_file: str = "results/bbsubset_da_ehrf_additive_analysis/logs_normalized.parquet",
    output_dir: str = "results/bbsubset_da_ehrf_additive_analysis/statistical_tables",
    baseline_id: str = "SMAC3_HPOFacade_ei",
    proposed_id: str = "CARPSDynamicRF_DAEHRF_AdditiveEI",
) -> pd.DataFrame:
    """Computes Wilcoxon Signed-Rank Test and Cliff's delta between proposed DA-EHRF Additive EI and SMAC3 baseline.

    Args:
        input_file: Path to parquet or CSV containing CARP-S normalized evaluation logs.
        output_dir: Directory where markdown, CSV, and LaTeX tables are written.
        baseline_id: Optimizer ID for the reference baseline.
        proposed_id: Optimizer ID for the proposed method.

    Returns:
        pd.DataFrame containing summary statistics.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        candidates = [
            input_path.parent / "logs.parquet",
            input_path.parent / "logs_normalized.parquet",
            input_path.parent / "logs.csv",
            Path("results/bbsubset_da_ehrf_additive_analysis/logs.parquet"),
            Path("results/bbsubset_da_ehrf_additive_analysis/logs_normalized.parquet"),
            Path("results/bbsubset_da_ehrf_additive_analysis/logs.csv"),
        ]
        found = False
        for cand in candidates:
            if cand.exists():
                input_path = cand
                input_file = str(cand)
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"Input file not found: {input_file}. Also checked candidates: {[str(c) for c in candidates]}"
            )

    print(f"Reading CARP-S logs from: {input_path}")
    if str(input_file).endswith(".csv"):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_parquet(input_file)

    os.makedirs(output_dir, exist_ok=True)

    # Determine column names flexibly
    task_col = "task_id" if "task_id" in df.columns else ("task" if "task" in df.columns else "benchmark_id")
    opt_col = "optimizer_id" if "optimizer_id" in df.columns else ("optimizer" if "optimizer" in df.columns else "optimizer_name")
    seed_col = "seed" if "seed" in df.columns else ("trial_info__seed" if "trial_info__seed" in df.columns else "seed")
    cost_col = (
        "trial_value__cost_inc_norm"
        if "trial_value__cost_inc_norm" in df.columns
        else ("trial_value__cost" if "trial_value__cost" in df.columns else "cost")
    )
    trials_col = "n_trials" if "n_trials" in df.columns else ("trial_id" if "trial_id" in df.columns else ("trial" if "trial" in df.columns else None))

    if trials_col:
        final_df = df.sort_values(trials_col).groupby([task_col, seed_col, opt_col]).last().reset_index()
    else:
        final_df = df.groupby([task_col, seed_col, opt_col]).last().reset_index()

    # Aggregate mean final regret per task across seeds
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
    wins = int(np.sum(diff < 0))    # Proposed strictly lower regret (better)
    losses = int(np.sum(diff > 0))  # Baseline strictly lower regret
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
        "Proposed Mean Regret": mean_prop,
        "Baseline Mean Regret": mean_base,
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
    formatted_df["Proposed Mean Regret"] = formatted_df["Proposed Mean Regret"].apply(lambda v: f"{v:.4f}")
    formatted_df["Baseline Mean Regret"] = formatted_df["Baseline Mean Regret"].apply(lambda v: f"{v:.4f}")
    formatted_df["Mean Diff vs Base"] = formatted_df["Mean Diff vs Base"].apply(lambda v: f"{v:+.4f}")
    formatted_df["Rel Reduction (%)"] = formatted_df["Rel Reduction (%)"].apply(lambda v: f"{v:+.1f}%")
    formatted_df["Wilcoxon W"] = formatted_df["Wilcoxon W"].apply(lambda v: f"{v:.1f}")
    formatted_df["p_val"] = formatted_df["p_val"].apply(lambda v: f"{v:.4f}")
    formatted_df["Cliff's delta"] = formatted_df["Cliff's delta"].apply(lambda v: f"{v:+.3f}")

    # Write files
    md_file = os.path.join(output_dir, "wilcoxon_results.md")
    csv_file = os.path.join(output_dir, "wilcoxon_results.csv")
    tex_file = os.path.join(output_dir, "wilcoxon_results.tex")

    with open(md_file, "w") as f:
        f.write("# Wilcoxon Signed-Rank Test: DA-EHRF Additive EI vs. SMAC3 Baseline\n\n")
        f.write(f"- Reference Baseline: `{baseline_id}`\n")
        f.write(f"- Proposed Model: `{proposed_id}`\n")
        f.write(f"- Evaluated Tasks: {n_tasks} tasks\n\n")
        f.write(dataframe_to_markdown(formatted_df))
        f.write("\n")

    res_df.to_csv(csv_file, index=False)
    formatted_df.to_latex(tex_file, index=False)

    print(f"Wilcoxon results successfully written to {output_dir}")
    return res_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute Wilcoxon Signed-Rank Test for CARP-S DA-EHRF Additive EI vs Baseline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/bbsubset_da_ehrf_additive_analysis/logs_normalized.parquet",
        help="Input normalized CARP-S logs (.parquet or .csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/bbsubset_da_ehrf_additive_analysis/statistical_tables",
        help="Output directory for statistical tables",
    )
    parser.add_argument(
        "--baseline_id",
        type=str,
        default="SMAC3_HPOFacade_ei",
        help="Optimizer ID for baseline",
    )
    parser.add_argument(
        "--proposed_id",
        type=str,
        default="CARPSDynamicRF_DAEHRF_AdditiveEI",
        help="Optimizer ID for proposed method",
    )

    args = parser.parse_args()
    compute_bbsubset_da_ehrf_additive_wilcoxon(
        input_file=args.input,
        output_dir=args.output_dir,
        baseline_id=args.baseline_id,
        proposed_id=args.proposed_id,
    )


if __name__ == "__main__":
    main()
