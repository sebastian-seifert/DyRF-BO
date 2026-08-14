#!/usr/bin/env python3
"""Master Results Parser & Cluster Analyzer for Standard 1D-15D Synthetic Sweep.

Aggregates all output JSON files in results/standard_sweep/, averages across seeds,
maps function names to dimensionalities (1D..15D), and generates comprehensive tables
for EVERY metric (AUROC, AUPR, Spearman, AURC, Oracle AURC, JSD, MI, Brier, NLPD) across
every approach and dimensionality.
"""

import os
import sys
import json
import glob
import re
import pandas as pd
import numpy as np

def extract_dim_from_func_name(func_name: str) -> int:
    """Extracts dimensionality integer (1 to 15) from standard synthetic function names."""
    # Check explicit dimension suffix like _4d, _15d, etc.
    match = re.search(r'_(\d+)d$', func_name)
    if match:
        return int(match.group(1))

    # Known 1D and 2D function names
    names_1d = {"sin", "cos_trend", "poly", "damped_osc", "log_mod"}
    names_2d = {"sin_cos", "quadratic", "sin_sum_mod", "gaussian", "abs_sin", "ackley_2d", "rosenbrock_2d"}
    
    if func_name in names_1d:
        return 1
    if func_name in names_2d:
        return 2

    return 1

def df_to_markdown_clean(df: pd.DataFrame) -> str:
    """Formats DataFrame as a clean GitHub Markdown table without requiring tabulate."""
    try:
        return df.to_markdown()
    except Exception:
        # Fallback manual markdown table generator
        headers = [str(col) for col in df.columns]
        if df.index.name or not isinstance(df.index, pd.RangeIndex):
            headers = [df.index.name or ""] + headers
            rows = [[str(idx)] + [str(val) for val in row] for idx, row in zip(df.index, df.values)]
        else:
            rows = [[str(val) for val in row] for row in df.values]
            
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_lines = ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join([header_line, sep_line] + body_lines)

def parse_standard_sweep_results(
    results_dir: str = "results/standard_sweep",
    output_dir: str = "results"
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Parses result JSON files, computes seed averages, groups by dimensionality, and saves tables."""
    json_files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    print(f"==================================================")
    print(f"Standard 1D-15D Sweep Cluster Analysis")
    print(f"Found {len(json_files)} result JSON files in '{results_dir}'")
    print(f"==================================================")

    if not json_files:
        print("No result JSON files found. Please ensure sweep tasks have completed.")
        return None, None

    records = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
                data["dim"] = extract_dim_from_func_name(data["func_name"])
                records.append(data)
        except Exception as e:
            pass

    df = pd.DataFrame(records)
    print(f"Successfully loaded {len(df)} run records across {df['func_name'].nunique()} benchmark functions and {df['dim'].nunique()} dimensionalities.")

    all_metrics = ["auroc", "aupr", "spearman", "aurc", "oracle_aurc", "jsd", "mi", "brier", "nlpd"]
    present_metrics = [m for m in all_metrics if m in df.columns]

    os.makedirs(output_dir, exist_ok=True)

    # 1. Average across seeds per (dim, func_name, gap_type, approach)
    func_summary = df.groupby(["dim", "func_name", "gap_type", "approach"])[present_metrics].mean().reset_index()
    func_csv = os.path.join(output_dir, "standard_sweep_per_function_averages.csv")
    func_summary.to_csv(func_csv, index=False)
    print(f"Saved per-function seed averages to '{func_csv}'")

    # 2. Dimensionality Average: Group by (dim, gap_type, approach)
    dim_summary = df.groupby(["dim", "gap_type", "approach"])[present_metrics].agg(["mean", "sem"]).reset_index()
    dim_csv = os.path.join(output_dir, "standard_sweep_per_dim_summary.csv")
    dim_summary.to_csv(dim_csv, index=False)
    print(f"Saved per-dimension summary table to '{dim_csv}'")

    # Flatten dim_summary columns for pretty printing and markdown report
    dim_mean = df.groupby(["dim", "gap_type", "approach"])[present_metrics].mean().reset_index()

    # Generate Publication-Ready Markdown Report
    report_file = os.path.join(output_dir, "standard_sweep_analysis_report.md")
    with open(report_file, "w") as mf:
        mf.write("# Standard 1D-15D Synthetic OOD Detection Benchmark Report\n\n")
        mf.write(f"**Total Evaluation Runs**: {len(df)} records across {df['dim'].nunique()} dimensionalities (1D..15D).\n")
        mf.write(f"**Evaluated Metrics**: {', '.join(present_metrics)}\n\n")

        for metric in present_metrics:
            mf.write(f"## Metric: {metric.upper()}\n\n")
            mf.write(f"### Mean {metric.upper()} by Approach and Dimensionality (Empty Gap)\n\n")
            
            piv_empty = dim_mean[dim_mean["gap_type"] == "empty"].pivot(index="approach", columns="dim", values=metric)
            mf.write(df_to_markdown_clean(piv_empty.round(4)) + "\n\n")

            mf.write(f"### Mean {metric.upper()} by Approach and Dimensionality (Sparse Gap)\n\n")
            piv_sparse = dim_mean[dim_mean["gap_type"] == "sparse"].pivot(index="approach", columns="dim", values=metric)
            mf.write(df_to_markdown_clean(piv_sparse.round(4)) + "\n\n")

        # Grand Mean across all dimensions
        mf.write("## Overall Grand Mean across All Dimensionalities (1D-15D)\n\n")
        grand_mean = df.groupby(["approach", "gap_type"])[present_metrics].mean().unstack(level=-1)
        mf.write(df_to_markdown_clean(grand_mean.round(4)) + "\n\n")

    print(f"Saved comprehensive Markdown report to '{report_file}'")


    # Print AUROC & Spearman summary to console
    print("\n==================================================")
    print("Mean AUROC by Approach and Dimensionality (Empty Gap)")
    print("==================================================")
    piv_auroc = dim_mean[dim_mean["gap_type"] == "empty"].pivot(index="approach", columns="dim", values="auroc")
    print(piv_auroc.round(4).to_string())

    print("\n==================================================")
    print("Mean Spearman Correlation by Approach and Dimensionality (Empty Gap)")
    print("==================================================")
    piv_spearman = dim_mean[dim_mean["gap_type"] == "empty"].pivot(index="approach", columns="dim", values="spearman")
    print(piv_spearman.round(4).to_string())

    return df, dim_mean

if __name__ == "__main__":
    res_dir = sys.argv[1] if len(sys.argv) > 1 else "results/standard_sweep"
    parse_standard_sweep_results(results_dir=res_dir)
