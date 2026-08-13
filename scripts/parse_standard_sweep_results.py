#!/usr/bin/env python3
"""Master Results Parser for Standard 1D-15D Synthetic OOD Benchmark Sweep (6,930 Tasks).

Parses output JSON files in results/standard_sweep/, computes mean and SEM across 7 seeds
for all 8 evaluation metrics across 55 benchmark functions, and outputs publication-ready summary reports.
"""

import os
import json
import glob
import pandas as pd

def parse_standard_sweep_results(results_dir: str = "results/standard_sweep"):
    json_files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    print(f"Found {len(json_files)} result JSON files in '{results_dir}'")

    if not json_files:
        print("No result JSON files found.")
        return

    records = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
                records.append(data)
        except Exception:
            pass

    df = pd.DataFrame(records)
    print(f"Successfully loaded {len(df)} run records across {df['func_name'].nunique()} benchmark functions.")

    metrics = ["auroc", "aupr", "spearman", "aurc", "brier", "jsd", "mi", "nlpd"]
    grouped = df.groupby(["func_name", "gap_type", "approach"])[metrics].agg(["mean", "std"]).reset_index()

    out_report = "results/standard_sweep_summary.csv"
    grouped.to_csv(out_report, index=False)
    print(f"Saved master CSV summary to '{out_report}'")

    # Print mean AUROC grouped by approach across all functions
    print("\n==================================================")
    print("Standard 1D-15D Sweep - Mean AUROC across all 55 Functions")
    print("==================================================")
    print(df.groupby(["approach", "gap_type"])["auroc"].mean().unstack().round(4).to_string())

    md_report = "results/standard_sweep_summary_report.md"
    with open(md_report, "w") as mf:
        mf.write("# Standard 1D-15D Synthetic OOD Detection Benchmark Report\n\n")
        mf.write(f"Evaluated across {len(df)} total runs (55 functions × 2 gap types × 9 approaches × 7 seeds).\n\n")
        mf.write("## Overall Mean AUROC by Approach and Gap Type\n\n")
        mf.write(df.groupby(["approach", "gap_type"])["auroc"].mean().unstack().round(4).to_markdown() + "\n\n")

    print(f"Saved Markdown report to '{md_report}'")
    return df

if __name__ == "__main__":
    parse_standard_sweep_results()
