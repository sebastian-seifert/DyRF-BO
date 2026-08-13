#!/usr/bin/env python3
"""Master Results Parser for Synthetic OOD Benchmark Sweep (630 Tasks).

Parses output JSON files in results/ood_sweep/, computes mean and SEM across 7 seeds
for all 8 evaluation metrics (AUROC, AUPR, Spearman rho, AURC, Oracle AURC, JSD, MI, Brier),
and outputs publication-ready summary tables.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

def parse_ood_sweep_results(results_dir: str = "results/ood_sweep"):
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
    print(f"Successfully loaded {len(df)} run records.")

    # Aggregate across 7 seeds per (func_name, gap_type, approach) tuple
    metrics = ["auroc", "aupr", "spearman", "aurc", "brier", "jsd", "mi", "nlpd"]
    
    grouped = df.groupby(["func_name", "gap_type", "approach"])[metrics].agg(["mean", "std"]).reset_index()

    # Save CSV and Markdown summary
    out_report = "results/ood_sweep_summary.csv"
    grouped.to_csv(out_report, index=False)
    print(f"Saved master CSV summary to '{out_report}'")

    # Print clean summary per metric for AUROC and Spearman
    print("\n==================================================")
    print("Synthetic OOD Benchmark Sweep - AUROC Summary")
    print("==================================================")
    
    piv_auroc = df.groupby(["func_name", "gap_type", "approach"])["auroc"].mean().unstack(level=-1)
    print(piv_auroc.round(4).to_string())

    print("\n==================================================")
    print("Synthetic OOD Benchmark Sweep - Spearman Rho Summary")
    print("==================================================")
    piv_spearman = df.groupby(["func_name", "gap_type", "approach"])["spearman"].mean().unstack(level=-1)
    print(piv_spearman.round(4).to_string())

    # Write complete Markdown report
    md_report = "results/ood_sweep_summary_report.md"
    with open(md_report, "w") as mf:
        mf.write("# Synthetic OOD Detection Benchmark Report\n\n")
        mf.write(f"Evaluated across {len(df)} total runs (5 functions × 2 gap types × 9 approaches × 7 seeds).\n\n")
        
        mf.write("## 1. AUROC (Out-of-Distribution Discrimination)\n\n")
        mf.write(piv_auroc.round(4).to_markdown() + "\n\n")
        
        mf.write("## 2. Spearman Correlation (U(x) vs |y - y_hat|)\n\n")
        mf.write(piv_spearman.round(4).to_markdown() + "\n\n")

    print(f"\nSaved Markdown report to '{md_report}'")
    return df

if __name__ == "__main__":
    parse_ood_sweep_results()
