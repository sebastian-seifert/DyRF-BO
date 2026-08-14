#!/usr/bin/env python3
"""Master Results Parser & Cluster Analyzer for Aleatoric Masterplan Sweep (2,250 Tasks).

Parses output JSON result files in results/aleatoric_masterplan/, averages across 5 seeds,
groups by dimensionality (1D..15D), RF configuration, and noise regime, and outputs
comprehensive Markdown reports and CSV tables comparing Shaker vs. Arithmetic Aleatoric UQ.
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np

def parse_aleatoric_masterplan_results(
    results_dir: str = "results/aleatoric_masterplan",
    output_dir: str = "results"
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Parses result JSON files, computes seed averages, groups by dimensionality and RF config, and outputs reports."""
    json_files = sorted(glob.glob(os.path.join(results_dir, "res_*.json")))
    print(f"==================================================")
    print(f"Aleatoric UQ Masterplan Cluster Analysis")
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
                # jf format: res_{func}_{noise}_{rf_cfg}_seed{s}.json
                parts = os.path.basename(jf).replace(".json", "").split("_")
                # Parse function, noise, rf_cfg, seed from dictionary keys
                for app_name, metrics in data.items():
                    # extract dim from function name
                    func_name = parts[1] if len(parts) > 1 else "unknown"
                    dim = 1
                    for d in range(15, 0, -1):
                        if f"_{d}d" in func_name:
                            dim = d
                            break

                    rec = {
                        "func_name": func_name,
                        "dim": dim,
                        "noise_name": parts[2] if len(parts) > 2 else "unknown",
                        "approach": app_name
                    }
                    rec.update(metrics)
                    records.append(rec)
        except Exception:
            pass

    df = pd.DataFrame(records)
    print(f"Successfully loaded {len(df)} evaluation records across {df['func_name'].nunique()} functions and {df['approach'].nunique()} approaches.")

    metrics = ["spearman_true", "spearman_resid", "log_pearson_true", "mse_var", "rmse_var", "nlpd_aleatoric"]
    present_metrics = [m for m in metrics if m in df.columns]

    os.makedirs(output_dir, exist_ok=True)

    # 1. Save Full CSV Table
    csv_file = os.path.join(output_dir, "aleatoric_masterplan_full_records.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved master records to '{csv_file}'")

    # 2. Group by Approach and Noise Regime
    noise_summary = df.groupby(["approach", "noise_name"])[present_metrics].mean().reset_index()
    noise_csv = os.path.join(output_dir, "aleatoric_masterplan_by_noise.csv")
    noise_summary.to_csv(noise_csv, index=False)

    # 3. Generate Publication Markdown Report
    report_file = os.path.join(output_dir, "aleatoric_masterplan_analysis_report.md")
    with open(report_file, "w") as mf:
        mf.write("# Aleatoric Noise Masterplan Sweep Report: Shaker vs. Arithmetic Leaf Variance\n\n")
        mf.write(f"**Total Records Evaluated**: {len(df)} runs across 15 dimensionalities (1D..15D).\n\n")

        # Heteroscedastic vs Homoscedastic Splits
        hetero_mask = df["noise_name"].str.startswith("hetero_")
        homo_mask = df["noise_name"].str.startswith("homoscedastic_")

        mf.write("## 1. Heteroscedastic Noise Regimes (All Metrics: Spearman, Log-Pearson, MSE, NLPD)\n\n")
        grand_hetero = df[hetero_mask].groupby("approach")[present_metrics].mean().round(4)
        mf.write(grand_hetero.to_markdown() + "\n\n")

        mf.write("## 2. Homoscedastic Noise Regimes (Calibration Metrics: MSE & NLPD)\n\n")
        homo_metrics = ["mse_var", "rmse_var", "nlpd_aleatoric"]
        grand_homo = df[homo_mask].groupby("approach")[homo_metrics].mean().round(4)
        mf.write(grand_homo.to_markdown() + "\n\n")

        mf.write("## 3. Spearman Rank Correlation vs. True Noise (Heteroscedastic Regimes by Noise Type)\n\n")
        piv_noise = noise_summary[noise_summary["noise_name"].str.startswith("hetero_")].pivot(index="approach", columns="noise_name", values="spearman_true")
        mf.write(piv_noise.round(4).to_markdown() + "\n\n")

        mf.write("## 4. Spearman Rank Correlation vs. True Noise by Approach and Dimensionality (1D-15D)\n\n")
        dim_summary = df[hetero_mask].groupby(["approach", "dim"])["spearman_true"].mean().unstack(level=-1)
        mf.write(dim_summary.round(4).to_markdown() + "\n\n")

        mf.write("## 5. NLPD by Approach and Dimensionality (1D-15D)\n\n")
        dim_nlpd = df.groupby(["approach", "dim"])["nlpd_aleatoric"].mean().unstack(level=-1)
        mf.write(dim_nlpd.round(4).to_markdown() + "\n\n")

    print(f"Saved comprehensive report to '{report_file}'")

    # Print summary to console
    print("\n==================================================")
    print("Heteroscedastic Mean Spearman Correlation vs. Ground-Truth Noise")
    print("==================================================")
    print(df[hetero_mask].groupby("approach")["spearman_true"].mean().round(4).to_string())


    return df, noise_summary

if __name__ == "__main__":
    res_dir = sys.argv[1] if len(sys.argv) > 1 else "results/aleatoric_masterplan"
    parse_aleatoric_masterplan_results(results_dir=res_dir)
