#!/usr/bin/env python3
"""
Results Parser and Statistical Suite for Aleatoric OOD Masterplan Sweep.

Aggregates sliced metrics across Global, In-Distribution, and Out-of-Distribution partitions.
Generates publication-ready CSV tables and a formatted Markdown report inside results/OOD_Aleatoric_Sweep/.
"""

import os
import glob
import json
import argparse
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

def parse_aleatoric_ood_results(
    json_dir: str = "results/OOD_Aleatoric_Sweep/json",
    output_dir: str = "results/OOD_Aleatoric_Sweep"
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(json_dir, "res_*.json"))

    # Fallback to master CSV if no individual JSONs found
    csv_master = os.path.join(output_dir, "aleatoric_ood_masterplan_full_records.csv")
    if not json_files and os.path.exists(csv_master):
        print(f"==================================================")
        print(f"Aleatoric OOD Masterplan Cluster Analysis")
        print(f"Loading master CSV file from '{csv_master}'")
        print(f"==================================================")
        df = pd.read_csv(csv_master)
        return _generate_ood_reports(df, output_dir)

    if not json_files:
        print(f"No JSON result files found in '{json_dir}'.")
        return None, None

    print(f"Found {len(json_files)} result JSON files in '{json_dir}'. Parsing...")

    records = []
    for jf in json_files:
        try:
            with open(jf, "r") as fp:
                data = json.load(fp)

            filename = os.path.basename(jf)
            # Example filename: res_sin_cos_1d_hetero_ood_step_double_RF_Default_seed1.json
            name_core = filename.replace("res_", "").replace(".json", "")
            parts = name_core.split("_")

            seed = 1
            for p in parts:
                if p.startswith("seed"):
                    try:
                        seed = int(p.replace("seed", ""))
                    except ValueError:
                        pass

            func_name = "unknown"
            for d in range(15, 0, -1):
                if f"sin_cos_{d}d" in name_core:
                    func_name = f"sin_cos_{d}d"
                    break

            noise_name = "unknown"
            for n_candidate in [
                "hetero_ood_step_double", "hetero_sinusoidal", "hetero_localized",
                "hetero_quadratic", "hetero_linear", "homoscedastic_low", "homoscedastic_high"
            ]:
                if n_candidate in name_core:
                    noise_name = n_candidate
                    break

            rf_config = "RF_Default"
            for rf_cand in ["RF_Overfit_Leaf1", "RF_Smoothed_Leaf15", "RF_Shallow", "RF_DeepEnsemble300", "RF_Default"]:
                if rf_cand in name_core:
                    rf_config = rf_cand
                    break

            dim = 1
            for d in range(15, 0, -1):
                if f"_{d}d" in filename or f"_{d}d" in func_name:
                    dim = d
                    break

            for app_name, metrics in data.items():
                rec = {
                    "func_name": func_name,
                    "dim": dim,
                    "noise_name": noise_name,
                    "rf_config": rf_config,
                    "seed": seed,
                    "approach": app_name
                }
                rec.update(metrics)
                records.append(rec)
        except Exception as e:
            print(f"Warning: Failed to parse {jf}: {e}")

    if not records:
        print("No valid evaluation records extracted.")
        return None, None

    df = pd.DataFrame(records)
    print(f"Successfully loaded {len(df)} evaluation records across {df['func_name'].nunique()} functions and {df['approach'].nunique()} approaches.")

    return _generate_ood_reports(df, output_dir)

def _df_to_markdown(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Dependency-free DataFrame to Markdown table converter."""
    df_copy = df.copy()
    if df_copy.index.name is not None or any(isinstance(val, str) for val in df_copy.index):
        df_copy = df_copy.reset_index()

    cols = [str(c) for c in df_copy.columns]
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, separator]
    for _, row in df_copy.iterrows():
        formatted_vals = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                formatted_vals.append(float_format.format(val))
            elif isinstance(val, (int, np.integer)):
                formatted_vals.append(str(val))
            else:
                formatted_vals.append(str(val))
        lines.append("| " + " | ".join(formatted_vals) + " |")
    return "\n".join(lines)

def _generate_ood_reports(df: pd.DataFrame, output_dir: str):
    # 1. Full master CSV
    csv_file = os.path.join(output_dir, "aleatoric_ood_masterplan_full_records.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved master records to '{csv_file}'")

    metric_cols = [c for c in df.columns if any(c.startswith(p) for p in ["global_", "id_only_", "ood_only_", "ood_id_"])]

    # 2. Grand Summary Table
    grand_summary = df.groupby("approach")[metric_cols].agg(["mean", "sem"]).round(4)
    grand_summary_flat = grand_summary.copy()
    grand_summary_flat.columns = [f"{col[0]}_{col[1]}" for col in grand_summary_flat.columns]
    grand_summary_flat.to_csv(os.path.join(output_dir, "aleatoric_ood_masterplan_grand_summary.csv"))

    # 3. Breakdown by Noise Regime
    noise_summary = df.groupby(["approach", "noise_name"])[metric_cols].mean().reset_index()
    noise_summary.to_csv(os.path.join(output_dir, "aleatoric_ood_masterplan_by_noise.csv"), index=False)

    # 4. Breakdown by Dimensionality (1D..15D)
    dim_summary = df.groupby(["approach", "dim"])[metric_cols].mean().reset_index()
    dim_summary.to_csv(os.path.join(output_dir, "aleatoric_ood_masterplan_by_dim.csv"), index=False)

    # 5. Breakdown by Random Forest Configuration
    rf_summary = df.groupby(["approach", "rf_config"])[metric_cols].mean().reset_index()
    rf_summary.to_csv(os.path.join(output_dir, "aleatoric_ood_masterplan_by_rf_config.csv"), index=False)

    # 6. Statistical Significance & Win/Tie/Loss Matrix vs. standard_ari_var
    wilcoxon_records = []
    baseline = "standard_ari_var"
    if baseline in df["approach"].unique():
        group_cols = [c for c in ["func_name", "noise_name", "rf_config", "seed"] if c in df.columns]
        baseline_df = df[df["approach"] == baseline].set_index(group_cols)
        for app in df["approach"].unique():
            if app == baseline:
                continue
            app_df = df[df["approach"] == app].set_index(group_cols)
            common_idx = baseline_df.index.intersection(app_df.index)
            if len(common_idx) >= 2:
                for target_metric in ["id_only_spearman_true", "ood_only_spearman_true", "ood_id_variance_ratio"]:
                    if target_metric not in df.columns:
                        continue
                    b_val = baseline_df.loc[common_idx, target_metric].values
                    a_val = app_df.loc[common_idx, target_metric].values
                    
                    diff = a_val - b_val
                    if len(np.unique(diff)) > 1:
                        try:
                            res = wilcoxon(a_val, b_val, alternative="two-sided")
                            p_val, stat = float(res.pvalue), float(res.statistic)
                        except Exception:
                            p_val, stat = 1.0, 0.0
                    else:
                        p_val, stat = 1.0, 0.0

                    wins = int(np.sum((diff > 1e-4) & (p_val < 0.05)))
                    losses = int(np.sum((diff < -1e-4) & (p_val < 0.05)))
                    ties = len(common_idx) - wins - losses

                    wilcoxon_records.append({
                        "approach": app,
                        "baseline": baseline,
                        "metric": target_metric,
                        "mean_diff": float(np.mean(diff)),
                        "p_value": float(p_val),
                        "w_statistic": float(stat),
                        "wins": wins,
                        "ties": ties,
                        "losses": losses
                    })
    wilcoxon_df = pd.DataFrame(wilcoxon_records)
    wilcoxon_csv = os.path.join(output_dir, "aleatoric_ood_masterplan_wilcoxon.csv")
    wilcoxon_df.to_csv(wilcoxon_csv, index=False)

    # 7. Generate Comprehensive Markdown Report
    report_file = os.path.join(output_dir, "aleatoric_ood_masterplan_analysis_report.md")
    with open(report_file, "w") as rf:
        rf.write("# Aleatoric OOD Masterplan Sweep Report: In-Distribution vs. Out-of-Distribution Analysis\n\n")
        rf.write(f"**Total Records Evaluated**: {len(df)} runs across {df['func_name'].nunique()} benchmark target functions, {df['noise_name'].nunique()} noise regimes, {df['rf_config'].nunique()} RF configs, and {df['seed'].nunique()} seeds.\n\n")

        rf.write("## 1. Grand Summary: In-Distribution (ID) vs. Out-of-Distribution (OOD) Scopes\n\n")
        key_metrics = [
            "id_only_spearman_true", "ood_only_spearman_true",
            "id_only_mse_var", "ood_only_mse_var",
            "id_only_nlpd_aleatoric", "ood_only_nlpd_aleatoric",
            "ood_id_variance_ratio"
        ]
        present_key = [m for m in key_metrics if m in df.columns]
        summary_table = df.groupby("approach")[present_key].mean().round(4).reset_index()
        rf.write(_df_to_markdown(summary_table))
        rf.write("\n\n")

        rf.write("## 2. OOD / ID Variance Ratio (Epistemic Explosion vs. Aleatoric Extrapolation)\n\n")
        ratio_table = df.groupby("approach")["ood_id_variance_ratio"].agg(["mean", "std"]).round(4).reset_index()
        ratio_table.columns = ["approach", "ratio_mean", "ratio_std"]
        rf.write(_df_to_markdown(ratio_table))
        rf.write("\n\n")

        rf.write("## 3. Breakdown by Noise Regime (Spearman Rank Correlation)\n\n")
        if "id_only_spearman_true" in df.columns:
            noise_pivot = df.pivot_table(index="approach", columns="noise_name", values="id_only_spearman_true", aggfunc="mean").round(4).reset_index()
            rf.write("### ID-Only Spearman Rank Correlation vs. True Noise:\n\n")
            rf.write(_df_to_markdown(noise_pivot))
            rf.write("\n\n")

        if "ood_only_spearman_true" in df.columns:
            noise_pivot_ood = df.pivot_table(index="approach", columns="noise_name", values="ood_only_spearman_true", aggfunc="mean").round(4).reset_index()
            rf.write("### OOD-Only Spearman Rank Correlation vs. True Noise:\n\n")
            rf.write(_df_to_markdown(noise_pivot_ood))
            rf.write("\n\n")

        rf.write("## 4. Breakdown by Random Forest Configuration\n\n")
        if "id_only_spearman_true" in df.columns:
            rf_pivot = df.pivot_table(index="approach", columns="rf_config", values="id_only_spearman_true", aggfunc="mean").round(4).reset_index()
            rf.write(_df_to_markdown(rf_pivot))
            rf.write("\n\n")

        if not wilcoxon_df.empty:
            rf.write("## 5. Statistical Significance & Win/Tie/Loss Matrix vs. Standard Arithmetic Variance\n\n")
            rf.write(_df_to_markdown(wilcoxon_df))
            rf.write("\n\n")

    print(f"Generated comprehensive report in '{report_file}'.")
    return df, report_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Aleatoric OOD Masterplan results.")
    parser.add_argument("--json_dir", type=str, default="results/OOD_Aleatoric_Sweep/json", help="Path to JSON results directory.")
    parser.add_argument("--output_dir", type=str, default="results/OOD_Aleatoric_Sweep", help="Path to output summary reports.")
    args = parser.parse_args()

    parse_aleatoric_ood_results(json_dir=args.json_dir, output_dir=args.output_dir)
