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
from scipy.stats import wilcoxon

def df_to_markdown(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Formats a pandas DataFrame as a GitHub Markdown table natively without external dependencies."""
    df_copy = df.copy()
    if df_copy.index.name is not None:
        df_copy = df_copy.reset_index()
    elif any(isinstance(val, str) for val in df_copy.index):
        df_copy = df_copy.reset_index()

    cols = [str(c) for c in df_copy.columns]
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = []
    for _, row in df_copy.iterrows():
        formatted_vals = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                formatted_vals.append(float_format.format(val))
            elif isinstance(val, (int, np.integer)):
                formatted_vals.append(str(val))
            else:
                formatted_vals.append(str(val))
        rows.append("| " + " | ".join(formatted_vals) + " |")

    return "\n".join([header, separator] + rows)

def parse_aleatoric_masterplan_results(
    results_dir: str = "results/aleatoric_masterplan",
    output_dir: str = "results"
) -> tuple[pd.DataFrame | None, dict[str, pd.DataFrame] | None]:
    """Parses result JSON files or summary CSV, computes seed averages, groups by metadata, and outputs full tables & Markdown reports."""
    json_files = sorted(glob.glob(os.path.join(results_dir, "*.json"))) + sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True))
    json_files = sorted(list(set(json_files)))

    # Also check if a single master CSV exists (e.g. aleatoric_masterplan_results.csv)
    csv_master = os.path.join(results_dir, "aleatoric_masterplan_results.csv")
    if not json_files and os.path.exists(csv_master):
        print(f"==================================================")
        print(f"Aleatoric UQ Masterplan Cluster Analysis")
        print(f"Loading master CSV file from '{csv_master}'")
        print(f"==================================================")
        df = pd.read_csv(csv_master)
        if "dim" not in df.columns and "func_name" in df.columns:
            df["dim"] = 1
            for d in range(15, 0, -1):
                df.loc[df["func_name"].str.contains(f"_{d}d"), "dim"] = d
        # Proceed with report generation
        present_metrics = [m for m in ["spearman_true", "spearman_resid", "log_pearson_true", "mse_var", "rmse_var", "nlpd_aleatoric"] if m in df.columns]
        os.makedirs(output_dir, exist_ok=True)
        # Skip loop and generate reports directly
        return _generate_reports_from_df(df, present_metrics, output_dir)

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
                filename = os.path.basename(jf).replace(".json", "")
                parts = filename.split("_")
                # Format: res_{func}_{noise}_{rf_cfg}_seed{s}
                # Find seed
                seed = 1
                if "seed" in parts[-1]:
                    try:
                        seed = int(parts[-1].replace("seed", ""))
                    except ValueError:
                        pass

                rf_config = parts[-2] if len(parts) >= 5 else "unknown"
                noise_name = parts[-3] if len(parts) >= 5 else "unknown"

                # Extract func_name and dimension
                func_parts = parts[1:-2]
                func_name = "_".join(func_parts) if func_parts else "unknown"

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
    metrics = ["spearman_true", "spearman_resid", "log_pearson_true", "mse_var", "rmse_var", "nlpd_aleatoric"]
    present_metrics = [m for m in metrics if m in df.columns]
    return _generate_reports_from_df(df, present_metrics, output_dir)

def _generate_reports_from_df(df: pd.DataFrame, present_metrics: list[str], output_dir: str):
    # 1. Save Full Master CSV Table
    csv_file = os.path.join(output_dir, "aleatoric_masterplan_full_records.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved master records to '{csv_file}'")

    # 2. Table 1: Grand Summary Table (Overall Approach Ranking)
    grand_summary = df.groupby("approach")[present_metrics].agg(["mean", "sem"]).round(4)
    grand_summary_flat = grand_summary.copy()
    grand_summary_flat.columns = [f"{col[0]}_{col[1]}" for col in grand_summary_flat.columns]
    grand_summary_flat.to_csv(os.path.join(output_dir, "aleatoric_masterplan_grand_summary.csv"))

    # 3. Table 2: Breakdown by Noise Regime
    noise_summary = df.groupby(["approach", "noise_name"])[present_metrics].mean().reset_index()
    noise_csv = os.path.join(output_dir, "aleatoric_masterplan_by_noise.csv")
    noise_summary.to_csv(noise_csv, index=False)

    # 4. Table 3: Breakdown by Dimensionality (1D..15D)
    dim_summary = df.groupby(["approach", "dim"])[present_metrics].mean().reset_index()
    dim_csv = os.path.join(output_dir, "aleatoric_masterplan_by_dim.csv")
    dim_summary.to_csv(dim_csv, index=False)

    # 5. Table 4: Breakdown by Random Forest Configuration
    rf_summary = df.groupby(["approach", "rf_config"])[present_metrics].mean().reset_index()
    rf_csv = os.path.join(output_dir, "aleatoric_masterplan_by_rf_config.csv")
    rf_summary.to_csv(rf_csv, index=False)

    # 6. Table 5: Statistical Significance & Win/Tie/Loss Matrix
    baseline = "standard_ari_var"
    wilcoxon_records = []
    if baseline in df["approach"].unique() and "seed" in df.columns:
        group_cols = [c for c in ["func_name", "noise_name", "rf_config", "seed"] if c in df.columns]
        baseline_df = df[df["approach"] == baseline].set_index(group_cols)
        for app in df["approach"].unique():
            if app == baseline:
                continue
            app_df = df[df["approach"] == app].set_index(group_cols)
            common_idx = baseline_df.index.intersection(app_df.index)
            if len(common_idx) > 5:
                b_sp = baseline_df.loc[common_idx, "spearman_true"].values
                a_sp = app_df.loc[common_idx, "spearman_true"].values
                try:
                    res_sp = wilcoxon(a_sp, b_sp, alternative="two-sided")
                    p_val = res_sp.pvalue
                    stat = res_sp.statistic
                except Exception:
                    p_val, stat = 1.0, 0.0

                diff = a_sp - b_sp
                wins = int(np.sum((diff > 1e-4) & (p_val < 0.05)))
                losses = int(np.sum((diff < -1e-4) & (p_val < 0.05)))
                ties = len(common_idx) - wins - losses

                wilcoxon_records.append({
                    "approach": app,
                    "baseline": baseline,
                    "mean_diff_spearman": float(np.mean(diff)),
                    "p_value": float(p_val),
                    "w_statistic": float(stat),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses
                })

    wilcoxon_df = pd.DataFrame(wilcoxon_records)
    wilcoxon_csv = os.path.join(output_dir, "aleatoric_masterplan_wilcoxon.csv")
    wilcoxon_df.to_csv(wilcoxon_csv, index=False)

    # 7. Generate Comprehensive Publication Markdown Report
    report_file = os.path.join(output_dir, "aleatoric_masterplan_analysis_report.md")
    with open(report_file, "w") as mf:
        mf.write("# Aleatoric Noise Masterplan Sweep Report: Shaker Entropy vs. Arithmetic Leaf Variance\n\n")
        num_funcs = df['func_name'].nunique() if 'func_name' in df.columns else 'N/A'
        num_noises = df['noise_name'].nunique() if 'noise_name' in df.columns else 'N/A'
        num_rf = df['rf_config'].nunique() if 'rf_config' in df.columns else 'N/A'
        num_seeds = df['seed'].nunique() if 'seed' in df.columns else 'N/A'
        mf.write(f"**Total Records Evaluated**: {len(df)} runs across {num_funcs} benchmark target functions, ")
        mf.write(f"{num_noises} noise regimes, {num_rf} RF hyperparameter configurations, and {num_seeds} seeds.\n\n")

        mf.write("## 1. Grand Summary Performance Across All Experiments\n\n")
        grand_mean = df.groupby("approach")[present_metrics].mean().round(4)
        mf.write(df_to_markdown(grand_mean) + "\n\n")

        mf.write("## 2. Breakdown by Noise Regime (Spearman Rank Correlation vs. True Noise)\n\n")
        if "noise_name" in df.columns:
            hetero_mask = df["noise_name"].astype(str).str.startswith("hetero_")
            if hetero_mask.any():
                piv_noise = df[hetero_mask].groupby(["approach", "noise_name"])["spearman_true"].mean().unstack(level=-1).round(4)
                mf.write(df_to_markdown(piv_noise) + "\n\n")
            else:
                piv_noise = df.groupby(["approach", "noise_name"])["spearman_true"].mean().unstack(level=-1).round(4)
                mf.write(df_to_markdown(piv_noise) + "\n\n")

        mf.write("## 3. Breakdown by Target Function & Dimensionality (1D-15D)\n\n")
        if "dim" in df.columns:
            piv_dim = df.groupby(["approach", "dim"])["spearman_true"].mean().unstack(level=-1).round(4)
            mf.write(df_to_markdown(piv_dim) + "\n\n")

        mf.write("## 4. Breakdown by Random Forest Configuration\n\n")
        if "rf_config" in df.columns:
            piv_rf = df.groupby(["approach", "rf_config"])["spearman_true"].mean().unstack(level=-1).round(4)
            mf.write(df_to_markdown(piv_rf) + "\n\n")

        mf.write("## 5. Statistical Significance & Win/Tie/Loss Matrix vs. Standard Arithmetic Variance\n\n")
        if not wilcoxon_df.empty:
            mf.write(df_to_markdown(wilcoxon_df) + "\n\n")
        else:
            mf.write("No Wilcoxon matrix calculated (insufficient overlapping seeds).\n\n")

    print(f"Saved comprehensive report to '{report_file}'")

    summaries = {
        "noise": noise_summary,
        "dim": dim_summary,
        "rf_config": rf_summary,
        "wilcoxon": wilcoxon_df
    }

    return df, summaries

if __name__ == "__main__":
    res_dir = sys.argv[1] if len(sys.argv) > 1 else "results/aleatoric_masterplan"
    parse_aleatoric_masterplan_results(results_dir=res_dir)
