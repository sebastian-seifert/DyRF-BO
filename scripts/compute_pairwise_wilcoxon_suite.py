#!/usr/bin/env python3
"""Pairwise Wilcoxon Signed-Rank Test Suite.

Processes CARP-S normalized parquet logs and computes formal non-parametric
statistical tests (Wilcoxon Signed-Rank, Cliff's delta, Holm-Bonferroni correction)
for every epistemic approach (Direct and Additive) against the Standard SMAC3 baseline,
saving results as Markdown, CSV, and LaTeX tables.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame to a GitHub markdown table without requiring tabulate."""
    headers = [str(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_str = [str(val) for val in row.values]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)

def calculate_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cliff's delta non-parametric effect size between two paired/unpaired vectors."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    greater = 0
    less = 0
    for val_x in x:
        greater += np.sum(val_x > y)
        less += np.sum(val_x < y)
    return float((greater - less) / (n_x * n_y))

def apply_holm_bonferroni(raw_p_values: list[float] | np.ndarray) -> list[float]:
    """Applies the step-down Holm-Bonferroni multiple testing correction."""
    p_vals = np.array(raw_p_values, dtype=float)
    n = len(p_vals)
    if n == 0:
        return []
    
    # Sort indices ascending
    sort_idx = np.argsort(p_vals)
    adj_p = np.zeros(n, dtype=float)
    
    for rank_idx, orig_idx in enumerate(sort_idx):
        multiplier = n - rank_idx
        adj_p[orig_idx] = min(1.0, p_vals[orig_idx] * multiplier)
        
    # Enforce monotonicity along sorted order
    sorted_adj = adj_p[sort_idx]
    for i in range(1, n):
        sorted_adj[i] = max(sorted_adj[i], sorted_adj[i - 1])
    adj_p[sort_idx] = sorted_adj
    
    return [float(p) for p in adj_p]

def parse_optimizer_info(opt_id: str) -> tuple[str, str]:
    """Parses paradigm and human-readable extractor name from optimizer_id."""
    if opt_id.startswith("CARPSDynamicRF_AdditiveEpistemic_"):
        parts = opt_id.replace("CARPSDynamicRF_AdditiveEpistemic_", "").split("_")
        acq = parts[0].upper()
        extractor = "_".join(parts[1:])
        return f"Additive Hybrid ({acq})", extractor
    elif opt_id.startswith("SMAC20_CustomUncertainty_"):
        parts = opt_id.replace("SMAC20_CustomUncertainty_", "").split("_")
        acq = parts[0].upper()
        extractor = "_".join(parts[1:])
        return f"Direct Replacement ({acq})", extractor
    elif "SMAC3_HPOFacade" in opt_id:
        acq = opt_id.split("_")[-1].upper()
        return f"Baseline ({acq})", "Standard SMAC3"
    return "Custom", opt_id

def compute_wilcoxon_suite(
    input_parquet: str = "results/ei_comparison_analysis/logs_normalized.parquet",
    output_dir: str = "results/ei_comparison_analysis/statistical_tables",
    baseline_id: str = "SMAC3_HPOFacade_ei"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes comprehensive Wilcoxon Signed-Rank tests for all methods vs Baseline and Head-to-Head."""
    df = pd.read_parquet(input_parquet)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract incumbent normalized cost at final evaluation per task & seed
    cost_col = "trial_value__cost_inc_norm" if "trial_value__cost_inc_norm" in df.columns else "trial_value__cost"
    trials_col = "n_trials" if "n_trials" in df.columns else "trial_id"
    
    final_df = df.sort_values(trials_col).groupby(["task_id", "seed", "optimizer_id"]).last().reset_index()
    
    # Compute mean performance per task across seeds
    task_matrix = final_df.groupby(["task_id", "optimizer_id"])[cost_col].mean().unstack()
    
    if baseline_id not in task_matrix.columns:
        raise ValueError(f"Baseline optimizer '{baseline_id}' not found in dataset columns: {list(task_matrix.columns)}")
    
    baseline_vals = task_matrix[baseline_id]
    n_tasks = len(baseline_vals)
    
    # --- 1. Pairwise vs Baseline Table ---
    all_optimizers = [col for col in task_matrix.columns if col != baseline_id]
    rows_vs_base = []
    
    for opt_id in all_optimizers:
        opt_vals = task_matrix[opt_id]
        paradigm, extractor = parse_optimizer_info(opt_id)
        
        diff = opt_vals - baseline_vals
        wins = int(np.sum(diff < 0))    # Opt has strictly lower regret
        losses = int(np.sum(diff > 0))  # Baseline has strictly lower regret
        ties = int(np.sum(diff == 0))
        
        try:
            res = wilcoxon(opt_vals, baseline_vals, zero_method="wilcox")
            stat, p_val = float(res.statistic), float(res.pvalue)
        except Exception:
            stat, p_val = np.nan, 1.0
            
        cliffs_d = calculate_cliffs_delta(opt_vals.values, baseline_vals.values)
        mean_regret = float(opt_vals.mean())
        base_mean = float(baseline_vals.mean())
        mean_diff = mean_regret - base_mean
        rel_reduction = (mean_diff / base_mean * 100.0) if base_mean != 0 else 0.0
        
        rows_vs_base.append({
            "optimizer_id": opt_id,
            "Paradigm": paradigm,
            "UQ Extractor": extractor,
            "Mean Regret": mean_regret,
            "Mean Diff vs Base": mean_diff,
            "Rel Reduction (%)": rel_reduction,
            "Win / Loss / Tie": f"{wins} / {losses} / {ties}",
            "Wilcoxon W": stat,
            "p_raw": p_val,
            "Cliff's delta": cliffs_d
        })
        
    df_vs_base = pd.DataFrame(rows_vs_base)
    
    # Apply Holm-Bonferroni correction grouped by paradigm
    df_vs_base["Holm-Bonferroni adj p"] = np.nan
    for p_group in df_vs_base["Paradigm"].unique():
        mask = df_vs_base["Paradigm"] == p_group
        raw_p_subset = df_vs_base.loc[mask, "p_raw"].values
        adj_subset = apply_holm_bonferroni(raw_p_subset)
        df_vs_base.loc[mask, "Holm-Bonferroni adj p"] = adj_subset
        
    df_vs_base["Significance (p < 0.05)"] = df_vs_base["p_raw"].apply(lambda p: "Significant (*)" if p < 0.05 else ("Borderline" if p < 0.10 else "Non-significant"))
    df_vs_base = df_vs_base.sort_values(["Paradigm", "p_raw"]).reset_index(drop=True)
    
    # Save Pairwise vs Baseline Tables
    formatted_vs_base = df_vs_base.copy()
    formatted_vs_base["Mean Regret"] = formatted_vs_base["Mean Regret"].apply(lambda v: f"{v:.4f}")
    formatted_vs_base["Mean Diff vs Base"] = formatted_vs_base["Mean Diff vs Base"].apply(lambda v: f"{v:+.4f}")
    formatted_vs_base["Rel Reduction (%)"] = formatted_vs_base["Rel Reduction (%)"].apply(lambda v: f"{v:+.1f}%")
    formatted_vs_base["Wilcoxon W"] = formatted_vs_base["Wilcoxon W"].apply(lambda v: f"{v:.1f}")
    formatted_vs_base["p_raw"] = formatted_vs_base["p_raw"].apply(lambda v: f"{v:.4f}")
    formatted_vs_base["Holm-Bonferroni adj p"] = formatted_vs_base["Holm-Bonferroni adj p"].apply(lambda v: f"{v:.4f}")
    formatted_vs_base["Cliff's delta"] = formatted_vs_base["Cliff's delta"].apply(lambda v: f"{v:+.3f}")
    
    md_file_1 = os.path.join(output_dir, "wilcoxon_tests_vs_baseline.md")
    csv_file_1 = os.path.join(output_dir, "wilcoxon_tests_vs_baseline.csv")
    tex_file_1 = os.path.join(output_dir, "wilcoxon_tests_vs_baseline.tex")
    
    with open(md_file_1, "w") as f:
        f.write(f"# Pairwise Wilcoxon Signed-Rank Tests vs. Standard SMAC3 Baseline (N={n_tasks} Tasks)\n\n")
        f.write(f"Reference Baseline: **`{baseline_id}`** (Mean Normalized Regret: **{baseline_vals.mean():.4f}**)\n\n")
        f.write(dataframe_to_markdown(formatted_vs_base.drop(columns=["optimizer_id"])))
        f.write("\n")
        
    df_vs_base.to_csv(csv_file_1, index=False)
    formatted_vs_base.drop(columns=["optimizer_id"]).to_latex(tex_file_1, index=False)
    
    # --- 2. Head-to-Head (Additive vs Direct) Table ---
    extractors = set()
    for opt_id in all_optimizers:
        _, ext = parse_optimizer_info(opt_id)
        if ext != "Standard SMAC3":
            extractors.add(ext)
            
    rows_h2h = []
    for ext in sorted(extractors):
        add_candidates = [o for o in all_optimizers if "Additive" in o and ext in o]
        dir_candidates = [o for o in all_optimizers if "CustomUncertainty" in o and ext in o]
        
        if add_candidates and dir_candidates:
            add_id = add_candidates[0]
            dir_id = dir_candidates[0]
            add_vals = task_matrix[add_id]
            dir_vals = task_matrix[dir_id]
            
            diff = add_vals - dir_vals
            wins = int(np.sum(diff < 0))    # Additive lower regret
            losses = int(np.sum(diff > 0))  # Direct lower regret
            ties = int(np.sum(diff == 0))
            
            try:
                res = wilcoxon(add_vals, dir_vals, zero_method="wilcox")
                stat, p_val = float(res.statistic), float(res.pvalue)
            except Exception:
                stat, p_val = np.nan, 1.0
                
            cliffs_d = calculate_cliffs_delta(add_vals.values, dir_vals.values)
            add_mean = float(add_vals.mean())
            dir_mean = float(dir_vals.mean())
            
            rows_h2h.append({
                "UQ Extractor": ext,
                "Additive Mean Regret": add_mean,
                "Direct Mean Regret": dir_mean,
                "Diff (Add - Dir)": add_mean - dir_mean,
                "Additive Win/Loss/Tie": f"{wins} / {losses} / {ties}",
                "Wilcoxon W": stat,
                "p_raw": p_val,
                "Cliff's delta": cliffs_d
            })
            
    df_h2h = pd.DataFrame(rows_h2h)
    if not df_h2h.empty:
        df_h2h["Holm-Bonferroni adj p"] = apply_holm_bonferroni(df_h2h["p_raw"].values)
        df_h2h = df_h2h.sort_values("p_raw").reset_index(drop=True)
        
        formatted_h2h = df_h2h.copy()
        formatted_h2h["Additive Mean Regret"] = formatted_h2h["Additive Mean Regret"].apply(lambda v: f"{v:.4f}")
        formatted_h2h["Direct Mean Regret"] = formatted_h2h["Direct Mean Regret"].apply(lambda v: f"{v:.4f}")
        formatted_h2h["Diff (Add - Dir)"] = formatted_h2h["Diff (Add - Dir)"].apply(lambda v: f"{v:+.4f}")
        formatted_h2h["Wilcoxon W"] = formatted_h2h["Wilcoxon W"].apply(lambda v: f"{v:.1f}")
        formatted_h2h["p_raw"] = formatted_h2h["p_raw"].apply(lambda v: f"{v:.4f}")
        formatted_h2h["Holm-Bonferroni adj p"] = formatted_h2h["Holm-Bonferroni adj p"].apply(lambda v: f"{v:.4f}")
        formatted_h2h["Cliff's delta"] = formatted_h2h["Cliff's delta"].apply(lambda v: f"{v:+.3f}")
        
        md_file_2 = os.path.join(output_dir, "wilcoxon_head_to_head_additive_vs_direct.md")
        csv_file_2 = os.path.join(output_dir, "wilcoxon_head_to_head_additive_vs_direct.csv")
        tex_file_2 = os.path.join(output_dir, "wilcoxon_head_to_head_additive_vs_direct.tex")
        
        with open(md_file_2, "w") as f:
            f.write(f"# Head-to-Head Wilcoxon Signed-Rank Tests: Additive Hybrid vs. Direct Replacement (N={n_tasks} Tasks)\n\n")
            f.write(dataframe_to_markdown(formatted_h2h))
            f.write("\n")
            
        df_h2h.to_csv(csv_file_2, index=False)
        formatted_h2h.to_latex(tex_file_2, index=False)
        
    print(f"\n[✓] Generated Wilcoxon statistical suite in: {output_dir}/")
    print(f"    - {md_file_1}")
    print(f"    - {csv_file_1}")
    print(f"    - {tex_file_1}")
    if not df_h2h.empty:
        print(f"    - {md_file_2}")
        print(f"    - {csv_file_2}")
        print(f"    - {tex_file_2}")
        
    return df_vs_base, df_h2h

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Pairwise Wilcoxon Statistical Test Suite")
    parser.add_argument("--input", type=str, default="results/ei_comparison_analysis/logs_normalized.parquet", help="Path to logs_normalized.parquet")
    parser.add_argument("--output", type=str, default="results/ei_comparison_analysis/statistical_tables", help="Directory to write output tables")
    parser.add_argument("--baseline", type=str, default="SMAC3_HPOFacade_ei", help="Optimizer ID of the reference baseline")
    args = parser.parse_args()
    
    compute_wilcoxon_suite(args.input, args.output, args.baseline)
