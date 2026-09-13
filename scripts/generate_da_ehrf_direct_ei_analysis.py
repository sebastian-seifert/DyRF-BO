#!/usr/bin/env python3
"""Statistical Analysis and Anytime Regret Plotting for CARPS DA-EHRF Direct EI Benchmark.

Generates:
1. Paired Wilcoxon signed-rank test and Cliff's delta effect size table with Holm-Bonferroni correction.
2. Formatted CSV and JSON statistical reports.
3. Publication-grade 4-panel anytime regret curves with mean ± SEM and median ± IQR shaded bands.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def calculate_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cliff's delta non-parametric effect size between two vectors: (P(x > y) - P(x < y))."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    greater = 0
    less = 0
    for val_x in x:
        greater += np.sum(val_x > y)
        less += np.sum(val_x < y)
    return float((greater - less) / (n_x * n_y))


def classify_cliffs_delta(delta: float) -> str:
    """Classifies magnitude of Cliff's delta according to Romano et al. (2006)."""
    abs_d = abs(delta)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"


def apply_holm_bonferroni(raw_p_values: list[float] | np.ndarray) -> list[float]:
    """Applies step-down Holm-Bonferroni correction to multiple testing p-values."""
    p_vals = np.array(raw_p_values, dtype=float)
    n = len(p_vals)
    if n == 0:
        return []
    sort_idx = np.argsort(p_vals)
    adj_p = np.zeros(n, dtype=float)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(sort_idx):
        multiplier = n - rank_idx
        cur = min(1.0, p_vals[orig_idx] * multiplier)
        running_max = max(running_max, cur)
        adj_p[orig_idx] = running_max
    return adj_p.tolist()


def perform_statistical_analysis(summary_csv_path: Path, output_dir: Path) -> pd.DataFrame:
    """Computes paired statistical tests across all 4 tasks and writes formatted tables."""
    df_summary = pd.read_csv(summary_csv_path)
    tasks = sorted(df_summary["task"].unique())

    stats_records = []
    raw_p_values = []

    for task in tasks:
        sub_base = df_summary[(df_summary["task"] == task) & (df_summary["optimizer"] == "baseline")].sort_values("seed")
        sub_prop = df_summary[(df_summary["task"] == task) & (df_summary["optimizer"] == "da_ehrf_direct_ei")].sort_values("seed")

        assert (sub_base["seed"].values == sub_prop["seed"].values).all(), f"Seeds not aligned on {task}"

        base_vals = sub_base["final_incumbent_regret"].values
        prop_vals = sub_prop["final_incumbent_regret"].values

        diff = prop_vals - base_vals
        wins = int(np.sum(diff < 0))
        ties = int(np.sum(diff == 0))
        losses = int(np.sum(diff > 0))

        # Wilcoxon signed-rank test
        stat_res = wilcoxon(prop_vals, base_vals, alternative="two-sided")
        pval = float(stat_res.pvalue)
        raw_p_values.append(pval)

        # Cliff's delta
        cd = calculate_cliffs_delta(prop_vals, base_vals)
        cd_interp = classify_cliffs_delta(cd)

        mean_b = float(np.mean(base_vals))
        mean_p = float(np.mean(prop_vals))
        std_b = float(np.std(base_vals))
        std_p = float(np.std(prop_vals))
        med_b = float(np.median(base_vals))
        med_p = float(np.median(prop_vals))

        mean_reduction_pct = float(((mean_b - mean_p) / max(mean_b, 1e-9)) * 100.0)
        med_reduction_pct = float(((med_b - med_p) / max(med_b, 1e-9)) * 100.0)

        stats_records.append({
            "task": task,
            "n_seeds": len(base_vals),
            "baseline_mean": mean_b,
            "baseline_std": std_b,
            "baseline_median": med_b,
            "proposed_mean": mean_p,
            "proposed_std": std_p,
            "proposed_median": med_p,
            "mean_reduction_pct": mean_reduction_pct,
            "median_reduction_pct": med_reduction_pct,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "wilcoxon_stat": float(stat_res.statistic),
            "wilcoxon_pval": pval,
            "cliffs_delta": cd,
            "cliffs_delta_class": cd_interp,
        })

    # Apply Holm-Bonferroni correction
    adj_p = apply_holm_bonferroni(raw_p_values)
    for i, rec in enumerate(stats_records):
        rec["wilcoxon_pval_adjusted"] = adj_p[i]
        rec["statistically_significant_05"] = bool(adj_p[i] < 0.05)

    df_stats = pd.DataFrame(stats_records)
    df_stats.to_csv(output_dir / "statistical_analysis.csv", index=False)

    with open(output_dir / "statistical_analysis.json", "w") as f:
        json.dump(stats_records, f, indent=2)

    return df_stats


def plot_anytime_regret_curves(logs_parquet_path: Path, output_dir: Path):
    """Generates 4-panel publication-grade anytime regret curves (mean ± SEM)."""
    df = pd.read_parquet(logs_parquet_path)
    tasks = [
        ("ackley_2d", "Ackley 2D ([-5, 5]²)"),
        ("rosenbrock_2d", "Rosenbrock 2D ([-5, 5]²)"),
        ("sphere_2d", "Sphere 2D ([-5, 5]²)"),
        ("rosenbrock_4d", "Rosenbrock 4D ([-5, 5]⁴)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    axes = axes.flatten()

    colors = {
        "baseline": "#1f77b4",          # Deep Blue
        "da_ehrf_direct_ei": "#d62728", # Crimson Red
    }
    labels = {
        "baseline": "SMAC3 Baseline (Standard EI)",
        "da_ehrf_direct_ei": "DA-EHRF Direct EI (Tree-Path U_E)",
    }

    for idx, (task_key, task_title) in enumerate(tasks):
        ax = axes[idx]
        sub_task = df[df["task"] == task_key]

        for opt_key in ["baseline", "da_ehrf_direct_ei"]:
            sub_opt = sub_task[sub_task["optimizer"] == opt_key]
            
            # Pivot seeds x trials
            pivot = sub_opt.pivot(index="seed", columns="trial", values="incumbent_regret")
            trials = pivot.columns.values
            mean_curve = pivot.mean(axis=0).values
            sem_curve = (pivot.std(axis=0) / np.sqrt(len(pivot))).values

            c = colors[opt_key]
            lbl = labels[opt_key]

            ax.plot(trials, mean_curve, label=lbl, color=c, lw=2.2)
            ax.fill_between(
                trials,
                np.maximum(mean_curve - sem_curve, 0.0),
                mean_curve + sem_curve,
                color=c,
                alpha=0.18,
            )

        # Indicate initial design transition line at trial 10
        ax.axvline(x=10, color="#666666", ls="--", lw=1.2, alpha=0.7, label="BO Start (Trial 10)")

        ax.set_title(task_title, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Trial Number $t$", fontsize=11)
        ax.set_ylabel("Incumbent Regret $R_t$", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        if task_key in ("sphere_2d", "rosenbrock_4d"):
            ax.set_yscale("log")
        ax.legend(fontsize=9, loc="upper right" if task_key != "sphere_2d" else "lower left")

    plt.suptitle(
        "CARPS Benchmark Anytime Incumbent Regret Across 30 Paired Seeds (Mean ± SEM)\nSMAC3 Baseline vs. DA-EHRF Direct Pure Epistemic EI",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    png_path = output_dir / "anytime_regret_curves.png"
    pdf_path = output_dir / "anytime_regret_curves.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved high-resolution anytime curves to {png_path} and {pdf_path}")


def main():
    base_dir = Path(PROJECT_ROOT) / "results" / "carps_da_ehrf_direct_ei"
    summary_path = base_dir / "summary.csv"
    logs_parquet_path = base_dir / "logs.parquet"

    print("Executing Statistical Analysis & Plotting Pipeline...")
    df_stats = perform_statistical_analysis(summary_path, base_dir)
    plot_anytime_regret_curves(logs_parquet_path, base_dir)

    print("\n================================================================")
    print("STATISTICAL VERIFICATION SUMMARY")
    print("================================================================")
    cols = ["task", "baseline_mean", "proposed_mean", "mean_reduction_pct", "baseline_median", "proposed_median", "median_reduction_pct", "wins", "losses", "wilcoxon_pval", "cliffs_delta", "cliffs_delta_class"]
    print(df_stats[cols].to_string(index=False))


if __name__ == "__main__":
    main()
