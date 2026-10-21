#!/usr/bin/env python3
"""
Visualizer for the Global Overall UQ Benchmark results.
Produces three horizontal bar plots (one for each metric: ECE, Winkler Score, NAURC)
comparing all 10 approaches using standard Matplotlib visuals,
with 'Variance Decomposition' baseline naming, and legend placed top-right.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Approach display names mapping (decluttered per user specification)
APPROACH_DISPLAY_NAMES: Dict[str, str] = {
    "prox_a_cv": "A (CV)",
    "prox_a_oob": "A (OOB)",
    "prox_ac_cv": "AC (CV)",
    "prox_ac_oob": "AC (OOB)",
    "prox_b_cv": "B (CV)",
    "prox_b_oob": "B (OOB)",
    "prox_bc_cv": "BC (CV)",
    "prox_bc_oob": "BC (OOB)",
    "hutter_total": "Variance Decomposition",
    "shaker_epistemic": "Shaker Epistemic",
}

# Standard clean color mapping (Tab10 palette)
APPROACH_COLORS: Dict[str, str] = {
    "hutter_total": "#7f7f7f",       # Standard Gray
    "shaker_epistemic": "#333333",   # Dark Gray
    "prox_a_cv": "#1f77b4",          # Tab Blue
    "prox_a_oob": "#aec7e8",         # Light Blue
    "prox_ac_cv": "#2ca02c",         # Tab Green
    "prox_ac_oob": "#98df8a",        # Light Green
    "prox_b_cv": "#ff7f0e",          # Tab Orange
    "prox_b_oob": "#ffbb78",         # Light Orange
    "prox_bc_cv": "#9467bd",         # Tab Purple
    "prox_bc_oob": "#c5b0d5",        # Light Purple
}


def clean_approach_name(approach: str) -> str:
    """Return clean, decluttered display name for an approach."""
    return APPROACH_DISPLAY_NAMES.get(approach, approach)


def plot_single_metric_bar(
    df: pd.DataFrame,
    metric_col: str,
    sem_col: str,
    title: str,
    xlabel: str,
    output_stem: Path,
    formats: List[str] = ["pdf", "png"],
) -> List[Path]:
    """
    Generate and save a horizontal bar plot for a single metric across all 10 approaches.
    Sorted so that best performance (lowest metric value) is at the top.
    """
    # Use standard matplotlib default style
    plt.rcdefaults()

    # Sort ascending=False so that smallest values appear at the top in barh
    df_sorted = df.sort_values(by=metric_col, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=300)

    y_pos = np.arange(len(df_sorted))
    means = df_sorted[metric_col].values
    sems = df_sorted[sem_col].values
    approaches = df_sorted["approach"].values
    labels = [clean_approach_name(a) for a in approaches]
    colors = [APPROACH_COLORS.get(a, "#1f77b4") for a in approaches]

    # Baseline reference value (Variance Decomposition)
    hutter_mask = df_sorted["approach"] == "hutter_total"
    hutter_val = float(df_sorted.loc[hutter_mask, metric_col].values[0]) if hutter_mask.any() else None

    # Reference baseline line
    if hutter_val is not None:
        ax.axvline(
            hutter_val,
            color="red",
            linestyle="--",
            linewidth=1.2,
            zorder=3,
            label=f"Baseline: Variance Decomposition ({hutter_val:.4g})",
        )

    # Plot horizontal bars
    ax.barh(
        y_pos,
        means,
        xerr=sems,
        height=0.65,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
        capsize=3.0,
        zorder=2,
    )

    # Annotate numeric values at the tip of each bar
    for i, (mean_val, sem_val) in enumerate(zip(means, sems)):
        ax.text(
            mean_val + sem_val,
            i,
            f" {mean_val:.4g}",
            va="center",
            ha="left",
            fontsize=8,
        )

    # Formatting axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.6, zorder=1)
    ax.set_axisbelow(True)

    # Extend x-lim slightly so text labels and top-right legend fit cleanly
    x_max = float((means + sems).max()) * 1.25
    ax.set_xlim(0, x_max)

    # Legend placed top-right
    if hutter_val is not None:
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    plt.tight_layout()

    saved_files = []
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target_path = output_stem.with_suffix(f".{fmt}")
        fig.savefig(target_path, bbox_inches="tight")
        saved_files.append(target_path)

    plt.close(fig)
    return saved_files


def generate_metric_bar_plots(
    input_csv: str,
    output_dir: str,
    formats: List[str] = ["pdf", "png"],
) -> List[str]:
    """
    Load main_global.csv and generate 3 standalone horizontal bar plots for ECE, Winkler Score, and NAURC.
    """
    df = pd.read_csv(input_csv)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. ECE Plot
    ece_files = plot_single_metric_bar(
        df=df,
        metric_col="ece_mean",
        sem_col="ece_sem",
        title="Expected Calibration Error (ECE)",
        xlabel="Expected Calibration Error (ECE)",
        output_stem=out_path / "global_ece",
        formats=formats,
    )
    results.extend([str(p) for p in ece_files])

    # 2. Winkler Score Plot
    winkler_col = "winkler_mean_mean" if "winkler_mean_mean" in df.columns else "winkler_mean"
    winkler_sem = "winkler_mean_sem" if "winkler_mean_sem" in df.columns else "winkler_sem"
    winkler_files = plot_single_metric_bar(
        df=df,
        metric_col=winkler_col,
        sem_col=winkler_sem,
        title="Winkler Score",
        xlabel="Mean Winkler Score",
        output_stem=out_path / "global_winkler_mean",
        formats=formats,
    )
    results.extend([str(p) for p in winkler_files])

    # 3. NAURC Plot
    naurc_files = plot_single_metric_bar(
        df=df,
        metric_col="naurc_mean",
        sem_col="naurc_sem",
        title="Normalized AURC (Selective Risk)",
        xlabel="Normalized AURC (NAURC)",
        output_stem=out_path / "global_naurc",
        formats=formats,
    )
    results.extend([str(p) for p in naurc_files])

    return results


def main():
    parser = argparse.ArgumentParser(description="Plot global overall metrics as horizontal bar plots.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="results/extrapolation_uq/analysis/main_global.csv",
        help="Path to main_global.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/extrapolation_uq/analysis/plots",
        help="Directory where plot files will be saved.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats (e.g., pdf png).",
    )
    args = parser.parse_args()

    files = generate_metric_bar_plots(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        formats=args.formats,
    )
    print(f"Generated {len(files)} plot files in {args.output_dir}:")
    for f in files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
