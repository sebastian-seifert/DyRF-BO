#!/usr/bin/env python3
"""
Publication-grade visualizer for the Global Overall UQ Benchmark results.
Produces three standalone horizontal bar plots (one for each metric: ECE, Winkler Score, NAURC)
comparing all 10 approaches, with decluttered labels ("A (CV)", "BC (OOB)", etc.).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    "hutter_total": "Hutter Total",
    "shaker_epistemic": "Shaker Epistemic",
}

# Family and color assignment (Okabe-Ito colorblind-safe)
FAMILY_COLORS: Dict[str, str] = {
    "hutter_total": "#5A6268",       # Slate Gray
    "shaker_epistemic": "#212529",   # Charcoal
    "A": "#0072B2",                  # Blue
    "AC": "#009E73",                 # Teal / Green
    "B": "#D55E00",                  # Vermilion
    "BC": "#785EF0",                 # Purple
}


def clean_approach_name(approach: str) -> str:
    """Return clean, decluttered display name for an approach."""
    return APPROACH_DISPLAY_NAMES.get(approach, approach)


def get_approach_color_and_hatch(approach: str) -> Tuple[str, Optional[str]]:
    """Determine bar facecolor and hatch based on family and CV/OOB estimator."""
    if approach == "hutter_total":
        return FAMILY_COLORS["hutter_total"], None
    if approach == "shaker_epistemic":
        return FAMILY_COLORS["shaker_epistemic"], None
    
    # Proximity approaches
    if "prox_bc" in approach:
        color = FAMILY_COLORS["BC"]
    elif "prox_ac" in approach:
        color = FAMILY_COLORS["AC"]
    elif "prox_b" in approach:
        color = FAMILY_COLORS["B"]
    elif "prox_a" in approach:
        color = FAMILY_COLORS["A"]
    else:
        color = "#4A90E2"

    hatch = "//" if "oob" in approach else None
    return color, hatch


def set_publication_style():
    """Configure matplotlib for clean publication-quality rendering."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9.0,
        "figure.titlesize": 11,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


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
    Generate and save a horizontal ranked bar plot for a single metric.
    Bars are sorted so that best performance (lowest metric value) is at the top.
    """
    set_publication_style()
    df_sorted = df.sort_values(by=metric_col, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)

    y_pos = np.arange(len(df_sorted))
    means = df_sorted[metric_col].values
    sems = df_sorted[sem_col].values
    approaches = df_sorted["approach"].values

    # Locate baseline value
    hutter_mask = df_sorted["approach"] == "hutter_total"
    hutter_val = float(df_sorted.loc[hutter_mask, metric_col].values[0]) if hutter_mask.any() else None

    # Reference baseline line
    if hutter_val is not None:
        ax.axvline(
            hutter_val,
            color="#e63946",
            linestyle="--",
            linewidth=1.2,
            zorder=2,
            label=f"Baseline: Hutter Total ({hutter_val:.4g})",
        )

    # Plot bars
    bar_height = 0.65
    for i, (approach, mean_val, sem_val) in enumerate(zip(approaches, means, sems)):
        color, hatch = get_approach_color_and_hatch(approach)
        edgecolor = "#212529" if approach in ["hutter_total", "shaker_epistemic"] else color
        
        # OOB uses slightly lighter alpha or hatch to distinguish from CV
        alpha = 0.82 if "oob" in approach else 0.95
        
        ax.barh(
            i,
            mean_val,
            xerr=sem_val,
            height=bar_height,
            color=color,
            alpha=alpha,
            edgecolor=edgecolor,
            hatch=hatch,
            linewidth=0.9,
            capsize=3.0,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#212529"},
            zorder=3,
        )

        # Value label at the tip of the bar
        label_text = f" {mean_val:.4g}"
        ax.text(
            mean_val + sem_val,
            i,
            label_text,
            va="center",
            ha="left",
            fontsize=7.5,
            color="#212529",
            zorder=4,
        )

    # Formatting axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels([clean_approach_name(a) for a in approaches])
    ax.set_xlabel(f"{xlabel}  ($\\leftarrow$ Lower is Better)")
    ax.set_title(title, pad=10, weight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Extend x-lim slightly for annotations
    x_max = (means + sems).max() * 1.15
    ax.set_xlim(0, x_max)

    # Legend for Hutter baseline
    if hutter_val is not None:
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=8)

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
    Load main_global.csv and generate 3 standalone bar plots for ECE, Winkler Score, and NAURC.
    Also produces a combined 1x3 panel overview plot.
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
        title="Expected Calibration Error (ECE) Across Approaches",
        xlabel="Expected Calibration Error (ECE)",
        output_stem=out_path / "global_ece",
        formats=formats,
    )
    results.extend([str(p) for p in ece_files])

    # 2. Winkler Score Plot (winkler_mean_mean or winkler_95_mean)
    winkler_col = "winkler_mean_mean" if "winkler_mean_mean" in df.columns else "winkler_mean"
    winkler_sem = "winkler_mean_sem" if "winkler_mean_sem" in df.columns else "winkler_sem"
    winkler_files = plot_single_metric_bar(
        df=df,
        metric_col=winkler_col,
        sem_col=winkler_sem,
        title="Prediction Interval Quality: Winkler Score",
        xlabel="Winkler Score (Mean)",
        output_stem=out_path / "global_winkler_mean",
        formats=formats,
    )
    results.extend([str(p) for p in winkler_files])

    # 3. NAURC Plot
    naurc_files = plot_single_metric_bar(
        df=df,
        metric_col="naurc_mean",
        sem_col="naurc_sem",
        title="Selective Classification Risk: Normalized AURC",
        xlabel="Normalized Area Under Risk-Coverage (NAURC)",
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
