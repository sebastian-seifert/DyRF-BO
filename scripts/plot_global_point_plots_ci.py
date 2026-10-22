#!/usr/bin/env python3
"""
Publication-grade Point Plot Visualizer with 95% Confidence Intervals (N = 23,040).
Generates three horizontal point plots (Cleveland Dot / Forest Plots with 95% CI)
for Expected Calibration Error (ECE), Winkler Score, and E-AURC.

Features:
- Point estimates (mean) with 95% Confidence Interval whiskers (± 1.96 * SEM).
- Horizontal row guide lines for clean alignment.
- Colorblind-safe Seaborn 'flare' warm palette with neutral slate baselines.
- Vertical dashed baseline reference line labeled 'Baseline' (top-right legend).
- Clean de-spined academic layout without redundant titles or in-plot text.
- Exports both vector PDF and 300 DPI PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Approach display names mapping
APPROACH_DISPLAY_NAMES: Dict[str, str] = {
    "prox_a_cv": "Proximity A (CV)",
    "prox_a_oob": "Proximity A (OOB)",
    "prox_ac_cv": "Proximity AC (CV)",
    "prox_ac_oob": "Proximity AC (OOB)",
    "prox_b_cv": "Proximity B (CV)",
    "prox_b_oob": "Proximity B (OOB)",
    "prox_bc_cv": "Proximity BC (CV)",
    "prox_bc_oob": "Proximity BC (OOB)",
    "hutter_total": "Variance Decomposition",
    "hutter_between": "Between-Tree Disagreement",
    "shaker_epistemic": "Shaker Epistemic",
}


def clean_approach_name(approach: str) -> str:
    """Return clean, decluttered display name for an approach."""
    return APPROACH_DISPLAY_NAMES.get(approach, approach)


def get_warm_colorblind_palette(approaches: List[str]) -> List[str]:
    """
    Generate an established, colorblind-safe warm palette using Seaborn's
    perceptually uniform 'flare' template.
    Baselines are assigned neutral slate tones to distinguish them clearly.
    """
    warm_colors = sns.color_palette("flare", n_colors=8).as_hex()
    
    proximity_color_map = {
        "prox_bc_oob": warm_colors[7], # Deepest rich wine
        "prox_bc_cv": warm_colors[6],
        "prox_b_oob": warm_colors[5],  # Terracotta / rust
        "prox_b_cv": warm_colors[4],
        "prox_ac_oob": warm_colors[3], # Coral / bronze
        "prox_ac_cv": warm_colors[2],
        "prox_a_oob": warm_colors[1],  # Warm apricot
        "prox_a_cv": warm_colors[0],
    }
    
    palette = []
    for a in approaches:
        if a in ("hutter_total", "hutter_between"):
            palette.append("#6B7280")       # Dignified Slate Gray
        elif a == "shaker_epistemic":
            palette.append("#374151")   # Dark Charcoal Gray
        else:
            palette.append(proximity_color_map.get(a, warm_colors[4]))
    return palette


def set_publication_rcparams() -> None:
    """Configure matplotlib rcParams for camera-ready academic figures."""
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans",
            "Helvetica Neue",
            "Helvetica",
            "Arial",
            "Liberation Sans",
        ],
        "mathtext.fontset": "dejavusans",
        "font.size": 9.0,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def plot_single_metric_point_ci(
    df: pd.DataFrame,
    metric_col: str,
    sem_col: str,
    xlabel: str,
    output_stem: Path,
    formats: List[str] = ["pdf", "png"],
) -> List[Path]:
    """
    Generate and save a horizontal point plot with 95% Confidence Intervals
    (± 1.96 * SEM) across all 10 approaches.
    Sorted with best performance (lowest metric value) on top.
    """
    set_publication_rcparams()

    # Sort ascending=False so that smallest values appear at the top in horizontal layout
    df_sorted = df.sort_values(by=metric_col, ascending=False).reset_index(drop=True)

    fig_height = max(2.2, 0.38 * len(df_sorted))
    fig, ax = plt.subplots(figsize=(6.8, fig_height), dpi=300)

    y_pos = np.arange(len(df_sorted))
    means = df_sorted[metric_col].values
    sems = df_sorted[sem_col].values
    ci_err = sems  # Exact Standard Error of the Mean (SEM) error bar
    approaches = df_sorted["approach"].values
    labels = [clean_approach_name(a) for a in approaches]
    colors = get_warm_colorblind_palette(list(approaches))

    # Baseline reference value (Between-Tree Disagreement or Variance Decomposition)
    hutter_mask = df_sorted["approach"] == "hutter_between"
    if not hutter_mask.any():
        hutter_mask = df_sorted["approach"] == "hutter_total"
    hutter_val = float(df_sorted.loc[hutter_mask, metric_col].values[0]) if hutter_mask.any() else None

    # Reference baseline line: subtle crimson dashed line
    if hutter_val is not None:
        ax.axvline(
            hutter_val,
            color="#B91C1C",
            linestyle=(0, (4, 3)),
            linewidth=1.1,
            alpha=0.9,
            zorder=2,
            label="Baseline",
        )

    # Horizontal guide lines for clean alignment across rows
    for y in y_pos:
        ax.axhline(y, color="#F3F4F6", linestyle="-", linewidth=0.9, zorder=1)

    # Plot 95% Confidence Interval error bars and points per row
    for y, mean_val, err_val, color in zip(y_pos, means, ci_err, colors):
        ax.errorbar(
            mean_val,
            y,
            xerr=err_val,
            fmt="o",
            color=color,
            ecolor=color,
            markersize=6.5,
            markeredgecolor="#1F2937",
            markeredgewidth=0.7,
            elinewidth=1.2,
            capsize=3.0,
            capthick=1.0,
            zorder=3,
        )

    # Formatting axes and labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.0, color="#111827")
    ax.set_xlabel(xlabel, fontsize=9.5, color="#111827", labelpad=6)

    # De-spine (remove top and right spines, lighten bottom and left)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.spines["bottom"].set_linewidth(0.7)

    # Tick styling: no Y ticks, subtle X ticks
    ax.tick_params(axis="y", which="both", length=0)
    ax.tick_params(axis="x", which="both", length=3.5, width=0.7, colors="#6B7280", labelsize=8.5)

    # Subtle vertical gridlines
    ax.grid(axis="x", linestyle="--", linewidth=0.55, color="#E5E7EB", alpha=0.85, zorder=0)
    ax.set_axisbelow(True)

    # Rescale X-axis dynamically to zoom in on the range from the top to worst performing approaches
    val_min = float((means - ci_err).min())
    val_max = float((means + ci_err).max())
    span = val_max - val_min if val_max > val_min else 1.0
    eps_buffer = span * 0.08
    ax.set_xlim(val_min - eps_buffer, val_max + eps_buffer * 1.4)

    # Minimalist, clean legend placed top-right
    if hutter_val is not None:
        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor="#E5E7EB",
            framealpha=0.95,
            fontsize=8.5,
            handlelength=1.5,
            borderpad=0.5,
        )

    plt.tight_layout()

    saved_files = []
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target_path = output_stem.with_suffix(f".{fmt}")
        fig.savefig(target_path, bbox_inches="tight")
        saved_files.append(target_path)

    plt.close(fig)
    return saved_files


def generate_point_plots_ci(
    input_csv: str,
    output_dir: str,
    approaches: List[str] | None = None,
    stem_prefix: str | None = None,
    formats: List[str] = ["pdf", "png"],
) -> List[str]:
    """
    Load main_global.csv and generate 3 standalone horizontal point plots with SEM
    for ECE, Winkler Score, and E-AURC.
    """
    df = pd.read_csv(input_csv)
    if "hutter_between" not in df["approach"].values and "hutter_total" in df["approach"].values:
        df = df.copy()
        df.loc[df["approach"] == "hutter_total", "approach"] = "hutter_between"

    if approaches is not None:
        df = df[df["approach"].isin(approaches)].copy()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prefix = stem_prefix
    if prefix is None:
        prefix = "global_" if out_path.name == "plots" else "point_ci_"

    results = []

    # 1. ECE Plot
    ece_files = plot_single_metric_point_ci(
        df=df,
        metric_col="ece_mean",
        sem_col="ece_sem",
        xlabel="Expected Calibration Error (ECE)",
        output_stem=out_path / f"{prefix}ece",
        formats=formats,
    )
    results.extend([str(p) for p in ece_files])

    # 2. Winkler Score Plot
    winkler_col = "winkler_mean_mean" if "winkler_mean_mean" in df.columns else "winkler_mean"
    winkler_sem = "winkler_mean_sem" if "winkler_mean_sem" in df.columns else "winkler_sem"
    winkler_files = plot_single_metric_point_ci(
        df=df,
        metric_col=winkler_col,
        sem_col=winkler_sem,
        xlabel="Mean Winkler Score",
        output_stem=out_path / f"{prefix}winkler_mean",
        formats=formats,
    )
    results.extend([str(p) for p in winkler_files])

    # 3. E-AURC Plot (Excess Area Under Risk-Coverage)
    e_aurc_files = plot_single_metric_point_ci(
        df=df,
        metric_col="e_aurc_mean",
        sem_col="e_aurc_sem",
        xlabel="Excess AURC (Selective Risk)",
        output_stem=out_path / f"{prefix}e_aurc",
        formats=formats,
    )
    results.extend([str(p) for p in e_aurc_files])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Plot global overall metrics as horizontal point plots with 95% Confidence Intervals."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="results/extrapolation_uq/analysis/main_global.csv",
        help="Path to main_global.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/extrapolation_uq/analysis/plots/point_plots",
        help="Directory where plot files will be saved.",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=None,
        help="Subset of approaches to plot (e.g., prox_bc_oob hutter_between shaker_epistemic).",
    )
    parser.add_argument(
        "--focal-trio",
        action="store_true",
        help="Plot only the focal comparison trio (prox_bc_oob, hutter_between, shaker_epistemic).",
    )
    parser.add_argument(
        "--stem-prefix",
        type=str,
        default=None,
        help="Prefix for output plot file stems (e.g. 'global_' or 'point_ci_').",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats (e.g., pdf png).",
    )
    args = parser.parse_args()

    approaches = args.approaches
    if args.focal_trio:
        approaches = ["prox_bc_oob", "hutter_between", "shaker_epistemic"]

    files = generate_point_plots_ci(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        approaches=approaches,
        stem_prefix=args.stem_prefix,
        formats=args.formats,
    )
    print(f"Generated {len(files)} plot files in {args.output_dir}:")
    for f in files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
