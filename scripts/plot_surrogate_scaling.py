#!/usr/bin/env python3
"""
Publication-grade Random Forest Surrogate Visualizer (N = 5,760 per surrogate).
Plots the performance comparison across four distinct Random Forest architectures:
  1. SMAC Default (B=10, deep)
  2. Mature (B=100, deep)
  3. Shallow (B=25, depth=4)
  4. Coarse (B=25, min_leaf=10)

For two focal approaches:
  - Proximity BC (OOB) [Warm Terracotta / Rust, #C2410C, circle 'o']
  - Variance Decomposition [Warm Dark Stone, #44403C, square 's']

Generates 3 horizontal paired dot-and-whisker (dumbbell/forest) plots (ECE, Winkler Score, E-AURC)
with 95% Confidence Interval error whiskers, clean connector bars, and camera-ready vector formatting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Target approaches to compare
TARGET_APPROACHES: List[str] = ["prox_bc_oob", "hutter_total"]

APPROACH_DISPLAY_NAMES: Dict[str, str] = {
    "prox_bc_oob": "Proximity BC (OOB)",
    "hutter_total": "Variance Decomposition",
}

APPROACH_STYLES: Dict[str, Dict[str, str]] = {
    "prox_bc_oob": {
        "color": "#C2410C",            # Warm Terracotta / Rust
        "marker": "o",
        "label": "Proximity BC (OOB)",
    },
    "hutter_total": {
        "color": "#44403C",            # Warm Dark Stone
        "marker": "s",
        "label": "Variance Decomposition",
    },
}

# Canonical order and human-readable names for Random Forest surrogates
SURROGATE_ORDER: List[str] = ["smac_default", "mature", "shallow", "coarse"]

SURROGATE_DISPLAY_NAMES: Dict[str, str] = {
    "smac_default": "SMAC Default ($B=10$)",
    "mature": "Mature ($B=100$)",
    "shallow": "Shallow (Depth $= 4$)",
    "coarse": "Coarse (Leaf $= 10$)",
}


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
        "lines.linewidth": 1.3,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def plot_single_surrogate_comparison(
    df: pd.DataFrame,
    metric_col: str,
    sem_col: str,
    xlabel: str,
    output_stem: Path,
    formats: List[str] = ["pdf", "png"],
) -> List[Path]:
    """
    Plot horizontal paired dot-and-whisker (dumbbell) chart comparing
    Proximity BC (OOB) vs Variance Decomposition across the 4 Random Forest surrogates.
    """
    set_publication_rcparams()

    fig, ax = plt.subplots(figsize=(5.8, 3.4), dpi=300)

    # Filter for the two target approaches
    df_filtered = df[df["approach"].isin(TARGET_APPROACHES)].copy()
    
    # Filter available surrogates in canonical order
    surrogates = [s for s in SURROGATE_ORDER if s in df_filtered["surrogate"].unique()]
    if not surrogates:
        surrogates = sorted(df_filtered["surrogate"].unique().tolist())

    y_positions = np.arange(len(surrogates))

    # Plot dumbbell connector lines first (underneath markers)
    for i, s in enumerate(surrogates):
        s_df = df_filtered[df_filtered["surrogate"] == s]
        sub_prox = s_df[s_df["approach"] == "prox_bc_oob"]
        sub_hutter = s_df[s_df["approach"] == "hutter_total"]

        if not sub_prox.empty and not sub_hutter.empty:
            x_prox = sub_prox[metric_col].values[0]
            x_hutter = sub_hutter[metric_col].values[0]
            # Subtle connector bar showing the gap
            ax.plot(
                [x_prox, x_hutter],
                [i, i],
                color="#E5E7EB",
                linewidth=3.2,
                solid_capstyle="round",
                zorder=1,
            )

    # Plot points and error bars for each approach
    for approach in TARGET_APPROACHES:
        style = APPROACH_STYLES[approach]
        x_vals = []
        ci_vals = []
        valid_y = []

        for i, s in enumerate(surrogates):
            sub = df_filtered[(df_filtered["surrogate"] == s) & (df_filtered["approach"] == approach)]
            if not sub.empty:
                x_val = sub[metric_col].values[0]
                sem_val = sub[sem_col].values[0]
                ci_val = sem_val * 1.96
                x_vals.append(x_val)
                ci_vals.append(ci_val)
                valid_y.append(i)

        if x_vals:
            ax.errorbar(
                x_vals,
                valid_y,
                xerr=ci_vals,
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                label=style["label"],
                markersize=6.0,
                markeredgecolor="#1F2937",
                markeredgewidth=0.7,
                capsize=3.5,
                capthick=0.9,
                elinewidth=1.1,
                linestyle="none",
                zorder=3,
            )

    # Y-axis ticks and labels (inverted so smac_default is at the top)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([SURROGATE_DISPLAY_NAMES.get(s, s) for s in surrogates])
    ax.set_ylim(len(surrogates) - 0.5, -0.5)

    # De-spine (remove top and right spines, lighten bottom and left)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.spines["bottom"].set_linewidth(0.7)

    # Tick styling
    ax.tick_params(axis="both", which="both", length=3.5, width=0.7, colors="#6B7280", labelsize=8.5)

    # Subtle horizontal & vertical gridlines
    ax.grid(True, axis="x", linestyle="--", linewidth=0.55, color="#E5E7EB", alpha=0.85, zorder=0)
    ax.set_axisbelow(True)

    # Labels
    ax.set_xlabel(xlabel, fontsize=9.5, color="#111827", labelpad=6)

    # Clean legend
    ax.legend(
        loc="best",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#E5E7EB",
        framealpha=0.95,
        fontsize=8.5,
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


def generate_surrogate_scaling_plots(
    input_csv: str,
    output_dir: str,
    formats: List[str] = ["pdf", "png"],
) -> List[str]:
    """
    Load main_global_by_surrogate.csv and generate 3 paired dot-and-whisker plots
    for ECE, Winkler Score, and E-AURC comparing Proximity BC (OOB) vs Variance Decomposition.
    """
    df = pd.read_csv(input_csv)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. ECE Plot
    ece_files = plot_single_surrogate_comparison(
        df=df,
        metric_col="ece_mean",
        sem_col="ece_sem",
        xlabel="Expected Calibration Error (ECE)",
        output_stem=out_path / "surrogate_ece",
        formats=formats,
    )
    results.extend([str(p) for p in ece_files])

    # 2. Winkler Score Plot
    winkler_col = "winkler_mean_mean" if "winkler_mean_mean" in df.columns else "winkler_mean"
    winkler_sem = "winkler_mean_sem" if "winkler_mean_sem" in df.columns else "winkler_sem"
    winkler_files = plot_single_surrogate_comparison(
        df=df,
        metric_col=winkler_col,
        sem_col=winkler_sem,
        xlabel="Mean Winkler Score",
        output_stem=out_path / "surrogate_winkler_mean",
        formats=formats,
    )
    results.extend([str(p) for p in winkler_files])

    # 3. E-AURC Plot (Excess Area Under Risk-Coverage)
    e_aurc_files = plot_single_surrogate_comparison(
        df=df,
        metric_col="e_aurc_mean",
        sem_col="e_aurc_sem",
        xlabel="Excess AURC (Selective Risk)",
        output_stem=out_path / "surrogate_e_aurc",
        formats=formats,
    )
    results.extend([str(p) for p in e_aurc_files])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Plot Random Forest surrogate comparisons for Proximity BC (OOB) vs Variance Decomposition."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="results/extrapolation_uq/analysis/main_global_by_surrogate.csv",
        help="Path to main_global_by_surrogate.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/extrapolation_uq/analysis/plots/surrogate_scaling",
        help="Directory where plot files will be saved.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats (e.g., pdf png).",
    )
    args = parser.parse_args()

    files = generate_surrogate_scaling_plots(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        formats=args.formats,
    )
    print(f"Generated {len(files)} plot files in {args.output_dir}:")
    for f in files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
