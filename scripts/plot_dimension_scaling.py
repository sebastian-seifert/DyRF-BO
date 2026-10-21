#!/usr/bin/env python3
"""
Publication-grade Dimension Scaling Visualizer (N = 2,880 per dimension).
Plots the scaling behavior across dimensions (D in {2, 3, 5, 8, 16, 32, 64, 128})
for two focal approaches:
  1. Proximity BC (OOB) [Seaborn 'flare' deep wine color]
  2. Variance Decomposition [Neutral slate gray]

Generates 3 figures (ECE, Winkler Score, E-AURC) with shaded 95% Confidence Interval bands,
log2-spaced dimension scale, de-spined layout, and camera-ready vector formatting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Target approaches to compare
TARGET_APPROACHES: List[str] = ["prox_bc_oob", "hutter_total"]
APPROACHES_WITH_ENTROPY: List[str] = ["prox_bc_oob", "hutter_total", "shaker_epistemic"]

APPROACH_DISPLAY_NAMES: Dict[str, str] = {
    "prox_bc_oob": "Proximity BC (OOB)",
    "hutter_total": "Variance Decomposition",
    "shaker_epistemic": "Shaker Entropy",
}

# Refined warm palette:
# Proximity BC (OOB) -> Warm Terracotta / Rust (#C2410C)
# Variance Decomposition -> Warm Dark Stone (#44403C)
# Shaker Entropy -> Warm Golden Amber (#D97706)
APPROACH_STYLES: Dict[str, Dict[str, str]] = {
    "prox_bc_oob": {
        "color": "#C2410C",            # Warm Terracotta / Rust
        "linestyle": "-",
        "marker": "o",
        "label": "Proximity BC (OOB)",
    },
    "hutter_total": {
        "color": "#44403C",            # Warm Dark Stone
        "linestyle": "--",
        "marker": "s",
        "label": "Variance Decomposition",
    },
    "shaker_epistemic": {
        "color": "#D97706",            # Warm Golden Amber
        "linestyle": ":",
        "marker": "^",
        "label": "Shaker Entropy",
    },
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


def plot_single_dimension_scaling(
    df: pd.DataFrame,
    metric_col: str,
    sem_col: str,
    ylabel: str,
    output_stem: Path,
    approaches: List[str] | None = None,
    formats: List[str] = ["pdf", "png"],
) -> List[Path]:
    """
    Plot dimension scaling curves with 95% Confidence Interval shaded bands
    comparing focal approaches across dimensions.
    """
    set_publication_rcparams()

    if approaches is None:
        approaches = TARGET_APPROACHES

    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)

    # Filter for the target approaches
    df_filtered = df[df["approach"].isin(approaches)].copy()
    dims = sorted(df_filtered["dimension"].unique())

    for approach in approaches:
        sub_df = df_filtered[df_filtered["approach"] == approach].sort_values("dimension")
        if sub_df.empty:
            continue

        x = sub_df["dimension"].values
        y = sub_df[metric_col].values
        sem = sub_df[sem_col].values
        ci = sem * 1.96  # 95% Confidence Interval half-width

        style = APPROACH_STYLES[approach]

        # 95% Confidence Interval shaded band
        ax.fill_between(
            x,
            y - ci,
            y + ci,
            facecolor=style["color"],
            edgecolor=style["color"],
            alpha=0.30,
            linewidth=0.6,
            zorder=2,
        )

        # Line plot with markers
        ax.plot(
            x,
            y,
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            linewidth=1.2,
            markersize=5.0,
            markeredgecolor="#1F2937",
            markeredgewidth=0.5,
            zorder=3,
        )

    # Use log2 scale for dimensions so {2, 3, 5, 8, 16, 32, 64, 128} are evenly spaced
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # De-spine (remove top and right spines, lighten bottom and left)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.spines["bottom"].set_linewidth(0.7)

    # Tick styling
    ax.tick_params(axis="both", which="both", length=3.5, width=0.7, colors="#6B7280", labelsize=8.5)

    # Gridlines
    ax.grid(True, linestyle="--", linewidth=0.55, color="#E5E7EB", alpha=0.85, zorder=0)
    ax.set_axisbelow(True)

    # Labels
    ax.set_xlabel("Dimension ($D$)", fontsize=9.5, color="#111827", labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9.5, color="#111827", labelpad=6)

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


def generate_dimension_scaling_plots(
    input_csv: str,
    output_dir: str,
    surrogate: str | None = None,
    approaches: List[str] | None = None,
    formats: List[str] = ["pdf", "png"],
) -> List[str]:
    """
    Load main_global_by_dimension.csv (or main_global_rf_by_dimension.csv)
    and generate 3 dimension scaling line plots for ECE, Winkler Score,
    and E-AURC comparing focal approaches. Optionally filters by surrogate.
    """
    df = pd.read_csv(input_csv)
    if surrogate is not None:
        if "surrogate" not in df.columns:
            raise ValueError(f"Cannot filter by surrogate '{surrogate}': column 'surrogate' not found in {input_csv}")
        df = df[df["surrogate"] == surrogate].copy()
        if df.empty:
            raise ValueError(f"No records found for surrogate '{surrogate}' in {input_csv}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. ECE Plot
    ece_files = plot_single_dimension_scaling(
        df=df,
        metric_col="ece_mean",
        sem_col="ece_sem",
        ylabel="Expected Calibration Error (ECE)",
        output_stem=out_path / "dim_scaling_ece",
        approaches=approaches,
        formats=formats,
    )
    results.extend([str(p) for p in ece_files])

    # 2. Winkler Score Plot
    winkler_col = "winkler_mean_mean" if "winkler_mean_mean" in df.columns else "winkler_mean"
    winkler_sem = "winkler_mean_sem" if "winkler_mean_sem" in df.columns else "winkler_sem"
    winkler_files = plot_single_dimension_scaling(
        df=df,
        metric_col=winkler_col,
        sem_col=winkler_sem,
        ylabel="Mean Winkler Score",
        output_stem=out_path / "dim_scaling_winkler_mean",
        approaches=approaches,
        formats=formats,
    )
    results.extend([str(p) for p in winkler_files])

    # 3. E-AURC Plot (Excess Area Under Risk-Coverage)
    e_aurc_files = plot_single_dimension_scaling(
        df=df,
        metric_col="e_aurc_mean",
        sem_col="e_aurc_sem",
        ylabel="Excess AURC (Selective Risk)",
        output_stem=out_path / "dim_scaling_e_aurc",
        approaches=approaches,
        formats=formats,
    )
    results.extend([str(p) for p in e_aurc_files])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Plot dimension scaling curves for focal UQ approaches."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="results/extrapolation_uq/analysis/main_global_by_dimension.csv",
        help="Path to main_global_by_dimension.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/extrapolation_uq/analysis/plots/dimension_scaling",
        help="Directory where plot files will be saved.",
    )
    parser.add_argument(
        "--surrogate",
        type=str,
        default=None,
        help="Optional surrogate to filter by (e.g. smac_default).",
    )
    parser.add_argument(
        "--include-entropy",
        action="store_true",
        help="Include Shaker Entropy in the comparison.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats (e.g., pdf png).",
    )
    args = parser.parse_args()

    approaches = APPROACHES_WITH_ENTROPY if args.include_entropy else TARGET_APPROACHES

    files = generate_dimension_scaling_plots(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        surrogate=args.surrogate,
        approaches=approaches,
        formats=args.formats,
    )
    print(f"Generated {len(files)} plot files in {args.output_dir}:")
    for f in files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()

