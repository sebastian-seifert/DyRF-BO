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
TARGET_APPROACHES: List[str] = ["prox_bc_oob", "hutter_between"]
APPROACHES_WITH_ENTROPY: List[str] = ["prox_bc_oob", "hutter_between", "shaker_epistemic"]

APPROACH_DISPLAY_NAMES: Dict[str, str] = {
    "prox_bc_oob": "Proximity BC (OOB)",
    "hutter_between": "Between-Tree Disagreement",
    "hutter_total": "Variance Decomposition",
    "shaker_epistemic": "Shaker Entropy",
}

APPROACH_STYLES: Dict[str, Dict[str, any]] = {
    "prox_bc_oob": {
        "color": "#C2410C",            # Warm Terracotta / Rust
        "marker": "o",
        "label": "Proximity BC (OOB)",
    },
    "hutter_between": {
        "color": "#44403C",            # Warm Dark Stone
        "marker": "s",
        "label": "Between-Tree Disagreement",
    },
    "hutter_total": {
        "color": "#44403C",            # Warm Dark Stone
        "marker": "s",
        "label": "Variance Decomposition",
    },
    "shaker_epistemic": {
        "color": "#D97706",            # Warm Golden Amber
        "marker": "^",
        "label": "Shaker Entropy",
    },
}


# Canonical order and clean names for Random Forest surrogates (no parentheses)
SURROGATE_ORDER: List[str] = ["smac_default", "mature", "shallow", "coarse"]

SURROGATE_DISPLAY_NAMES: Dict[str, str] = {
    "smac_default": "SMAC Default",
    "mature": "Mature",
    "shallow": "Shallow",
    "coarse": "Coarse",
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
    approaches: List[str] | None = None,
    formats: List[str] = ["pdf", "png"],
    legend_loc: str = "best",
    legend_fontsize: float = 8.5,
    legend_kwargs: dict | None = None,
    ylim: tuple[float, float] | None = None,
) -> List[Path]:
    """
    Plot horizontal paired dot-and-whisker chart comparing
    focal approaches across the 4 Random Forest surrogates
    with vertical dodging to eliminate overlap.
    """
    set_publication_rcparams()

    if approaches is None:
        approaches = TARGET_APPROACHES

    fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=300)

    # Filter for the target approaches
    df_filtered = df[df["approach"].isin(approaches)].copy()
    
    # Filter available surrogates in canonical order
    surrogates = [s for s in SURROGATE_ORDER if s in df_filtered["surrogate"].unique()]
    if not surrogates:
        surrogates = sorted(df_filtered["surrogate"].unique().tolist())

    y_positions = np.arange(len(surrogates))

    # Very subtle alternating row shading for clean distinction without visual clutter
    for i in range(len(surrogates)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, facecolor="#F9FAFB", edgecolor="none", zorder=0)

    # Subtle horizontal divider lines between surrogates
    for y_div in np.arange(0.5, len(surrogates) - 0.5, 1.0):
        ax.axhline(y_div, color="#E5E7EB", linewidth=0.6, alpha=0.6, zorder=0)

    # Compute vertical dodging offsets based on number of approaches (tightened for grouped cohesion)
    if len(approaches) == 1:
        offsets = [0.0]
    elif len(approaches) == 2:
        offsets = [-0.11, 0.11]
    elif len(approaches) == 3:
        offsets = [-0.14, 0.0, 0.14]
    else:
        offsets = np.linspace(-0.16, 0.16, len(approaches)).tolist()
    offset_map = {app: offsets[idx] for idx, app in enumerate(approaches)}


    # Plot points and error bars for each approach with distinct vertical height (y_offset)
    for approach in approaches:
        style = APPROACH_STYLES[approach]
        y_offset = offset_map.get(approach, 0.0)
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
                valid_y.append(i + y_offset)

        if x_vals:
            ax.errorbar(
                x_vals,
                valid_y,
                xerr=ci_vals,
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                label=style["label"],
                markersize=5.5,
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
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
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
    default_legend_kwargs = {
        "loc": legend_loc,
        "frameon": True,
        "facecolor": "#FFFFFF",
        "edgecolor": "#E5E7EB",
        "framealpha": 0.95,
        "fontsize": legend_fontsize,
        "borderpad": 0.35,
    }
    if legend_kwargs:
        default_legend_kwargs.update(legend_kwargs)
    ax.legend(**default_legend_kwargs)

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
    approaches: List[str] | None = None,
    formats: List[str] = ["pdf", "png"],
) -> List[str]:
    """
    Load main_global_by_surrogate.csv and generate 3 paired dot-and-whisker plots
    for ECE, Winkler Score, and E-AURC comparing focal approaches across Random Forest surrogates.
    """
    df = pd.read_csv(input_csv)
    if approaches is None:
        approaches = TARGET_APPROACHES

    if "hutter_between" in approaches and "hutter_between" not in df["approach"].values and "hutter_total" in df["approach"].values:
        df = df.copy()
        df.loc[df["approach"] == "hutter_total", "approach"] = "hutter_between"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []

    compact_legend_kwargs = {
        "borderpad": 0.25,
        "handletextpad": 0.35,
        "labelspacing": 0.2,
        "borderaxespad": 0.2,
    }

    # 1. ECE Plot (Lower right is completely empty across surrogates)
    ece_files = plot_single_surrogate_comparison(
        df=df,
        metric_col="ece_mean",
        sem_col="ece_sem",
        xlabel="Expected Calibration Error (ECE)",
        output_stem=out_path / "surrogate_ece",
        approaches=approaches,
        formats=formats,
        legend_loc="lower right",
        legend_fontsize=7.5,
        legend_kwargs=compact_legend_kwargs,
    )
    results.extend([str(p) for p in ece_files])

    # 2. Winkler Score Plot (Lower right is completely empty across surrogates)
    winkler_col = "winkler_mean_mean" if "winkler_mean_mean" in df.columns else "winkler_mean"
    winkler_sem = "winkler_mean_sem" if "winkler_mean_sem" in df.columns else "winkler_sem"
    winkler_files = plot_single_surrogate_comparison(
        df=df,
        metric_col=winkler_col,
        sem_col=winkler_sem,
        xlabel="Mean Winkler Score",
        output_stem=out_path / "surrogate_winkler_mean",
        approaches=approaches,
        formats=formats,
        legend_loc="lower right",
        legend_fontsize=7.5,
        legend_kwargs=compact_legend_kwargs,
    )
    results.extend([str(p) for p in winkler_files])

    # 3. E-AURC Plot (Excess Area Under Risk-Coverage)
    # Compact legend in top left corner with headroom so it never touches data points
    e_aurc_files = plot_single_surrogate_comparison(
        df=df,
        metric_col="e_aurc_mean",
        sem_col="e_aurc_sem",
        xlabel="Excess AURC (Selective Risk)",
        output_stem=out_path / "surrogate_e_aurc",
        approaches=approaches,
        formats=formats,
        legend_loc="upper left",
        legend_fontsize=7.5,
        legend_kwargs=compact_legend_kwargs,
        ylim=(3.5, -0.75),
    )
    results.extend([str(p) for p in e_aurc_files])



    return results


def main():
    parser = argparse.ArgumentParser(
        description="Plot Random Forest surrogate comparisons for focal UQ approaches."
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

    files = generate_surrogate_scaling_plots(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        approaches=approaches,
        formats=args.formats,
    )
    print(f"Generated {len(files)} plot files in {args.output_dir}:")
    for f in files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()

