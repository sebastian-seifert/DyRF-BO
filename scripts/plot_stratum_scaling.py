#!/usr/bin/env python3
"""
Publication-grade Stratum Scaling Visualizer (N = 23,040 per stratum).
Plots the scaling behavior across extrapolation strata (0: Interpolation, 1: Near, 2: Moderate, 3: Far)
for two focal approaches:
  1. Proximity BC (OOB) [Warm Terracotta / Rust, #C2410C]
  2. Variance Decomposition [Warm Dark Stone, #44403C]

Generates 3 figures (ECE, Winkler Score, E-AURC) with shaded 95% Confidence Interval bands,
de-spined layout, and camera-ready vector formatting.
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


# Human-readable labels for the 4 extrapolation strata
STRATUM_LABELS: Dict[int, str] = {
    0: "0\n(Interpolation)",
    1: "1\n(Near)",
    2: "2\n(Moderate)",
    3: "3\n(Far)",
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


def plot_single_stratum_scaling(
    df: pd.DataFrame,
    metric_col: str,
    sem_col: str,
    ylabel: str,
    output_stem: Path,
    approaches: List[str] | None = None,
    formats: List[str] = ["pdf", "png"],
) -> List[Path]:
    """
    Plot stratum scaling curves with 95% Confidence Interval shaded bands
    comparing focal approaches across strata.
    """
    set_publication_rcparams()

    if approaches is None:
        approaches = TARGET_APPROACHES

    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)

    # Filter for the target approaches and ensure stratum_scope is integer
    df_filtered = df[df["approach"].isin(approaches)].copy()
    df_filtered["stratum_scope"] = pd.to_numeric(df_filtered["stratum_scope"], errors="coerce")
    df_filtered = df_filtered.dropna(subset=["stratum_scope"])
    df_filtered["stratum_scope"] = df_filtered["stratum_scope"].astype(int)

    strata = sorted(df_filtered["stratum_scope"].unique())

    for approach in approaches:
        sub_df = df_filtered[df_filtered["approach"] == approach].sort_values("stratum_scope")
        if sub_df.empty:
            continue


        x = sub_df["stratum_scope"].values
        y = sub_df[metric_col].values
        sem = sub_df[sem_col].values
        ci = sem * 1.96  # 95% Confidence Interval half-width

        style = APPROACH_STYLES[approach]

        # 95% Confidence Interval shaded band (connecting CI intervals across strata)
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

    # Discrete x-ticks for strata 0, 1, 2, 3
    ax.set_xticks(strata)
    ax.set_xticklabels([STRATUM_LABELS.get(s, str(s)) for s in strata])
    ax.set_xlim(-0.3, max(strata) + 0.3)

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
    ax.set_xlabel("Extrapolation Stratum", fontsize=9.5, color="#111827", labelpad=6)
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


def generate_stratum_scaling_plots(
    input_csv: str,
    output_dir: str,
    approaches: List[str] | None = None,
    formats: List[str] = ["pdf", "png"],
) -> List[str]:
    """
    Load stratum_global.csv and generate 3 stratum scaling line plots
    for ECE, Winkler Score, and E-AURC comparing focal approaches.
    """
    df = pd.read_csv(input_csv)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. ECE Plot
    ece_files = plot_single_stratum_scaling(
        df=df,
        metric_col="ece_mean",
        sem_col="ece_sem",
        ylabel="Expected Calibration Error (ECE)",
        output_stem=out_path / "stratum_scaling_ece",
        approaches=approaches,
        formats=formats,
    )
    results.extend([str(p) for p in ece_files])

    # 2. Winkler Score Plot
    winkler_col = "winkler_mean_mean" if "winkler_mean_mean" in df.columns else "winkler_mean"
    winkler_sem = "winkler_mean_sem" if "winkler_mean_sem" in df.columns else "winkler_sem"
    winkler_files = plot_single_stratum_scaling(
        df=df,
        metric_col=winkler_col,
        sem_col=winkler_sem,
        ylabel="Mean Winkler Score",
        output_stem=out_path / "stratum_scaling_winkler_mean",
        approaches=approaches,
        formats=formats,
    )
    results.extend([str(p) for p in winkler_files])

    # 3. E-AURC Plot (Excess Area Under Risk-Coverage)
    e_aurc_files = plot_single_stratum_scaling(
        df=df,
        metric_col="e_aurc_mean",
        sem_col="e_aurc_sem",
        ylabel="Excess AURC (Selective Risk)",
        output_stem=out_path / "stratum_scaling_e_aurc",
        approaches=approaches,
        formats=formats,
    )
    results.extend([str(p) for p in e_aurc_files])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Plot stratum scaling curves for focal UQ approaches."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="results/extrapolation_uq/analysis/stratum_global.csv",
        help="Path to stratum_global.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/extrapolation_uq/analysis/plots/stratum_scaling",
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

    files = generate_stratum_scaling_plots(
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

