#!/usr/bin/env python3
"""
Publication-grade Visualizer for Global Overall UQ Benchmark results (N = 23,040).
Generates three publication-grade horizontal bar plots (ECE, Winkler Score, NAURC)
adhering to Nature / NeurIPS graphic standards:
- De-spined layout with subtle gridlines.
- Disciplined, semantic academic color palette (no rainbow/Tab10).
- Clean typographical hierarchy with Type 42 vector font embedding.
- Variance Decomposition baseline reference line with top-right legend.
- Horizontal bars sorted with best on top.
- Outputs both vector PDF and 300 DPI PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Approach display names mapping
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

# Refined academic palette: semantic hierarchy across method families
APPROACH_COLORS: Dict[str, str] = {
    # Baselines (Dignified Neutrals)
    "hutter_total": "#64748B",       # Slate Gray
    "shaker_epistemic": "#475569",   # Dark Slate Gray
    # BC Family (Deep Royal / Indigo - Best Overall Performers)
    "prox_bc_oob": "#1E3A8A",        # Deep Navy
    "prox_bc_cv": "#2563EB",         # Cobalt Blue
    # B Family (Deep / Muted Teal)
    "prox_b_oob": "#0D9488",         # Deep Teal
    "prox_b_cv": "#14B8A6",          # Muted Teal
    # AC Family (Indigo Slate)
    "prox_ac_oob": "#4338CA",        # Rich Indigo
    "prox_ac_cv": "#6366F1",         # Soft Indigo
    # A Family (Steel Ocean / Sky)
    "prox_a_oob": "#0284C7",         # Steel Blue
    "prox_a_cv": "#38BDF8",          # Sky Slate
}

# Edge colors for crisp, refined contrast
APPROACH_EDGE_COLORS: Dict[str, str] = {
    k: ("#1E293B" if "shaker" in k or "hutter" in k else v)
    for k, v in APPROACH_COLORS.items()
}


def clean_approach_name(approach: str) -> str:
    """Return clean, decluttered display name for an approach."""
    return APPROACH_DISPLAY_NAMES.get(approach, approach)


def set_publication_rcparams() -> None:
    """Configure matplotlib rcParams for camera-ready academic figures."""
    plt.rcParams.update({
        # Vector font embedding for camera-ready submission (Type 42 / TrueType)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Font family & typography
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans",
            "Helvetica Neue",
            "Helvetica",
            "Arial",
            "Liberation Sans",
        ],
        "mathtext.fontset": "dejavusans",
        # Font sizes
        "font.size": 9.0,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.5,
        # Spines and lines
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.0,
        # Rendering
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
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
    Generate and save a horizontal bar plot for a single metric across all 10 approaches.
    Sorted with best performance (lowest metric value) on top.
    """
    set_publication_rcparams()

    # Sort ascending=False so that smallest values appear at the top in barh (y = max_y)
    df_sorted = df.sort_values(by=metric_col, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=300)

    y_pos = np.arange(len(df_sorted))
    means = df_sorted[metric_col].values
    sems = df_sorted[sem_col].values
    approaches = df_sorted["approach"].values
    labels = [clean_approach_name(a) for a in approaches]
    colors = [APPROACH_COLORS.get(a, "#2563EB") for a in approaches]
    edgecolors = [APPROACH_EDGE_COLORS.get(a, "#1E293B") for a in approaches]

    # Baseline reference value (Variance Decomposition)
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
            zorder=3,
            label=f"Baseline: Variance Decomposition ({hutter_val:.4g})",
        )

    # Plot horizontal bars
    bar_height = 0.60
    ax.barh(
        y_pos,
        means,
        xerr=sems,
        height=bar_height,
        color=colors,
        edgecolor=edgecolors,
        linewidth=0.5,
        capsize=2.5,
        error_kw={
            "elinewidth": 0.8,
            "capthick": 0.8,
            "ecolor": "#475569",
        },
        zorder=2,
    )

    # Annotate numeric values cleanly at the tip of each bar
    x_max_val = float((means + sems).max())
    x_offset = x_max_val * 0.015
    for i, (mean_val, sem_val) in enumerate(zip(means, sems)):
        ax.text(
            mean_val + sem_val + x_offset,
            i,
            f"{mean_val:.4g}",
            va="center",
            ha="left",
            fontsize=8.0,
            color="#334155",
            fontweight="normal",
            zorder=4,
        )

    # Formatting axes and labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.0, color="#1E293B")
    ax.set_xlabel(xlabel, fontsize=9.5, color="#1E293B", labelpad=6)
    ax.set_title(title, fontsize=10.5, fontweight="semibold", color="#0F172A", pad=10)

    # De-spine (remove top and right spines, lighten bottom and left)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.spines["bottom"].set_linewidth(0.7)

    # Tick styling: no Y ticks, subtle X ticks
    ax.tick_params(axis="y", which="both", length=0)
    ax.tick_params(axis="x", which="both", length=3.5, width=0.7, colors="#64748B", labelsize=8.5)

    # Subtle vertical gridlines only
    ax.grid(axis="x", linestyle="--", linewidth=0.55, color="#E2E8F0", alpha=0.85, zorder=0)
    ax.set_axisbelow(True)

    # Extend x-lim slightly for annotations and top-right legend headroom
    ax.set_xlim(0, x_max_val * 1.25)

    # Minimalist, clean legend placed top-right
    if hutter_val is not None:
        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor="#E2E8F0",
            framealpha=0.95,
            fontsize=8.2,
            handlelength=1.6,
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
        title="Winkler Score (Prediction Interval Quality)",
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
    parser = argparse.ArgumentParser(description="Plot global overall metrics as publication-grade horizontal bar plots.")
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
