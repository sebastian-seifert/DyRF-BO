#!/usr/bin/env python3
"""
Publication-grade Visualizer for Global Overall UQ Benchmark results (N = 23,040).
Generates three horizontal bar plots (ECE, Winkler Score, E-AURC)
adhering to Nature / NeurIPS graphic standards:
- Established ColorBrewer / Seaborn 'flare' warm, colorblind-safe palette.
- Explicit Standard Error (SE) reported in bar annotations (mean ± SE).
- E-AURC (Excess AURC) instead of NAURC.
- Baseline reference line labeled compactly as 'Baseline (<value>)'.
- De-spined layout, horizontal bars sorted with best on top.
- Outputs both vector PDF and 300 DPI PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def clean_approach_name(approach: str) -> str:
    """Return clean, decluttered display name for an approach."""
    return APPROACH_DISPLAY_NAMES.get(approach, approach)


def get_warm_colorblind_palette(approaches: List[str]) -> List[str]:
    """
    Generate an established, colorblind-safe warm palette using Seaborn's
    perceptually uniform 'flare' template.
    Baselines are assigned neutral slate tones to distinguish them clearly.
    """
    # Generate 8 distinct warm tones from Seaborn's preset colorblind-safe 'flare' palette
    warm_colors = sns.color_palette("flare", n_colors=8).as_hex()
    
    # Map proximity families to warm shades:
    # BC (top performers) -> deep wine / rose
    # B -> warm terracotta / rust
    # AC -> amber coral
    # A -> golden apricot
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
        if a == "hutter_total":
            palette.append("#6B7280")       # Dignified Slate Gray
        elif a == "shaker_epistemic":
            palette.append("#374151")   # Dark Charcoal Gray
        else:
            palette.append(proximity_color_map.get(a, warm_colors[4]))
    return palette


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


def format_mean_se(mean_val: float, sem_val: float) -> str:
    """Format mean ± SE string dynamically based on magnitude."""
    if mean_val >= 10.0:
        return f"{mean_val:.1f} ± {sem_val:.1f}"
    elif mean_val >= 1.0:
        return f"{mean_val:.2f} ± {sem_val:.2f}"
    else:
        return f"{mean_val:.3f} ± {sem_val:.3f}"


def plot_single_metric_bar(
    df: pd.DataFrame,
    metric_col: str,
    err_col: str,
    title: str,
    xlabel: str,
    output_stem: Path,
    formats: List[str] = ["pdf", "png"],
) -> List[Path]:
    """
    Generate and save a horizontal bar plot for a single metric across all 10 approaches.
    Sorted with best performance (lowest metric value) on top.
    Error bars display standard deviation (_std).
    """
    set_publication_rcparams()

    # Sort ascending=False so that smallest values appear at the top in barh (y = max_y)
    df_sorted = df.sort_values(by=metric_col, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)

    y_pos = np.arange(len(df_sorted))
    means = df_sorted[metric_col].values
    errs = df_sorted[err_col].values
    approaches = df_sorted["approach"].values
    labels = [clean_approach_name(a) for a in approaches]
    colors = get_warm_colorblind_palette(list(approaches))

    # Lower whisker clipped at 0 since metrics are non-negative
    lower_err = np.minimum(means, errs)
    upper_err = errs
    xerr = [lower_err, upper_err]

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
            label="Baseline",
        )

    # Plot horizontal bars with standard deviation error bars
    bar_height = 0.60
    ax.barh(
        y_pos,
        means,
        xerr=xerr,
        height=bar_height,
        color=colors,
        edgecolor="#374151",
        linewidth=0.4,
        capsize=2.5,
        error_kw={
            "elinewidth": 0.8,
            "capthick": 0.8,
            "ecolor": "#374151",
        },
        zorder=2,
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

    # Subtle vertical gridlines only
    ax.grid(axis="x", linestyle="--", linewidth=0.55, color="#E5E7EB", alpha=0.85, zorder=0)
    ax.set_axisbelow(True)

    # Extend x-lim slightly for top-right legend headroom
    x_max_val = float((means + upper_err).max())
    ax.set_xlim(0, x_max_val * 1.08)

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


def generate_metric_bar_plots(
    input_csv: str,
    output_dir: str,
    formats: List[str] = ["pdf", "png"],
) -> List[str]:
    """
    Load main_global.csv and generate 3 standalone horizontal bar plots for ECE, Winkler Score, and E-AURC
    with standard deviation error bars.
    """
    df = pd.read_csv(input_csv)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. ECE Plot (using ece_std)
    ece_std_col = "ece_std" if "ece_std" in df.columns else "ece_sem"
    ece_files = plot_single_metric_bar(
        df=df,
        metric_col="ece_mean",
        err_col=ece_std_col,
        title="Expected Calibration Error (ECE)",
        xlabel="Expected Calibration Error (ECE)",
        output_stem=out_path / "global_ece",
        formats=formats,
    )
    results.extend([str(p) for p in ece_files])

    # 2. Winkler Score Plot (using winkler_mean_std)
    winkler_col = "winkler_mean_mean" if "winkler_mean_mean" in df.columns else "winkler_mean"
    winkler_std_col = "winkler_mean_std" if "winkler_mean_std" in df.columns else (
        "winkler_std" if "winkler_std" in df.columns else "winkler_mean_sem"
    )
    winkler_files = plot_single_metric_bar(
        df=df,
        metric_col=winkler_col,
        err_col=winkler_std_col,
        title="Winkler Score (Prediction Interval Quality)",
        xlabel="Mean Winkler Score",
        output_stem=out_path / "global_winkler_mean",
        formats=formats,
    )
    results.extend([str(p) for p in winkler_files])

    # 3. E-AURC Plot (using e_aurc_std)
    e_aurc_std_col = "e_aurc_std" if "e_aurc_std" in df.columns else "e_aurc_sem"
    e_aurc_files = plot_single_metric_bar(
        df=df,
        metric_col="e_aurc_mean",
        err_col=e_aurc_std_col,
        title="Excess Area Under Risk-Coverage (E-AURC)",
        xlabel="Excess AURC (Selective Risk)",
        output_stem=out_path / "global_e_aurc",
        formats=formats,
    )
    results.extend([str(p) for p in e_aurc_files])

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
