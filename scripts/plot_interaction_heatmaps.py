#!/usr/bin/env python3
"""
Publication-grade 2D Interaction Heatmap Generator for Extrapolation UQ Benchmark.
Visualizes the pairwise 2D interaction planes from the 3D OLAP Cube:
  1. Stratum x Dimension (4 Strata x 8 Dimensions)
  2. Stratum x Surrogate (4 Strata x 4 Surrogates)
  3. Surrogate x Dimension (4 Surrogates x 8 Dimensions)

Across 3 key performance metrics (9 heatmaps total):
  - ECE (Expected Calibration Error)
  - Winkler Score (Prediction Interval Quality)
  - Excess AURC (Selective Risk / Rejection Quality)

Each cell shows the relative improvement (%) of Proximity BC (OOB) over Variance Decomposition:
  RelGain (%) = (Baseline - Proximity) / Baseline * 100%
(Positive values indicate Proximity outperforms baseline, colored in warm terracotta).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

STRATUM_DISPLAY_NAMES: Dict[int, str] = {
    0: "0 (Interp)",
    1: "1 (Near)",
    2: "2 (Moderate)",
    3: "3 (Far)",
}

SURROGATE_ORDER: List[str] = ["smac_default", "mature", "shallow", "coarse"]
SURROGATE_DISPLAY_NAMES: Dict[str, str] = {
    "smac_default": "SMAC Default",
    "mature": "Mature",
    "shallow": "Shallow",
    "coarse": "Coarse",
}

DIMENSION_ORDER: List[int] = [2, 3, 5, 8, 16, 32, 64, 128]

METRIC_CONFIGS = [
    {
        "key": "ece",
        "col": "ece_mean",
        "title": "ECE Relative Improvement",
        "filename": "heatmap_{pair}_ece",
    },
    {
        "key": "winkler",
        "col": "winkler_mean_mean",
        "fallback_col": "winkler_mean",
        "title": "Winkler Score Relative Improvement",
        "filename": "heatmap_{pair}_winkler",
    },
    {
        "key": "e_aurc",
        "col": "e_aurc_mean",
        "title": "Excess AURC Relative Improvement",
        "filename": "heatmap_{pair}_e_aurc",
    },
]


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
        "font.size": 8.5,
        "axes.titlesize": 10.0,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def create_warm_diverging_cmap() -> mcolors.LinearSegmentedColormap:
    """
    Creates an academic warm diverging colormap:
    - Slate / Dark Stone (#334155 / #475569) for negative relative gains (Baseline better)
    - Neutral light cream / white (#F5F5F4 / #FFFFFF) at 0% parity
    - Warm Terracotta / Rust (#C2410C / #9A3412) for positive relative gains (Proximity better)
    """
    colors = [
        "#1E293B",  # Deep Slate (strong negative)
        "#64748B",  # Medium Slate
        "#F1F5F9",  # Light neutral off-white (parity 0%)
        "#FB923C",  # Light Amber/Terracotta
        "#C2410C",  # Rich Terracotta (strong positive)
    ]
    return mcolors.LinearSegmentedColormap.from_list("academic_diverging", colors, N=256)


def compute_relative_gain_matrix(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    metric_col: str,
    baseline_approach: str = "hutter_total",
    target_approach: str = "prox_bc_oob",
) -> pd.DataFrame:
    """
    Computes relative improvement matrix (%):
      RelGain = (Metric_baseline - Metric_target) / Metric_baseline * 100%
    """
    sub_df = df[df["approach"].isin([baseline_approach, target_approach])].copy()

    # Pivot to get baseline and target columns
    pivoted = sub_df.pivot_table(
        index=[row_col, col_col],
        columns="approach",
        values=metric_col,
    )

    if baseline_approach not in pivoted.columns or target_approach not in pivoted.columns:
        raise ValueError(f"Missing required approaches {baseline_approach} or {target_approach}")

    base_vals = pivoted[baseline_approach]
    target_vals = pivoted[target_approach]

    # Lower metric is better -> positive gain means target is lower (better)
    rel_gain = (base_vals - target_vals) / base_vals * 100.0
    matrix = rel_gain.unstack(level=col_col)
    return matrix


def plot_single_heatmap(
    matrix: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    metric_title: str,
    output_stem: Path,
    xticklabels: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
    formats: List[str] = ["pdf", "png"],
) -> List[Path]:
    """Plots and saves an academic interaction heatmap with relative percentage annotations."""
    set_publication_rcparams()

    # Determine symmetric or robust bounds centered at 0
    vals = matrix.values.flatten()
    valid_vals = vals[~np.isnan(vals)]
    max_abs = max(10.0, float(np.percentile(np.abs(valid_vals), 98))) if len(valid_vals) > 0 else 50.0
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = create_warm_diverging_cmap()

    # Determine figure dimensions based on matrix size
    nrows, ncols = matrix.shape
    fig_w = max(4.6, 1.2 + ncols * 0.55)
    fig_h = max(3.4, 1.2 + nrows * 0.55)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    # Format annotation text with explicit +/- signs
    annot_matrix = matrix.map(lambda v: f"{v:+.1f}%" if pd.notnull(v) else "")


    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        norm=norm,
        annot=annot_matrix.values,
        fmt="",
        annot_kws={"fontsize": 7.5, "fontweight": "medium"},
        linewidths=0.75,
        linecolor="#FFFFFF",
        cbar_kws={
            "label": "Relative Improvement (%)",
            "shrink": 0.82,
            "pad": 0.04,
            "aspect": 18,
        },
    )

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=0, ha="center")
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, rotation=0, va="center")

    ax.set_xlabel(xlabel, fontsize=9.5, color="#111827", labelpad=7)
    ax.set_ylabel(ylabel, fontsize=9.5, color="#111827", labelpad=7)
    ax.tick_params(axis="both", which="both", length=0, labelsize=8.5)

    # Clean border styling
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("#E5E7EB")
        spine.set_linewidth(0.8)

    plt.tight_layout()

    saved_files = []
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target_path = output_stem.with_suffix(f".{fmt}")
        fig.savefig(target_path, bbox_inches="tight")
        saved_files.append(target_path)

    plt.close(fig)
    return saved_files


def generate_all_interaction_heatmaps(
    stratum_dim_csv: str,
    stratum_surrogate_csv: str,
    surrogate_dim_csv: str,
    output_dir: str,
    baseline_approach: str = "hutter_total",
    target_approach: str = "prox_bc_oob",
    formats: List[str] = ["pdf", "png"],
) -> List[str]:
    """Generates all 9 2D interaction heatmaps across the 3 planes and 3 metrics."""
    out_path = Path(output_dir)
    results: List[str] = []

    # --- Plane 1: Stratum x Dimension ---
    if Path(stratum_dim_csv).exists():
        df_sd = pd.read_csv(stratum_dim_csv)
        df_sd["stratum_scope"] = pd.to_numeric(df_sd["stratum_scope"], errors="coerce")
        df_sd = df_sd.dropna(subset=["stratum_scope"])
        df_sd["stratum_scope"] = df_sd["stratum_scope"].astype(int)

        strata_available = sorted(df_sd["stratum_scope"].unique())
        dims_available = sorted(df_sd["dimension"].unique())

        sub_out = out_path / "stratum_by_dimension"
        for m in METRIC_CONFIGS:
            metric_col = m["col"] if m["col"] in df_sd.columns else m.get("fallback_col", m["col"])
            mat = compute_relative_gain_matrix(
                df=df_sd,
                row_col="stratum_scope",
                col_col="dimension",
                metric_col=metric_col,
                baseline_approach=baseline_approach,
                target_approach=target_approach,
            )
            mat = mat.reindex(index=strata_available, columns=dims_available)

            files = plot_single_heatmap(
                matrix=mat,
                xlabel="Dimension ($D$)",
                ylabel="Extrapolation Stratum",
                metric_title=m["title"],
                output_stem=sub_out / m["filename"].format(pair="stratum_dim"),
                xticklabels=[str(d) for d in dims_available],
                yticklabels=[STRATUM_DISPLAY_NAMES.get(s, str(s)) for s in strata_available],
                formats=formats,
            )
            results.extend([str(p) for p in files])

    # --- Plane 2: Stratum x Surrogate ---
    if Path(stratum_surrogate_csv).exists():
        df_ss = pd.read_csv(stratum_surrogate_csv)
        df_ss["stratum_scope"] = pd.to_numeric(df_ss["stratum_scope"], errors="coerce")
        df_ss = df_ss.dropna(subset=["stratum_scope"])
        df_ss["stratum_scope"] = df_ss["stratum_scope"].astype(int)

        strata_available = sorted(df_ss["stratum_scope"].unique())
        surrogates_available = [s for s in SURROGATE_ORDER if s in df_ss["surrogate"].unique()]

        sub_out = out_path / "stratum_by_surrogate"
        for m in METRIC_CONFIGS:
            metric_col = m["col"] if m["col"] in df_ss.columns else m.get("fallback_col", m["col"])
            mat = compute_relative_gain_matrix(
                df=df_ss,
                row_col="stratum_scope",
                col_col="surrogate",
                metric_col=metric_col,
                baseline_approach=baseline_approach,
                target_approach=target_approach,
            )
            mat = mat.reindex(index=strata_available, columns=surrogates_available)

            files = plot_single_heatmap(
                matrix=mat,
                xlabel="Random Forest Surrogate",
                ylabel="Extrapolation Stratum",
                metric_title=m["title"],
                output_stem=sub_out / m["filename"].format(pair="stratum_surrogate"),
                xticklabels=[SURROGATE_DISPLAY_NAMES.get(s, s) for s in surrogates_available],
                yticklabels=[STRATUM_DISPLAY_NAMES.get(s, str(s)) for s in strata_available],
                formats=formats,
            )
            results.extend([str(p) for p in files])

    # --- Plane 3: Surrogate x Dimension ---
    if Path(surrogate_dim_csv).exists():
        df_rd = pd.read_csv(surrogate_dim_csv)
        surrogates_available = [s for s in SURROGATE_ORDER if s in df_rd["surrogate"].unique()]
        dims_available = sorted(df_rd["dimension"].unique())

        sub_out = out_path / "surrogate_by_dimension"
        for m in METRIC_CONFIGS:
            metric_col = m["col"] if m["col"] in df_rd.columns else m.get("fallback_col", m["col"])
            mat = compute_relative_gain_matrix(
                df=df_rd,
                row_col="surrogate",
                col_col="dimension",
                metric_col=metric_col,
                baseline_approach=baseline_approach,
                target_approach=target_approach,
            )
            mat = mat.reindex(index=surrogates_available, columns=dims_available)

            files = plot_single_heatmap(
                matrix=mat,
                xlabel="Dimension ($D$)",
                ylabel="Random Forest Surrogate",
                metric_title=m["title"],
                output_stem=sub_out / m["filename"].format(pair="surrogate_dim"),
                xticklabels=[str(d) for d in dims_available],
                yticklabels=[SURROGATE_DISPLAY_NAMES.get(s, s) for s in surrogates_available],
                formats=formats,
            )
            results.extend([str(p) for p in files])

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate 2D interaction heatmaps for Extrapolation UQ.")
    parser.add_argument(
        "--stratum-dim-csv",
        type=str,
        default="results/extrapolation_uq/analysis/stratum_by_dimension.csv",
        help="Path to stratum_by_dimension.csv",
    )
    parser.add_argument(
        "--stratum-surrogate-csv",
        type=str,
        default="results/extrapolation_uq/analysis/stratum_by_surrogate.csv",
        help="Path to stratum_by_surrogate.csv",
    )
    parser.add_argument(
        "--surrogate-dim-csv",
        type=str,
        default="results/extrapolation_uq/analysis/main_global_rf_by_dimension.csv",
        help="Path to main_global_rf_by_dimension.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/extrapolation_uq/analysis/plots/heatmaps",
        help="Directory to save generated heatmaps.",
    )
    parser.add_argument(
        "--target-approach",
        type=str,
        default="prox_bc_oob",
        help="Target approach to evaluate against baseline (default: prox_bc_oob).",
    )
    parser.add_argument(
        "--baseline-approach",
        type=str,
        default="hutter_total",
        help="Baseline approach (default: hutter_total).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats (e.g. pdf png).",
    )
    args = parser.parse_args()

    files = generate_all_interaction_heatmaps(
        stratum_dim_csv=args.stratum_dim_csv,
        stratum_surrogate_csv=args.stratum_surrogate_csv,
        surrogate_dim_csv=args.surrogate_dim_csv,
        output_dir=args.output_dir,
        baseline_approach=args.baseline_approach,
        target_approach=args.target_approach,
        formats=args.formats,
    )
    print(f"Generated {len(files)} interaction heatmap files in {args.output_dir}:")
    for f in files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
