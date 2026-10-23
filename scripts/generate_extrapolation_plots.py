#!/usr/bin/env python3
"""Unified Publication-Grade Plot Generator for Extrapolation Uncertainty Quantification.

Renders:
1. Global Point Plots (whiskers = ± 1.96 * SEM, best on top, baseline at Between-Tree Disagreement).
2. Dimension Scaling Plots (discrete log-scale ticks, ± 1.96 * SEM shaded CI bands).
3. Stratum Scaling Plots (strata 0-3, ± 1.96 * SEM shaded CI bands).
4. Surrogate Scaling Plots (category point plot with ± 1.96 * SEM error whiskers).

Generates two complete figure trees:
- `focal_trio`: Proximity BC (OOB), Shaker Entropy, Between-Tree Disagreement
- `with_variance_decomposition`: adds Variance Decomposition (Total) as 4th approach
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Display names mapping
APPROACH_DISPLAY_NAMES: Dict[str, str] = {
    "prox_bc_oob": "Proximity BC (OOB)",
    "prox_bc_cv": "Proximity BC (CV)",
    "prox_b_oob": "Proximity B (OOB)",
    "prox_b_cv": "Proximity B (CV)",
    "prox_ac_oob": "Proximity AC (OOB)",
    "prox_ac_cv": "Proximity AC (CV)",
    "prox_a_oob": "Proximity A (OOB)",
    "prox_a_cv": "Proximity A (CV)",
    "shaker_epistemic": "Shaker Entropy",
    "hutter_between": "Between-Tree Disagreement",
    "hutter_total": "Variance Decomposition",
}

# Approaches configurations
FOCAL_TRIO: List[str] = ["prox_bc_oob", "shaker_epistemic", "hutter_between"]
QUARTET_WITH_TOTAL: List[str] = ["prox_bc_oob", "shaker_epistemic", "hutter_between", "hutter_total"]

# Visual styling per approach
APPROACH_STYLES: Dict[str, Dict[str, any]] = {
    "prox_bc_oob": {
        "color": "#C2410C",             # Warm Terracotta / Rust
        "linestyle": "-",
        "marker": "o",
        "label": "Proximity BC (OOB)",
    },
    "shaker_epistemic": {
        "color": "#D97706",             # Warm Golden Amber
        "linestyle": "-.",
        "marker": "^",
        "label": "Shaker Entropy",
    },
    "hutter_between": {
        "color": "#44403C",             # Warm Dark Stone
        "linestyle": "--",
        "marker": "s",
        "label": "Between-Tree Disagreement",
    },
    "hutter_total": {
        "color": "#475569",             # Slate Blue
        "linestyle": ":",
        "marker": "D",
        "label": "Variance Decomposition",
    },
}

METRIC_SPECS = [
    {
        "stem": "ece",
        "mean_col": "ece_mean",
        "sem_col": "ece_sem",
        "title": "Expected Calibration Error (ECE)",
        "label": "ECE (lower is better)",
        "fmt": ".3f",
    },
    {
        "stem": "winkler_mean",
        "mean_col": "winkler_mean_mean",
        "sem_col": "winkler_mean_sem",
        "title": "Multi-Level Winkler Score",
        "label": "Winkler Score (lower is better)",
        "fmt": ".1f",
    },
    {
        "stem": "e_aurc",
        "mean_col": "e_aurc_mean",
        "sem_col": "e_aurc_sem",
        "title": "Excess AURC (E-AURC)",
        "label": "E-AURC (lower is better)",
        "fmt": ".3f",
    },
]

STRATUM_LABELS: Dict[int, str] = {
    0: "0\n(Interpolation)",
    1: "1\n(Near)",
    2: "2\n(Moderate)",
    3: "3\n(Far)",
}

SURROGATE_ORDER: List[str] = ["smac_default", "mature", "shallow", "coarse"]
SURROGATE_DISPLAY_NAMES: Dict[str, str] = {
    "smac_default": "SMAC Default",
    "mature": "Mature",
    "shallow": "Shallow",
    "coarse": "Coarse",
}


def clean_approach_name(approach: str) -> str:
    """Return clean human-readable name for an approach."""
    return APPROACH_DISPLAY_NAMES.get(approach, approach)


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


def get_warm_colorblind_palette(approaches: List[str]) -> List[str]:
    """Assign established colorblind-safe tones to approaches for global plots."""
    warm_colors = sns.color_palette("flare", n_colors=8).as_hex()
    proximity_color_map = {
        "prox_bc_oob": warm_colors[7],
        "prox_bc_cv": warm_colors[6],
        "prox_b_oob": warm_colors[5],
        "prox_b_cv": warm_colors[4],
        "prox_ac_oob": warm_colors[3],
        "prox_ac_cv": warm_colors[2],
        "prox_a_oob": warm_colors[1],
        "prox_a_cv": warm_colors[0],
    }
    palette = []
    for a in approaches:
        if a == "hutter_between":
            palette.append("#44403C")       # Dark Stone
        elif a == "hutter_total":
            palette.append("#6B7280")       # Slate
        elif a == "shaker_epistemic":
            palette.append("#D97706")       # Warm Golden Amber
        else:
            palette.append(proximity_color_map.get(a, warm_colors[4]))
    return palette


# ---------------------------------------------------------------------------
# 1. Global Point Plots
# ---------------------------------------------------------------------------

def plot_global_point_plots(
    df: pd.DataFrame,
    output_dir: Path,
    exclude_approaches: Optional[List[str]] = None,
    formats: List[str] = ["pdf", "png"],
) -> List[Path]:
    """Generate horizontal point plots with 95% Confidence Intervals (± 1.96 * SEM)."""
    set_publication_rcparams()
    output_dir.mkdir(parents=True, exist_ok=True)
    if exclude_approaches is None:
        exclude_approaches = ["hutter_total"]

    df_plot = df[~df["approach"].isin(exclude_approaches)].copy()
    saved_files: List[Path] = []

    for spec in METRIC_SPECS:
        mean_col = spec["mean_col"]
        sem_col = spec["sem_col"]
        if mean_col not in df_plot.columns or sem_col not in df_plot.columns:
            continue

        # Sort with best (lowest) on top (so ascending=False in horizontal layout)
        df_sorted = df_plot.sort_values(by=mean_col, ascending=False).reset_index(drop=True)

        fig_height = max(2.4, 0.38 * len(df_sorted))
        fig, ax = plt.subplots(figsize=(6.0, fig_height), dpi=300)

        y_pos = np.arange(len(df_sorted))
        means = df_sorted[mean_col].values
        sems = df_sorted[sem_col].values
        ci_err = sems * 1.96  # Exact 95% Confidence Interval half-width
        approaches = df_sorted["approach"].values
        labels = [clean_approach_name(a) for a in approaches]
        colors = get_warm_colorblind_palette(list(approaches))

        # Baseline reference line at Between-Tree Disagreement
        hutter_mask = df_sorted["approach"] == "hutter_between"
        if hutter_mask.any():
            hutter_val = df_sorted.loc[hutter_mask, mean_col].iloc[0]
            ax.axvline(
                x=hutter_val,
                color="#78716C",
                linestyle="--",
                linewidth=0.8,
                alpha=0.85,
                zorder=1,
            )

        # Plot error whiskers and points
        for i in range(len(df_sorted)):
            ax.errorbar(
                x=means[i],
                y=y_pos[i],
                xerr=ci_err[i],
                fmt="o",
                color=colors[i],
                ecolor=colors[i],
                elinewidth=1.2,
                capsize=2.5,
                capthick=1.0,
                markersize=4.8,
                zorder=3,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel(spec["label"])
        ax.set_title(f"Global Extrapolation {spec['title']} (95% CI)", pad=10)

        sns.despine(ax=ax, left=False, bottom=False)
        ax.xaxis.grid(True, linestyle=":", alpha=0.4, color="#9CA3AF")
        ax.set_axisbelow(True)

        # Standard padding on x-limits
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        ax.set_xlim(x_min - 0.03 * x_range, x_max + 0.03 * x_range)

        stem_path = output_dir / f"global_{spec['stem']}"
        for fmt in formats:
            p = stem_path.with_suffix(f".{fmt}")
            fig.savefig(p, bbox_inches="tight")
            saved_files.append(p)
        plt.close(fig)

    return saved_files


# ---------------------------------------------------------------------------
# 2. Dimension Scaling Plots
# ---------------------------------------------------------------------------

def plot_dimension_scaling(
    df: pd.DataFrame,
    approaches: List[str],
    output_dir: Path,
    formats: List[str] = ["pdf", "png"],
) -> List[Path]:
    """Plot dimension scaling curves with 95% CI shaded bands across dimensions."""
    set_publication_rcparams()
    output_dir.mkdir(parents=True, exist_ok=True)
    df_filtered = df[df["approach"].isin(approaches)].copy()
    dims = sorted(df_filtered["dimension"].unique())
    saved_files: List[Path] = []

    for spec in METRIC_SPECS:
        mean_col = spec["mean_col"]
        sem_col = spec["sem_col"]
        if mean_col not in df_filtered.columns or sem_col not in df_filtered.columns:
            continue

        fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)

        for approach in approaches:
            sub = df_filtered[df_filtered["approach"] == approach].sort_values("dimension")
            if sub.empty:
                continue

            x = sub["dimension"].values
            y = sub[mean_col].values
            sem = sub[sem_col].values
            ci = sem * 1.96

            style = APPROACH_STYLES.get(approach, {
                "color": "#333333", "linestyle": "-", "marker": "o", "label": approach
            })

            # 95% Confidence Interval shaded band
            ax.fill_between(
                x,
                y - ci,
                y + ci,
                facecolor=style["color"],
                edgecolor=style["color"],
                alpha=0.25,
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

        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])
        ax.minorticks_off()

        ax.set_xlabel("Dimension D")
        ax.set_ylabel(spec["label"])
        ax.set_title(f"Dimension Scaling — {spec['title']}", pad=10)

        ax.legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="#E5E7EB")
        sns.despine(ax=ax)
        ax.grid(True, linestyle=":", alpha=0.4, color="#9CA3AF")
        ax.set_axisbelow(True)

        stem_path = output_dir / f"dimension_scaling_{spec['stem']}"
        for fmt in formats:
            p = stem_path.with_suffix(f".{fmt}")
            fig.savefig(p, bbox_inches="tight")
            saved_files.append(p)
        plt.close(fig)

    return saved_files


# ---------------------------------------------------------------------------
# 3. Stratum Scaling Plots
# ---------------------------------------------------------------------------

def plot_stratum_scaling(
    df: pd.DataFrame,
    approaches: List[str],
    output_dir: Path,
    formats: List[str] = ["pdf", "png"],
) -> List[Path]:
    """Plot stratum scaling curves (strata 0-3) with 95% CI shaded bands."""
    set_publication_rcparams()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter numeric strata 0, 1, 2, 3 (exclude 'all')
    df_filtered = df[df["stratum_scope"].isin([0, 1, 2, 3, "0", "1", "2", "3"])].copy()
    df_filtered["stratum"] = df_filtered["stratum_scope"].astype(int)
    df_filtered = df_filtered[df_filtered["approach"].isin(approaches)].copy()

    saved_files: List[Path] = []

    for spec in METRIC_SPECS:
        mean_col = spec["mean_col"]
        sem_col = spec["sem_col"]
        if mean_col not in df_filtered.columns or sem_col not in df_filtered.columns:
            continue

        fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)

        for approach in approaches:
            sub = df_filtered[df_filtered["approach"] == approach].sort_values("stratum")
            if sub.empty:
                continue

            x = sub["stratum"].values
            y = sub[mean_col].values
            sem = sub[sem_col].values
            ci = sem * 1.96

            style = APPROACH_STYLES.get(approach, {
                "color": "#333333", "linestyle": "-", "marker": "o", "label": approach
            })

            # Shaded CI band
            ax.fill_between(
                x,
                y - ci,
                y + ci,
                facecolor=style["color"],
                edgecolor=style["color"],
                alpha=0.25,
                linewidth=0.6,
                zorder=2,
            )

            # Line plot
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

        strata_ticks = [0, 1, 2, 3]
        ax.set_xticks(strata_ticks)
        ax.set_xticklabels([STRATUM_LABELS.get(s, str(s)) for s in strata_ticks])

        ax.set_xlabel("Extrapolation Stratum")
        ax.set_ylabel(spec["label"])
        ax.set_title(f"Stratum Scaling — {spec['title']}", pad=10)

        ax.legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="#E5E7EB")
        sns.despine(ax=ax)
        ax.grid(True, linestyle=":", alpha=0.4, color="#9CA3AF")
        ax.set_axisbelow(True)

        stem_path = output_dir / f"stratum_scaling_{spec['stem']}"
        for fmt in formats:
            p = stem_path.with_suffix(f".{fmt}")
            fig.savefig(p, bbox_inches="tight")
            saved_files.append(p)
        plt.close(fig)

    return saved_files


# ---------------------------------------------------------------------------
# 4. Surrogate Scaling Plots
# ---------------------------------------------------------------------------

def plot_surrogate_scaling(
    df: pd.DataFrame,
    approaches: List[str],
    output_dir: Path,
    formats: List[str] = ["pdf", "png"],
) -> List[Path]:
    """Plot surrogate scaling point plots with 95% CI error whiskers across RF surrogates."""
    set_publication_rcparams()
    output_dir.mkdir(parents=True, exist_ok=True)
    df_filtered = df[df["approach"].isin(approaches)].copy()

    # Determine surrogate order present
    surrogates = [s for s in SURROGATE_ORDER if s in df_filtered["surrogate"].unique()]
    if not surrogates:
        surrogates = sorted(df_filtered["surrogate"].unique())

    saved_files: List[Path] = []
    x_base = np.arange(len(surrogates))
    n_app = len(approaches)
    # Slight horizontal jitter for clarity
    jitter_width = 0.28
    offsets = np.linspace(-jitter_width / 2, jitter_width / 2, n_app) if n_app > 1 else [0.0]

    for spec in METRIC_SPECS:
        mean_col = spec["mean_col"]
        sem_col = spec["sem_col"]
        if mean_col not in df_filtered.columns or sem_col not in df_filtered.columns:
            continue

        fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=300)

        for i, approach in enumerate(approaches):
            style = APPROACH_STYLES.get(approach, {
                "color": "#333333", "linestyle": "-", "marker": "o", "label": approach
            })

            x_coords = []
            y_coords = []
            ci_coords = []

            for idx, surr in enumerate(surrogates):
                sub = df_filtered[(df_filtered["surrogate"] == surr) & (df_filtered["approach"] == approach)]
                if sub.empty:
                    continue
                x_coords.append(x_base[idx] + offsets[i])
                y_coords.append(sub[mean_col].iloc[0])
                ci_coords.append(sub[sem_col].iloc[0] * 1.96)

            if not x_coords:
                continue

            ax.errorbar(
                x=x_coords,
                y=y_coords,
                yerr=ci_coords,
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                label=style["label"],
                elinewidth=1.2,
                capsize=3.0,
                capthick=1.0,
                markersize=5.5,
                markeredgecolor="#1F2937",
                markeredgewidth=0.5,
                zorder=3,
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels([SURROGATE_DISPLAY_NAMES.get(s, s) for s in surrogates])
        ax.set_xlabel("Random Forest Surrogate")
        ax.set_ylabel(spec["label"])
        ax.set_title(f"Surrogate Scaling — {spec['title']}", pad=10)

        ax.legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="#E5E7EB")
        sns.despine(ax=ax)
        ax.grid(True, linestyle=":", alpha=0.4, color="#9CA3AF")
        ax.set_axisbelow(True)

        stem_path = output_dir / f"surrogate_scaling_{spec['stem']}"
        for fmt in formats:
            p = stem_path.with_suffix(f".{fmt}")
            fig.savefig(p, bbox_inches="tight")
            saved_files.append(p)
        plt.close(fig)

    return saved_files


# ---------------------------------------------------------------------------
# Master Orchestrator
# ---------------------------------------------------------------------------

def generate_all_extrapolation_plots(
    csv_dir: Path,
    output_dir: Path,
    formats: List[str] = ["pdf", "png"],
) -> Dict[str, List[Path]]:
    """Generate both focal_trio and with_variance_decomposition plot suites."""
    csv_main_global = csv_dir / "main_global.csv"
    csv_stratum_global = csv_dir / "stratum_global.csv"
    csv_by_dimension = csv_dir / "main_global_by_dimension.csv"
    csv_by_surrogate = csv_dir / "main_global_by_surrogate.csv"

    for req in [csv_main_global, csv_stratum_global, csv_by_dimension, csv_by_surrogate]:
        if not req.exists():
            raise FileNotFoundError(f"Required CSV not found: {req}")

    df_main_global = pd.read_csv(csv_main_global)
    df_stratum_global = pd.read_csv(csv_stratum_global)
    df_by_dimension = pd.read_csv(csv_by_dimension)
    df_by_surrogate = pd.read_csv(csv_by_surrogate)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, List[Path]] = {}

    # Configuration 1: focal_trio
    trio_dir = output_dir / "focal_trio"
    trio_global = plot_global_point_plots(
        df=df_main_global,
        output_dir=trio_dir / "global",
        exclude_approaches=["hutter_total"],
        formats=formats,
    )
    trio_dim = plot_dimension_scaling(
        df=df_by_dimension,
        approaches=FOCAL_TRIO,
        output_dir=trio_dir / "dimension",
        formats=formats,
    )
    trio_stratum = plot_stratum_scaling(
        df=df_stratum_global,
        approaches=FOCAL_TRIO,
        output_dir=trio_dir / "stratum",
        formats=formats,
    )
    trio_surr = plot_surrogate_scaling(
        df=df_by_surrogate,
        approaches=FOCAL_TRIO,
        output_dir=trio_dir / "surrogate",
        formats=formats,
    )
    generated["focal_trio"] = trio_global + trio_dim + trio_stratum + trio_surr

    # Configuration 2: with_variance_decomposition
    var_dir = output_dir / "with_variance_decomposition"
    var_global = plot_global_point_plots(
        df=df_main_global,
        output_dir=var_dir / "global",
        exclude_approaches=[],  # Keep all approaches including hutter_total
        formats=formats,
    )
    var_dim = plot_dimension_scaling(
        df=df_by_dimension,
        approaches=QUARTET_WITH_TOTAL,
        output_dir=var_dir / "dimension",
        formats=formats,
    )
    var_stratum = plot_stratum_scaling(
        df=df_stratum_global,
        approaches=QUARTET_WITH_TOTAL,
        output_dir=var_dir / "stratum",
        formats=formats,
    )
    var_surr = plot_surrogate_scaling(
        df=df_by_surrogate,
        approaches=QUARTET_WITH_TOTAL,
        output_dir=var_dir / "surrogate",
        formats=formats,
    )
    generated["with_variance_decomposition"] = var_global + var_dim + var_stratum + var_surr

    print(f"[INFO] Generated {len(generated['focal_trio'])} files in {trio_dir}")
    print(f"[INFO] Generated {len(generated['with_variance_decomposition'])} files in {var_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade extrapolation UQ plots.")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("results/extrapolation_uq/analysis"),
        help="Directory containing the 4 analysis CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/extrapolation_uq/analysis/plots_updated"),
        help="Target output directory for the generated plots.",
    )

    args = parser.parse_args()
    generate_all_extrapolation_plots(csv_dir=args.csv_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
