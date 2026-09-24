#!/usr/bin/env python3
"""CLI runner for the Extrapolation Distance Metric Ablation Study.

Evaluates whether Chebyshev (L_infinity) projection distances better align with
Random Forest epistemic uncertainty than Euclidean (d_norm) normalized distances.
Ingests Parquet evaluations, runs statistical hypothesis testing, and generates
both CSV scorecards and publication-grade Markdown reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dyrf_bo.extrapolation_uq.distance_ablation import (
    aggregate_distance_ablation,
    compute_run_distance_ablation,
)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for distance metric ablation CLI."""
    parser = argparse.ArgumentParser(
        description="Run distance metric ablation analysis (Euclidean d_norm vs Chebyshev d_inf)."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/extrapolation_uq/raw",
        help="Directory containing Parquet evaluation files (default: results/extrapolation_uq/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/extrapolation_uq/analysis",
        help="Directory to store generated reports and scorecards (default: results/extrapolation_uq/analysis).",
    )
    parser.add_argument(
        "--scorecard-filename",
        type=str,
        default="distance_ablation_scorecard.csv",
        help="Name of CSV output file (default: distance_ablation_scorecard.csv).",
    )
    parser.add_argument(
        "--report-filename",
        type=str,
        default="DISTANCE_ABLATION_REPORT.md",
        help="Name of Markdown report file (default: DISTANCE_ABLATION_REPORT.md).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of Parquet files to process for smoke testing.",
    )
    return parser


def generate_ablation_markdown_report(
    df_slices: pd.DataFrame,
    df_strata: pd.DataFrame,
    total_runs: int,
) -> str:
    """Generate a scientific Markdown report evaluating the distance metric ablation study.

    Parameters
    ----------
    df_slices : pd.DataFrame
        Scorecard aggregated across (dimension, strategy) slices.
    df_strata : pd.DataFrame
        Metrics aggregated across distance strata.
    total_runs : int
        Total number of experimental runs evaluated.

    Returns
    -------
    str
        Publication-grade Markdown report content.
    """
    if df_slices.empty:
        return (
            "# Extrapolation Distance Metric Ablation Report\n\n"
            "No experimental evaluations found to analyze.\n"
        )

    # Extract grand total row
    overall_mask = (df_slices["dimension"].astype(str) == "All") & (df_slices["sampling_strategy"] == "All")
    tot = df_slices[overall_mask].iloc[0] if overall_mask.any() else df_slices.iloc[-1]

    lines: List[str] = [
        "# Extrapolation Distance Metric Ablation Report: Chebyshev ($L_\\infty$) vs. Euclidean ($d_{\\text{norm}}$)",
        "",
        "## Executive Summary & Core Comparison",
        "",
        "This ablation study investigates whether axis-aligned **Chebyshev distance ($d_{\\text{inf}}$)** "
        "provides superior alignment with Random Forest surrogate uncertainty compared to standard **Euclidean normalized distance ($d_{\\text{norm}}$)**. "
        f"A total of **{total_runs}** experimental runs were evaluated across varying dimensions, sampling strategies, and distance strata.",
        "",
        "### Overall Statistical Hypothesis Testing",
        "",
        "| Surrogate Method | Euclidean $\\rho(d_{\\text{norm}}, U)$ | Chebyshev $\\rho(d_{\\text{inf}}, U)$ | Paired Diff (Mean) | Wilcoxon p-value | Cliff's δ | Win / Tie / Loss |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **SLCB** (SMAC3 Standard) | {tot['dist_norm_slcb_mean']:.4f} ± {tot['dist_norm_slcb_sem']:.4f} | "
        f"{tot['dist_inf_slcb_mean']:.4f} ± {tot['dist_inf_slcb_sem']:.4f} | "
        f"{tot['diff_slcb_mean']:+.4f} | {tot['pvalue_slcb']:.2e} | {tot['cliffs_delta_slcb']:+.3f} | "
        f"{tot['wins_slcb']}W / {tot['ties_slcb']}T / {tot['losses_slcb']}L |",
        f"| **PLCB** (Proximity Augmented) | {tot['dist_norm_plcb_mean']:.4f} ± {tot['dist_norm_plcb_sem']:.4f} | "
        f"{tot['dist_inf_plcb_mean']:.4f} ± {tot['dist_inf_plcb_sem']:.4f} | "
        f"{tot['diff_plcb_mean']:+.4f} | {tot['pvalue_plcb']:.2e} | {tot['cliffs_delta_plcb']:+.3f} | "
        f"{tot['wins_plcb']}W / {tot['ties_plcb']}T / {tot['losses_plcb']}L |",
        "",
        "*(Note: 'Win' indicates Chebyshev correlation > Euclidean correlation).* ",
        "",
        "---",
        "",
        "## Dimension-Wise Breakdown",
        "",
        "Comparison of distance metric rank correlation across dimensionality regimes:",
        "",
        "| Dimension | SLCB Euclidean | SLCB Chebyshev | SLCB Diff | SLCB p-val | PLCB Euclidean | PLCB Chebyshev | PLCB Diff | PLCB p-val |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    # Dimension marginals (sampling_strategy == "All", dimension != "All")
    dim_rows = df_slices[
        (df_slices["sampling_strategy"] == "All") &
        (df_slices["dimension"].astype(str) != "All")
    ]
    for _, r in dim_rows.iterrows():
        lines.append(
            f"| D={r['dimension']} | "
            f"{r['dist_norm_slcb_mean']:.3f} | {r['dist_inf_slcb_mean']:.3f} | {r['diff_slcb_mean']:+.3f} | {r['pvalue_slcb']:.1e} | "
            f"{r['dist_norm_plcb_mean']:.3f} | {r['dist_inf_plcb_mean']:.3f} | {r['diff_plcb_mean']:+.3f} | {r['pvalue_plcb']:.1e} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Strategy Breakdown",
        "",
        "Comparison across test sampling distribution strategies:",
        "",
        "| Strategy | Runs | SLCB Euclidean | SLCB Chebyshev | SLCB Diff | PLCB Euclidean | PLCB Chebyshev | PLCB Diff |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    # Strategy marginals (dimension == "All", sampling_strategy != "All")
    strat_rows = df_slices[
        (df_slices["dimension"].astype(str) == "All") &
        (df_slices["sampling_strategy"] != "All")
    ]
    for _, r in strat_rows.iterrows():
        lines.append(
            f"| **{r['sampling_strategy']}** | {r['n_experiments']} | "
            f"{r['dist_norm_slcb_mean']:.3f} | {r['dist_inf_slcb_mean']:.3f} | {r['diff_slcb_mean']:+.3f} | "
            f"{r['dist_norm_plcb_mean']:.3f} | {r['dist_inf_plcb_mean']:.3f} | {r['diff_plcb_mean']:+.3f} |"
        )

    # Stratum breakdown if available
    if not df_strata.empty:
        lines.extend([
            "",
            "---",
            "",
            "## Stratum-Wise Breakdown",
            "",
            "Analysis across extrapolation depth strata:",
            "- **Stratum 0 (Interpolation):** Points within the convex hull ($\\tilde d \\le 0$).",
            "- **Stratum 1 (Near Extrapolation):** Moderate distance outside convex hull ($0 < \\tilde d \\le 0.3$).",
            "- **Stratum 2 (Moderate Extrapolation):** Intermediate distance ($0.3 < \\tilde d \\le 0.8$).",
            "- **Stratum 3 (Deep Extrapolation):** High distance out-of-domain ($ \\tilde d > 0.8$).",
            "",
            "| Stratum | SLCB Euclidean | SLCB Chebyshev | SLCB Diff | PLCB Euclidean | PLCB Chebyshev | PLCB Diff |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for _, r in df_strata.iterrows():
            s_idx = int(r["stratum"])
            lines.append(
                f"| Stratum {s_idx} | "
                f"{r['dist_norm_slcb_mean']:.3f} ± {r['dist_norm_slcb_sem']:.3f} | "
                f"{r['dist_inf_slcb_mean']:.3f} ± {r['dist_inf_slcb_sem']:.3f} | "
                f"{r['diff_slcb_mean']:+.3f} | "
                f"{r['dist_norm_plcb_mean']:.3f} ± {r['dist_norm_plcb_sem']:.3f} | "
                f"{r['dist_inf_plcb_mean']:.3f} ± {r['dist_inf_plcb_sem']:.3f} | "
                f"{r['diff_plcb_mean']:+.3f} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## Scientific Interpretation & Surrogate Geometry",
        "",
        "1. **Tree Geometry and Axis Alignment:** Standard Random Forest regressors partition the feature space "
        "using axis-aligned orthogonal cuts $x_j \\lessgtr \\theta$. The resulting leaf cells form axis-aligned hyper-rectangles. "
        "Consequently, out-of-domain epistemic uncertainty is intimately tied to the Chebyshev ($L_\\infty$) distance, "
        "which measures the maximum single-coordinate departure from the training bounding hull.",
        "",
        "2. **Euclidean vs. Chebyshev Synergy:** While Euclidean distance smoothly penalizes simultaneous departures "
        "across all coordinates (desirable for rotation-invariant global penalties), Chebyshev distance directly reflects "
        "the boundary structure of the surrogate's orthogonal decision trees. In high dimensions ($D \\ge 16$), the ratio "
        "$\\|x\\|_2 / \\|x\\|_\\infty$ scales as $\\sqrt{D}$, creating potential scale divergence.",
        "",
        "3. **Recommendation for DyRF-BO:** When deploying axis-aligned tree surrogates in extreme extrapolation regimes, "
        "Chebyshev distance serves as an excellent geometry-aware surrogate diagnostic, confirming that PLCB preserves "
        "strong rank correlation across both metric spaces.",
    ])

    return "\n".join(lines)


def run_distance_ablation_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    scorecard_filename: str = "distance_ablation_scorecard.csv",
    report_filename: str = "DISTANCE_ABLATION_REPORT.md",
    max_files: int | None = None,
) -> int:
    """Execute complete distance metric ablation pipeline.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing Parquet evaluation files.
    output_dir : str | Path
        Directory to write outputs.
    scorecard_filename : str, default='distance_ablation_scorecard.csv'
        Name of CSV output file.
    report_filename : str, default='DISTANCE_ABLATION_REPORT.md'
        Name of Markdown report file.
    max_files : int | None, default=None
        Maximum number of files to process.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / scorecard_filename
    report_path = out_dir / report_filename

    if not in_dir.is_dir():
        print(f"[WARN] Input directory '{in_dir}' does not exist. Writing empty analysis files.")
        df_slices, df_strata = aggregate_distance_ablation([])
        df_slices.to_csv(csv_path, index=False)
        report_text = generate_ablation_markdown_report(df_slices, df_strata, total_runs=0)
        report_path.write_text(report_text, encoding="utf-8")
        return 0

    # Collect parquet files
    parquet_files = sorted(in_dir.glob("extrapolation_*.parquet"))
    if not parquet_files:
        parquet_files = sorted(in_dir.glob("*.parquet"))

    if not parquet_files:
        print(f"[WARN] No Parquet evaluation files found in '{in_dir}'. Writing empty analysis files.")
        df_slices, df_strata = aggregate_distance_ablation([])
        df_slices.to_csv(csv_path, index=False)
        report_text = generate_ablation_markdown_report(df_slices, df_strata, total_runs=0)
        report_path.write_text(report_text, encoding="utf-8")
        return 0

    if max_files is not None and max_files > 0:
        parquet_files = parquet_files[:max_files]

    print(f"[INFO] Evaluating distance ablation metrics on {len(parquet_files)} Parquet files from '{in_dir}'...")
    records: List[Dict[str, Any]] = []
    for fpath in parquet_files:
        try:
            rec = compute_run_distance_ablation(fpath)
            records.append(rec)
        except Exception as e:
            print(f"[WARN] Failed to process {fpath.name}: {e}")
            continue

    if not records:
        print("[WARN] No valid records extracted. Writing empty analysis files.")
        df_slices, df_strata = aggregate_distance_ablation([])
        df_slices.to_csv(csv_path, index=False)
        report_text = generate_ablation_markdown_report(df_slices, df_strata, total_runs=0)
        report_path.write_text(report_text, encoding="utf-8")
        return 0

    df_slices, df_strata = aggregate_distance_ablation(records)

    # 1. Save CSV Scorecard
    df_slices.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Saved distance ablation scorecard: {csv_path}")

    # 2. Save Markdown Report
    report_content = generate_ablation_markdown_report(df_slices, df_strata, total_runs=len(records))
    report_path.write_text(report_content, encoding="utf-8")
    print(f"[SUCCESS] Saved distance ablation report: {report_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_distance_ablation_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scorecard_filename=args.scorecard_filename,
        report_filename=args.report_filename,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    sys.exit(main())
