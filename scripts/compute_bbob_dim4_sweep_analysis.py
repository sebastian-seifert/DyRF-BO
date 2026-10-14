#!/usr/bin/env python3
"""Statistical Analysis and Scorecard Generation for BBOB Dimension 4 Sweeps.

Computes:
1. Per-task Min-Max scaled regret on 4D tasks across all evaluated approaches.
2. Demšar omnibus Friedman test.
3. Paired Wilcoxon signed-rank tests vs Control 1 (LCB) and Control 2 (EI).
4. Multiplicity correction via step-down Holm-Bonferroni method.
5. Non-parametric Cliff's Delta effect size and Win/Loss/Tie counts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.compute_bbob_full_sweep_analysis import (
    run_bbob_statistical_analysis,
    export_analysis_results,
)


def run_bbob_dim4_analysis(
    df: pd.DataFrame,
    cost_col: str = "trial_value__cost_inc",
    controls: Sequence[str] = ("SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"),
) -> Dict[str, Any]:
    """Runs statistical analysis filtered strictly to dimension 4 tasks."""
    if "dimension" in df.columns:
        df_4d = df[df["dimension"] == 4].copy()
    else:
        df_4d = df.copy()

    full_results = run_bbob_statistical_analysis(df_4d, cost_col=cost_col, controls=controls)
    return {
        "scorecard": full_results["global_scorecard"],
        "friedman_stat": full_results["friedman_stat"],
        "friedman_p": full_results["friedman_p"],
        "raw_analysis": full_results,
    }


def export_dim4_analysis(results: Dict[str, Any], output_dir: str) -> None:
    """Exports Dimension 4 scorecards to CSV and formatted Markdown report."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scorecard = results["scorecard"]
    scorecard.to_csv(out_path / "bbob_dim4_scorecard.csv", index=False)

    try:
        table_md = scorecard.to_markdown(index=False)
    except Exception:
        # Fallback to string representation if tabulate is not installed
        table_md = scorecard.to_string(index=False)

    md_report = [
        "# CARP-S BBOB Dimension 4 Benchmark Scorecard & Analysis",
        "",
        f"**Omnibus Friedman Test**: $\\chi^2 = {results['friedman_stat']}$, $p = {results['friedman_p']}$",
        "",
        "## 4D Global Scorecard",
        "",
        table_md,
        "",
    ]
    report_path = out_path / "bbob_dim4_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_report))
    print(f"[SUCCESS] Exported Dimension 4 report and scorecard to {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BBOB Dimension 4 statistical analysis.")
    parser.add_argument("--input", type=str, default="results/sweep_bbob_dim4/logs.parquet")
    parser.add_argument("--output-dir", type=str, default="results/sweep_bbob_dim4/analysis")
    parser.add_argument("--cost-col", type=str, default="trial_value__cost_inc")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        alt_path = in_path.with_suffix(".csv") if in_path.suffix == ".parquet" else in_path.with_suffix(".parquet")
        if alt_path.exists():
            in_path = alt_path
        else:
            raise FileNotFoundError(f"Input file {args.input} does not exist.")

    data = None
    if in_path.suffix == ".parquet":
        try:
            data = pd.read_parquet(in_path)
        except Exception:
            csv_path = in_path.with_suffix(".csv")
            if csv_path.exists():
                data = pd.read_csv(csv_path)
            else:
                raise
    if data is None:
        data = pd.read_csv(in_path)

    res = run_bbob_dim4_analysis(data, cost_col=args.cost_col)
    export_dim4_analysis(res, args.output_dir)
