#!/usr/bin/env python3
"""Statistical Analysis and Scorecard Generation for HPOBench Tabular ML Sweeps.

Computes:
1. Per-task Min-Max scaled regret on HPOBench Tabular ML tasks.
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
from typing import Any, Dict, Sequence
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.compute_bbob_full_sweep_analysis import (
    run_bbob_statistical_analysis,
)


def run_hpobench_ml_analysis(
    df: pd.DataFrame,
    cost_col: str = "trial_value__cost_inc",
    controls: Sequence[str] = ("SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"),
) -> Dict[str, Any]:
    """Runs statistical analysis on HPOBench Tabular ML benchmark logs."""
    full_results = run_bbob_statistical_analysis(df, cost_col=cost_col, controls=controls)
    return {
        "scorecard": full_results["global_scorecard"],
        "friedman_stat": full_results["friedman_stat"],
        "friedman_p": full_results["friedman_p"],
        "raw_analysis": full_results,
    }


def export_hpobench_ml_analysis(results: Dict[str, Any], output_dir: str) -> None:
    """Exports HPOBench Tabular ML scorecards to CSV and formatted Markdown report."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scorecard = results["scorecard"]
    scorecard.to_csv(out_path / "hpobench_ml_scorecard.csv", index=False)

    try:
        table_md = scorecard.to_markdown(index=False)
    except Exception:
        table_md = scorecard.to_string()

    stat_val = f"{results['friedman_stat']:.4f}" if results.get("friedman_stat") is not None else "N/A"
    p_val = f"{results['friedman_p']:.4e}" if results.get("friedman_p") is not None else "N/A"

    report_lines = [
        "# HPOBench Tabular ML Benchmark Sweep Scorecard",
        "",
        "## Omnibus Statistical Test (Demšar)",
        f"- **Friedman $\\chi^2$ Statistic**: {stat_val}",
        f"- **p-value**: {p_val}",
        "",
        "## Method Comparison Scorecard",
        table_md,
        "",
    ]
    with open(out_path / "hpobench_ml_scorecard.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute statistical analysis for HPOBench Tabular ML sweep."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="results/sweep_hpobench_ml/logs.parquet",
        help="Path to consolidated logs (.parquet or .csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sweep_hpobench_ml",
        help="Directory to save scorecard CSV and Markdown.",
    )
    parser.add_argument(
        "--cost-col",
        type=str,
        default="trial_value__cost_inc",
        help="Column name representing incumbent loss.",
    )

    args = parser.parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        fallback_csv = data_path.with_suffix(".csv")
        if fallback_csv.exists():
            data_path = fallback_csv
        else:
            raise FileNotFoundError(f"Logs file not found: {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    results = run_hpobench_ml_analysis(df, cost_col=args.cost_col)
    export_hpobench_ml_analysis(results, args.output_dir)
    print(f"Analysis successfully exported to {args.output_dir}")


if __name__ == "__main__":
    main()
