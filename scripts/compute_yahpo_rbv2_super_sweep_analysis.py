#!/usr/bin/env python3
"""Statistical Analysis and Scorecard Generation for YAHPO Gym rbv2_super Sweeps.

Computes:
1. Per-task Min-Max scaled regret on YAHPO rbv2_super tasks.
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
    format_dataframe_safe,
    run_bbob_statistical_analysis,
)
from scripts.compute_yahpo_rbv2_ranger_sweep_analysis import (
    export_yahpo_analysis,
)


def run_yahpo_rbv2_super_analysis(
    df: pd.DataFrame,
    cost_col: str = "trial_value__cost_inc",
    controls: Sequence[str] = ("SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"),
    holm_scope: str = "joint",
) -> Dict[str, Any]:
    """Runs statistical analysis on YAHPO rbv2_super benchmark logs."""
    full_results = run_bbob_statistical_analysis(
        df,
        cost_col=cost_col,
        controls=controls,
        holm_scope=holm_scope,
    )
    return {
        "scorecard": full_results["global_scorecard"],
        "friedman_stat": full_results["friedman_stat"],
        "friedman_p": full_results["friedman_p"],
        "raw_analysis": full_results,
    }


def export_yahpo_rbv2_super_analysis(results: Dict[str, Any], output_dir: str | Path) -> None:
    """Exports YAHPO rbv2_super scorecards to CSV and formatted Markdown report."""
    export_yahpo_analysis(results, output_dir, suite="super")


def build_parser(default_suite: str = "super") -> argparse.ArgumentParser:
    """Builds unified CLI argument parser for YAHPO super sweep analysis."""
    parser = argparse.ArgumentParser(
        description=f"Compute statistical analysis for YAHPO rbv2_{default_suite} sweep."
    )
    parser.add_argument(
        "--input",
        "--data-path",
        dest="input",
        type=str,
        default=f"results/sweep_yahpo_rbv2_{default_suite}/logs.parquet",
        help="Path to consolidated logs (.parquet or .csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"results/sweep_yahpo_rbv2_{default_suite}",
        help="Directory to save scorecard CSV and Markdown.",
    )
    parser.add_argument(
        "--cost-col",
        type=str,
        default="trial_value__cost_inc",
        help="Column name representing incumbent loss.",
    )
    parser.add_argument(
        "--holm-scope",
        type=str,
        default="joint",
        choices=["joint", "per_control"],
        help="FWER correction scope across controls (joint or per_control).",
    )
    parser.add_argument(
        "--controls",
        type=str,
        nargs="+",
        default=["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"],
        help="Control optimizer identifiers for paired comparisons.",
    )
    return parser


def main() -> None:
    parser = build_parser(default_suite="super")
    args = parser.parse_args()
    data_path = Path(args.input)
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

    results = run_yahpo_rbv2_super_analysis(
        df,
        cost_col=args.cost_col,
        controls=args.controls,
        holm_scope=args.holm_scope,
    )
    export_yahpo_rbv2_super_analysis(results, args.output_dir)
    print(f"Analysis successfully exported to {args.output_dir}")


if __name__ == "__main__":
    main()
