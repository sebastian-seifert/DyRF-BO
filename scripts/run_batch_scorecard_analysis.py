#!/usr/bin/env python3
"""Batch Runner for Sweep Scorecard Analysis.

Executes compute_sweep_scorecard over one or multiple benchmark log files (Parquet or CSV).
For each input file, outputs are placed into a subfolder named `new_table` under the
parent directory of that input file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compute_sweep_scorecard import (
    DEFAULT_CONTROLS,
    DEFAULT_COST_COL,
    DEFAULT_TIE_TOLERANCE,
    compute_sweep_scorecard,
    export_scorecard,
    load_logs_file,
)

logger = logging.getLogger("batch_scorecard")


def resolve_batch_output_dir(input_path: str | Path, subfolder: str = "new_table") -> Path:
    """Resolves output directory to `<input_parent>/<subfolder>`."""
    resolved_path = Path(input_path).resolve()
    return resolved_path.parent / subfolder


def process_single_log_file(
    input_path: str | Path,
    subfolder: str = "new_table",
    controls: Sequence[str] = DEFAULT_CONTROLS,
    cost_col: str = DEFAULT_COST_COL,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    holm_scope: str = "per_control",
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Processes a single log file and writes scorecard outputs to `<input_dir>/<subfolder>`."""
    file_path = Path(input_path).resolve()
    if not file_path.exists():
        msg = f"Input file not found: {file_path}"
        logger.error(msg)
        return {"status": "error", "path": str(file_path), "error": msg}

    output_dir = resolve_batch_output_dir(file_path, subfolder=subfolder)
    output_dir.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = f"Scorecard: {file_path.parent.name}"

    try:
        logger.info(f"Loading data from {file_path} ...")
        df = load_logs_file(file_path)

        logger.info(f"Computing scorecard for {file_path} ...")
        results = compute_sweep_scorecard(
            df=df,
            controls=controls,
            cost_col=cost_col,
            tie_tolerance=tie_tolerance,
            holm_scope=holm_scope,
        )

        csv_path, md_path = export_scorecard(
            results=results,
            output_dir=output_dir,
            title=title,
            controls=controls,
        )
        logger.info(f"Exported scorecard to {csv_path} and {md_path}")

        return {
            "status": "success",
            "path": str(file_path),
            "output_dir": str(output_dir),
            "csv_path": str(csv_path),
            "md_path": str(md_path),
            "results": results,
        }
    except Exception as e:
        logger.exception(f"Failed processing {file_path}: {e}")
        return {"status": "error", "path": str(file_path), "error": str(e)}


def run_batch_scorecards(
    input_paths: Sequence[str | Path],
    subfolder: str = "new_table",
    controls: Sequence[str] = DEFAULT_CONTROLS,
    cost_col: str = DEFAULT_COST_COL,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    holm_scope: str = "per_control",
) -> List[Dict[str, Any]]:
    """Runs scorecard analysis across all provided input file paths."""
    outcomes = []
    for p in input_paths:
        outcome = process_single_log_file(
            input_path=p,
            subfolder=subfolder,
            controls=controls,
            cost_col=cost_col,
            tie_tolerance=tie_tolerance,
            holm_scope=holm_scope,
        )
        outcomes.append(outcome)
    return outcomes


def build_batch_arg_parser() -> argparse.ArgumentParser:
    """Builds argument parser for the batch scorecard runner."""
    parser = argparse.ArgumentParser(
        description="Batch Runner for Sweep Scorecard Analysis across multiple log files."
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="One or more paths to logs.parquet or logs.csv files.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="new_table",
        help="Name of the output subfolder inside each input file's parent directory (default: 'new_table').",
    )
    parser.add_argument(
        "--controls",
        nargs="+",
        default=DEFAULT_CONTROLS,
        help=f"Baseline optimizer IDs to use as reference controls (default: {DEFAULT_CONTROLS}).",
    )
    parser.add_argument(
        "--cost-col",
        type=str,
        default=DEFAULT_COST_COL,
        help=f"Column representing incumbent loss/cost (default: {DEFAULT_COST_COL}).",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=DEFAULT_TIE_TOLERANCE,
        help=f"Numerical threshold to filter out exact ties (default: {DEFAULT_TIE_TOLERANCE}).",
    )
    parser.add_argument(
        "--holm-scope",
        type=str,
        choices=["per_control", "joint"],
        default="per_control",
        help="Multiple testing correction scope ('per_control' or 'joint').",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for batch scorecard runner."""
    parser = build_batch_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    logger.info(f"Starting batch scorecard execution for {len(args.input)} input file(s)...")
    outcomes = run_batch_scorecards(
        input_paths=args.input,
        subfolder=args.subfolder,
        controls=args.controls,
        cost_col=args.cost_col,
        tie_tolerance=args.tie_tolerance,
        holm_scope=args.holm_scope,
    )

    success_count = sum(1 for o in outcomes if o.get("status") == "success")
    error_count = len(outcomes) - success_count
    logger.info(f"Batch processing completed: {success_count} succeeded, {error_count} failed.")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

