#!/usr/bin/env python3
"""
Generate CARPS reports for a specific subset of optimizers/approaches.

Can filter `logs.parquet` by exact optimizer IDs or by flexible patterns (e.g. 'shaker_entropy', 'proximity_bc').
Also provides an `--inspect` flag to inspect available optimizers in any parquet file.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


def resolve_optimizer_column(df: pd.DataFrame) -> str:
    """Identify which column contains the optimizer identifier."""
    if "optimizer_id" in df.columns:
        return "optimizer_id"
    if "optimizer" in df.columns:
        return "optimizer"
    raise KeyError("Could not find 'optimizer_id' or 'optimizer' column in DataFrame.")


def filter_logs_dataframe(
    df: pd.DataFrame,
    optimizers: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter logs DataFrame to only include selected optimizers or matching patterns."""
    opt_col = resolve_optimizer_column(df)
    unique_opts = df[opt_col].unique().tolist()

    selected_set = set()

    if optimizers:
        for opt in optimizers:
            if opt in unique_opts:
                selected_set.add(opt)

    if patterns:
        for pat in patterns:
            for opt in unique_opts:
                if re.search(pat, str(opt), re.IGNORECASE):
                    selected_set.add(opt)

    if not selected_set:
        raise ValueError(
            f"No matching optimizers found!\n"
            f"Requested exact: {optimizers}\n"
            f"Requested patterns: {patterns}\n"
            f"Available in logs: {unique_opts}"
        )

    filtered_df = df[df[opt_col].isin(selected_set)].copy()
    return filtered_df


def inspect_optimizers(parquet_path: Path) -> List[str]:
    """Inspect and return unique optimizer names in a parquet file."""
    df_sample = pd.read_parquet(parquet_path)
    opt_col = resolve_optimizer_column(df_sample)
    opts = sorted(df_sample[opt_col].unique().tolist())
    print(f"\nAvailable optimizers in {parquet_path} (column '{opt_col}', total {len(opts)}):")
    for opt in opts:
        print(f"  - {opt}")
    return opts


def main():
    parser = argparse.ArgumentParser(
        description="Filter CARPS logs.parquet and generate a focused subset report."
    )
    parser.add_argument(
        "result_path",
        type=str,
        help="Path to logs.parquet file.",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=None,
        help="Exact optimizer_id names to include.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        help="Substring or regex patterns to match optimizer names (e.g. 'shaker_entropy' 'proximity_bc' 'HPOFacade').",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory to save the generated report. Defaults to '<result_path_dir>/reports_subset'.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="subset_report",
        help="Subfolder name for the report.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable results normalization in generate_report.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="List all available optimizers in the parquet file and exit without generating a report.",
    )
    args = parser.parse_args()

    parquet_path = Path(args.result_path)
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}", file=sys.stderr)
        sys.exit(1)

    if args.inspect:
        inspect_optimizers(parquet_path)
        sys.exit(0)

    if not args.optimizers and not args.patterns:
        print(
            "Error: Please specify either --optimizers or --patterns to filter (or use --inspect to view options).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    opt_col = resolve_optimizer_column(df)
    filtered_df = filter_logs_dataframe(df, optimizers=args.optimizers, patterns=args.patterns)
    matched_opts = filtered_df[opt_col].unique().tolist()
    print(f"Matched {len(matched_opts)} optimizers ({len(filtered_df)} / {len(df)} rows):")
    for opt in matched_opts:
        print(f"  * {opt}")

    report_dir = Path(args.report_dir) if args.report_dir else parquet_path.parent / "reports_subset"
    report_dir.mkdir(parents=True, exist_ok=True)

    filtered_parquet_path = report_dir / f"{args.report_name}_filtered_logs.parquet"
    print(f"Writing filtered dataset to {filtered_parquet_path}...")
    filtered_df.to_parquet(filtered_parquet_path, index=False)

    print("\nRunning CARPS generate_report...")
    from carps.analysis.generate_report import generate_report

    generate_report(
        result_path=str(filtered_parquet_path),
        report_dir=report_dir,
        report_name=args.report_name,
        normalize_results=not args.no_normalize,
    )
    print(f"\nReport successfully generated in: {report_dir / args.report_name}")


if __name__ == "__main__":
    main()
