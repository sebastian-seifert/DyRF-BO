#!/usr/bin/env python3
"""CARP-S Data Gathering Script for Real-World ML 30-Seed Benchmark Suite.

Processes execution directories in `runs/` matching CARPSDynamicRF_DAEHRF_AdditiveEI*
and reference SMAC3_HPOFacade_ei* baselines into structured parquet and CSV tables.
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
from pathlib import Path
from carps.analysis.gather_data import filelogs_to_df


def gather_bbsubset_realworld_30seeds_data(
    runs_base: str = "runs",
    outdir: str = "results/bbsubset_realworld_30seeds_analysis",
) -> None:
    """Collects and normalizes CARP-S 30-seed real-world ML benchmark run logs.

    Args:
        runs_base: Base directory where CARP-S execution logs are stored.
        outdir: Destination directory for the generated CSV/parquet tables.
    """
    runs_path = Path(runs_base)
    if not runs_path.exists():
        print(f"Error: Base runs directory '{runs_base}' does not exist.")
        sys.exit(1)

    # Filter directories matching the benchmark optimizers
    valid_prefix_patterns = [
        "CARPSDynamicRF_DAEHRF_AdditiveEI*",
        "SMAC3_HPOFacade_ei*",
        "SMAC3_HPOFacade*",
    ]

    target_rundirs = []
    seen = set()
    for pattern in valid_prefix_patterns:
        matched = sorted(glob.glob(str(runs_path / pattern)))
        for m in matched:
            if os.path.isdir(m) and m not in seen:
                seen.add(m)
                target_rundirs.append(m)

    print(f"Found {len(target_rundirs)} target optimizer directories for Real-World ML 30-Seed Suite:")
    for d in target_rundirs:
        print(f"  - {d}")

    if not target_rundirs:
        print("No matching optimizer directories found in runs directory.")
        print(f"Searched patterns: {valid_prefix_patterns} in '{runs_base}'")
        sys.exit(1)

    print(f"\nGathering and normalizing data into '{outdir}'...")
    os.makedirs(outdir, exist_ok=True)
    df, df_cfg = filelogs_to_df(rundir=target_rundirs, outdir=outdir)
    print(f"Success! Processed {len(df)} total evaluation rows across {len(df_cfg)} configurations.")
    print(f"Dataframes saved to: {outdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Gather CARP-S logs for 30-Seed Real-World ML Benchmark Suite"
    )
    parser.add_argument(
        "runs_dir_pos",
        nargs="?",
        default=None,
        help="Positional argument for runs directory (optional)",
    )
    parser.add_argument(
        "out_dir_pos",
        nargs="?",
        default=None,
        help="Positional argument for output directory (optional)",
    )
    parser.add_argument(
        "--runs-dir",
        "--runs_dir",
        dest="runs_dir",
        type=str,
        default="runs",
        help="Base runs directory (default: runs)",
    )
    parser.add_argument(
        "--outdir",
        "--out_dir",
        dest="out_dir",
        type=str,
        default="results/bbsubset_realworld_30seeds_analysis",
        help="Output directory (default: results/bbsubset_realworld_30seeds_analysis)",
    )

    args = parser.parse_args()
    runs_dir = args.runs_dir_pos if args.runs_dir_pos else args.runs_dir
    out_dir = args.out_dir_pos if args.out_dir_pos else args.out_dir

    gather_bbsubset_realworld_30seeds_data(runs_base=runs_dir, outdir=out_dir)


if __name__ == "__main__":
    main()
