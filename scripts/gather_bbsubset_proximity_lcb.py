#!/usr/bin/env python3
"""CARP-S Data Gathering Script for BBsubset Proximity LCB Benchmark Suite.

Collects and normalizes raw execution logs in `runs/bbsubset_proximity_lcb`
for `SMAC20_ProximityLCB_k10` and `SMAC3_HPOFacade_lcb` into parquet and CSV tables.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from carps.analysis.gather_data import filelogs_to_df


def gather_bbsubset_proximity_lcb_data(
    runs_base: str = "runs/bbsubset_proximity_lcb",
    outdir: str = "results/bbsubset_proximity_lcb_analysis",
) -> None:
    """Collects and aggregates raw CARP-S run logs into structured dataframes.

    Args:
        runs_base: Root directory containing CARP-S execution logs.
        outdir: Destination directory where aggregated parquet and CSV tables are written.
    """
    runs_path = Path(runs_base)
    if not runs_path.exists():
        # Fallback check under "runs"
        if Path("runs").exists():
            runs_path = Path("runs")
        else:
            print(f"Error: Base runs directory '{runs_base}' does not exist.")
            sys.exit(1)

    valid_prefix_patterns = [
        "SMAC20_ProximityLCB*",
        "SMAC3_HPOFacade_lcb*",
        "*ProximityLCB*",
    ]

    target_rundirs = []
    seen = set()
    for pattern in valid_prefix_patterns:
        # Search direct children and 1 level down
        matched = sorted(glob.glob(str(runs_path / pattern))) + sorted(glob.glob(str(runs_path / "**" / pattern)))
        for m in matched:
            if os.path.isdir(m) and m not in seen:
                seen.add(m)
                target_rundirs.append(m)

    print(f"Found {len(target_rundirs)} target optimizer directories for Proximity LCB Suite:")
    for d in target_rundirs:
        print(f"  - {d}")

    if not target_rundirs:
        print(f"Warning: No matching optimizer directories found in '{runs_base}'.")
        print(f"Searched patterns: {valid_prefix_patterns}")
        return

    print(f"\nGathering and normalizing data into '{outdir}'...")
    os.makedirs(outdir, exist_ok=True)
    df, df_cfg = filelogs_to_df(rundir=target_rundirs, outdir=outdir)
    print(f"Success! Processed {len(df)} total evaluation rows across {len(df_cfg)} configurations.")
    print(f"Dataframes saved to: {outdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Gather CARP-S logs for BBsubset Proximity LCB Benchmark Suite"
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
        dest="runs_dir",
        type=str,
        default="runs/bbsubset_proximity_lcb",
        help="Base runs directory (default: runs/bbsubset_proximity_lcb)",
    )
    parser.add_argument(
        "--outdir",
        dest="out_dir",
        type=str,
        default="results/bbsubset_proximity_lcb_analysis",
        help="Output directory (default: results/bbsubset_proximity_lcb_analysis)",
    )

    args = parser.parse_args()
    runs_dir = args.runs_dir_pos if args.runs_dir_pos else args.runs_dir
    out_dir = args.out_dir_pos if args.out_dir_pos else args.out_dir
    gather_bbsubset_proximity_lcb_data(runs_dir, out_dir)


if __name__ == "__main__":
    main()
