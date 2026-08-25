#!/usr/bin/env python3
"""Wrapper script to execute CARP-S native data gathering for the Additive Hybrid Sweep.

Processes run directories in `runs/` matching CARPSDynamicRF_AdditiveEpistemic*
and the reference SMAC3_HPOFacade* baselines.
"""

import os
import sys
import glob
from pathlib import Path
from carps.analysis.gather_data import filelogs_to_df

def gather_additive_carps_data(runs_base: str = "runs", outdir: str = "results/bbsubset_additive_analysis"):
    runs_path = Path(runs_base)
    if not runs_path.exists():
        print(f"Error: Base runs directory '{runs_base}' does not exist.")
        sys.exit(1)

    # Filter directories matching our sweep optimizers
    valid_prefix_patterns = [
        "CARPSDynamicRF_AdditiveEpistemic*",
        "SMAC3_HPOFacade*"
    ]

    target_rundirs = []
    for pattern in valid_prefix_patterns:
        matched = sorted(glob.glob(str(runs_path / pattern)))
        target_rundirs.extend([m for m in matched if os.path.isdir(m)])

    print(f"Found {len(target_rundirs)} target optimizer directories for Additive Hybrid Sweep:")
    for d in target_rundirs:
        print(f"  - {d}")

    if not target_rundirs:
        print("No matching optimizer directories found in 'runs/'.")
        print("Note: If runs were written to a custom output directory, specify it as an argument: python3 scripts/gather_bbsubset_additive_carps.py <runs_dir> <out_dir>")
        sys.exit(1)

    print(f"\nGathering and normalizing data into '{outdir}'...")
    df, df_cfg = filelogs_to_df(rundir=target_rundirs, outdir=outdir)
    print(f"Success! Processed {len(df)} total evaluation rows across {len(df_cfg)} configurations.")
    print(f"Dataframes saved to: {outdir}")

if __name__ == "__main__":
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results/bbsubset_additive_analysis"
    gather_additive_carps_data(runs_dir, out_dir)
