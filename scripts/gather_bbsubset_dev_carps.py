#!/usr/bin/env python3
"""Wrapper script to execute CARP-S data gathering strictly for the BBsubset Dev Sweep.

Filters run directories in `runs/` to include ONLY the 9 optimizers and 20 `dev` tasks
of the master CARP-S BBsubset sweep, ignoring unrelated legacy runs.
"""

import os
import sys
import glob
from pathlib import Path
from carps.analysis.gather_data import filelogs_to_df

def gather_bbsubset_dev_data(
    runs_base: str = "runs",
    outdir: str = "results/bbsubset_dev_analysis",
) -> None:
    """Collects and aggregates raw CARP-S run logs into structured dataframes.

    Filters run directories in `runs_base` matching the custom uncertainty and
    baseline optimizer prefix patterns, exporting normalized data to `outdir`.

    Args:
        runs_base: Path to the root directory containing CARP-S run outputs.
        outdir: Output directory where aggregated CSV/parquet logs will be saved.
    """
    runs_path = Path(runs_base)
    if not runs_path.exists():
        print(f"Error: Base runs directory '{runs_base}' does not exist.")
        sys.exit(1)

    # Filter directories matching our sweep optimizers
    # Custom Uncertainty extractors & SMAC3 HPOFacade baseline
    valid_prefix_patterns = [
        "SMAC3_HPOFacade*",
        "SMAC20_CustomUncertainty*"
    ]

    target_rundirs = []
    for pattern in valid_prefix_patterns:
        matched = sorted(glob.glob(str(runs_path / pattern)))
        target_rundirs.extend([m for m in matched if os.path.isdir(m)])

    print(f"Found {len(target_rundirs)} target optimizer directories for BBsubset Dev Sweep:")
    for d in target_rundirs:
        print(f"  - {d}")

    if not target_rundirs:
        print("No matching optimizer directories found for BBsubset Dev Sweep.")
        sys.exit(1)

    print(f"\nGathering and normalizing data into '{outdir}'...")
    df, df_cfg = filelogs_to_df(rundir=target_rundirs, outdir=outdir)
    print(f"Success! Processed {len(df)} total evaluation rows across {len(df_cfg)} configurations.")

if __name__ == "__main__":
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results/bbsubset_dev_analysis"
    gather_bbsubset_dev_data(runs_dir, out_dir)
