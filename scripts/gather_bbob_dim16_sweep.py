#!/usr/bin/env python3
"""Data Gathering and Aggregation Script for BBOB Dimension 16 Benchmark Sweep.

Recursively gathers trial execution logs (`trial_logs.jsonl`) and fallback telemetry
files from `runs/sweep_bbob_dim16`, enriching every row with explicit metadata
and enforcing filtering to dimension 16.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.gather_bbob_full_sweep import gather_bbob_trial_logs

BBOB_DIM16_RUNS_BASE = "runs/sweep_bbob_dim16"
BBOB_DIM16_OUTDIR = "results/sweep_bbob_dim16"


def gather_bbob_dim16_trial_logs(
    runs_base: Path | str = BBOB_DIM16_RUNS_BASE,
    outdir: Path | str = BBOB_DIM16_OUTDIR,
) -> pd.DataFrame:
    """Gathers trial logs from runs_base, filtering strictly to dimension 16."""
    df = gather_bbob_trial_logs(runs_base=runs_base, outdir=outdir)
    if not df.empty and "dimension" in df.columns:
        df = df[df["dimension"] == 16].copy()

        # Re-save filtered logs to parquet and csv
        out_path = Path(outdir)
        parquet_path = out_path / "logs.parquet"
        csv_path = out_path / "logs.csv"
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception:
            pass
        df.to_csv(csv_path, index=False)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather BBOB Dimension 16 sweep execution logs.")
    parser.add_argument("--runs-base", type=str, default=BBOB_DIM16_RUNS_BASE)
    parser.add_argument("--outdir", type=str, default=BBOB_DIM16_OUTDIR)
    args = parser.parse_args()

    gather_bbob_dim16_trial_logs(runs_base=args.runs_base, outdir=args.outdir)
