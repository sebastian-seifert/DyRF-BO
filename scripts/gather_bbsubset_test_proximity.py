#!/usr/bin/env python3
"""CARP-S Data Gathering Script for BBsubset Held-Out Test Proximity LCB Suite.

Collects and normalizes raw execution logs in `runs/sweep_bbsubset_test_proximity`
for `SMAC20_ProximityLCB_tuned` and `SMAC3_HPOFacade_lcb` into parquet and CSV tables
in `results/sweep_bbsubset_test_proximity/logs.parquet`.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional

from carps.analysis.gather_data import filelogs_to_df


def fallback_gather(runs_base: Path, outdir: Path):
    """Fallback parser if CARP-S filelogs_to_df encounters nesting or config errors."""
    import json
    import numpy as np
    import pandas as pd

    trial_log_files = list(runs_base.glob("**/trial_logs.jsonl"))
    if not trial_log_files:
        raise FileNotFoundError(f"No trial_logs.jsonl found in {runs_base}")

    records = []
    for t_file in trial_log_files:
        run_dir = t_file.parent
        hydra_cfg_file = run_dir / ".hydra" / "config.yaml"

        opt_id = "unknown"
        task_id = "unknown"
        seed = 1

        if hydra_cfg_file.exists():
            try:
                import yaml
                with open(hydra_cfg_file, "r") as f:
                    cfg_data = yaml.safe_load(f)
                if isinstance(cfg_data, dict):
                    opt_id = cfg_data.get("optimizer_id", opt_id)
                    task_id = cfg_data.get("task_id", cfg_data.get("task_name", task_id))
                    seed = int(cfg_data.get("seed", seed))
            except Exception:
                pass

        if opt_id == "unknown" or task_id == "unknown":
            parts = run_dir.parts
            for p in parts:
                if "ProximityLCB" in p or "SMAC3_HPOFacade" in p:
                    opt_id = p
                if p.isdigit() and seed == 1:
                    seed = int(p)

        with open(t_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                n_trials = row.get("n_trials", row.get("n_function_calls", 1))
                trial_val = row.get("trial_value", {})
                cost = trial_val.get("cost", np.nan)

                records.append({
                    "task_id": task_id,
                    "optimizer_id": opt_id,
                    "seed": seed,
                    "n_trials": n_trials,
                    "trial_value__cost": cost,
                    "trial_value__cost_inc": cost,
                })

    df = pd.DataFrame(records)
    grouper = ["task_id", "optimizer_id", "seed"]
    df = df.sort_values(by=grouper + ["n_trials"]).reset_index(drop=True)
    df["trial_value__cost_inc"] = df.groupby(grouper)["trial_value__cost"].cummin()

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "logs.csv", index=False)
    df.to_parquet(outdir / "logs.parquet", index=False)

    df_cfg = pd.DataFrame()
    return df, df_cfg


def gather_bbsubset_test_proximity_data(
    runs_base: str = "runs/sweep_bbsubset_test_proximity",
    outdir: str = "results/sweep_bbsubset_test_proximity",
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
        "SMAC20_ProximityLCB_tuned*",
        "SMAC3_HPOFacade_lcb*",
        "*ProximityLCB*",
        "*SMAC3_HPOFacade*",
    ]

    target_rundirs: List[str] = []
    seen = set()
    for pattern in valid_prefix_patterns:
        # Search direct children and 1-2 levels down
        matched = (
            sorted(glob.glob(str(runs_path / pattern)))
            + sorted(glob.glob(str(runs_path / "*" / pattern)))
            + sorted(glob.glob(str(runs_path / "**" / pattern)))
        )
        for m in matched:
            if os.path.isdir(m) and m not in seen:
                seen.add(m)
                target_rundirs.append(m)

    print(f"Found {len(target_rundirs)} target optimizer directories for BBsubset Test Proximity Suite:")
    for d in target_rundirs:
        print(f"  - {d}")

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not target_rundirs:
        trial_logs = list(runs_path.glob("**/trial_logs.jsonl"))
        if trial_logs:
            print("Notice: No pattern-matched directories, but trial_logs.jsonl detected. Running resilient fallback gather...")
            df, df_cfg = fallback_gather(runs_path, out_path)
            print(f"Success! Fallback processed {len(df)} total evaluation rows.")
            print(f"Dataframes saved to: {out_path}")
            return
        print(f"Warning: No matching optimizer directories found in '{runs_base}'.")
        print(f"Searched patterns: {valid_prefix_patterns}")
        return

    print(f"\nGathering and normalizing data into '{outdir}'...")
    try:
        df, df_cfg = filelogs_to_df(rundir=target_rundirs, outdir=outdir)
        print(f"Success! Processed {len(df)} total evaluation rows across {len(df_cfg)} configurations.")
    except Exception as exc:
        print(f"Notice: filelogs_to_df raised ({exc}). Running resilient fallback gather...")
        df, df_cfg = fallback_gather(runs_path, out_path)
        print(f"Success! Fallback processed {len(df)} total evaluation rows.")

    print(f"Dataframes saved to: {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather CARP-S logs for BBsubset Held-Out Test Proximity LCB Benchmark Suite"
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
        default="runs/sweep_bbsubset_test_proximity",
        help="Base runs directory (default: runs/sweep_bbsubset_test_proximity)",
    )
    parser.add_argument(
        "--outdir",
        dest="out_dir",
        type=str,
        default="results/sweep_bbsubset_test_proximity",
        help="Output directory (default: results/sweep_bbsubset_test_proximity)",
    )

    args = parser.parse_args()
    runs_dir = args.runs_dir_pos if args.runs_dir_pos is not None else args.runs_dir
    out_dir = args.out_dir_pos if args.out_dir_pos is not None else args.out_dir

    gather_bbsubset_test_proximity_data(runs_base=runs_dir, outdir=out_dir)


if __name__ == "__main__":
    main()
