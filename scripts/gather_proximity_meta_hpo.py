#!/usr/bin/env python3
"""CARP-S Data Gathering Script for BBsubset Proximity Meta-HPO Sweep.

Collects and aggregates raw execution logs from `runs/sweep_proximity_meta_hpo`
for all 50 Sobol configurations (`SMAC20_ProximityLCB_cfg_01` to `_cfg_50`)
into normalized `logs.parquet` and `logs.csv` files.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def find_target_rundirs(runs_path: Path) -> List[str]:
    """Finds all target optimizer directories or directories containing trial logs."""
    valid_prefix_patterns = [
        "SMAC20_ProximityLCB_cfg_*",
        "*ProximityLCB*",
        "SMAC20_ProximityLCB*",
    ]

    target_rundirs: List[str] = []
    seen = set()

    for pattern in valid_prefix_patterns:
        # Search direct children and subdirectories
        for match in runs_path.glob(pattern):
            if match.is_dir() and str(match) not in seen:
                seen.add(str(match))
                target_rundirs.append(str(match))
        for match in runs_path.glob(f"**/{pattern}"):
            if match.is_dir() and str(match) not in seen:
                seen.add(str(match))
                target_rundirs.append(str(match))

    if not target_rundirs and runs_path.exists():
        # If no pattern matched, check if runs_path directly contains trial logs
        trial_logs = list(runs_path.glob("**/trial_logs.jsonl"))
        if trial_logs:
            target_rundirs.append(str(runs_path))

    return sorted(target_rundirs)


def fallback_gather(runs_base: Path, outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback parser if CARP-S filelogs_to_df encounters nesting or config errors."""
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

        # Fallback to directory structure if unknown
        if opt_id == "unknown" or task_id == "unknown":
            parts = run_dir.parts
            for p in parts:
                if "ProximityLCB" in p:
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
                    "trial_value__cost_inc": cost,  # will cummin
                })

    df = pd.DataFrame(records)
    # Compute incumbent cost
    grouper = ["task_id", "optimizer_id", "seed"]
    df = df.sort_values(by=grouper + ["n_trials"]).reset_index(drop=True)
    df["trial_value__cost_inc"] = df.groupby(grouper)["trial_value__cost"].cummin()

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "logs.csv", index=False)
    df.to_parquet(outdir / "logs.parquet", index=False)

    df_cfg = pd.DataFrame()
    return df, df_cfg


def gather_proximity_meta_data(
    runs_base: str = "runs/sweep_proximity_meta_hpo",
    outdir: str = "results/sweep_proximity_meta_hpo",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collects and aggregates raw CARP-S run logs into structured dataframes.

    Args:
        runs_base: Root directory containing CARP-S execution logs.
        outdir: Destination directory where aggregated parquet and CSV tables are written.

    Returns:
        Tuple of (df, df_cfg).
    """
    runs_path = Path(runs_base)
    if not runs_path.exists():
        if Path("runs").exists() and (Path("runs") / Path(runs_base).name).exists():
            runs_path = Path("runs") / Path(runs_base).name
        elif Path("runs").exists():
            runs_path = Path("runs")
        else:
            raise FileNotFoundError(f"Base runs directory '{runs_base}' does not exist.")

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_rundirs = find_target_rundirs(runs_path)

    print(f"Gathering CARP-S logs from: {runs_path}")
    print(f"Target directories identified: {len(target_rundirs)}")

    if not target_rundirs:
        raise FileNotFoundError(f"No run directories with trial logs found in '{runs_base}'.")

    try:
        from carps.analysis.gather_data import filelogs_to_df
        df, df_cfg = filelogs_to_df(rundir=target_rundirs, outdir=str(out_path))
        print(f"Successfully gathered {len(df)} trials across {len(target_rundirs)} directories.")
    except Exception as exc:
        print(f"Notice: carps filelogs_to_df raised ({exc}). Running resilient fallback gather...")
        df, df_cfg = fallback_gather(runs_path, out_path)
        print(f"Fallback successfully gathered {len(df)} trials.")

    print(f"Dataframes saved to: {out_path / 'logs.parquet'} and {out_path / 'logs.csv'}")
    return df, df_cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather CARP-S logs for Proximity Lower Bound Meta-HPO Sweep."
    )
    parser.add_argument(
        "--runs-dir",
        "-r",
        default="runs/sweep_proximity_meta_hpo",
        help="Base runs directory containing CARP-S outputs (default: runs/sweep_proximity_meta_hpo)",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default="results/sweep_proximity_meta_hpo",
        help="Output directory where logs.parquet will be written (default: results/sweep_proximity_meta_hpo)",
    )
    args = parser.parse_args()

    gather_proximity_meta_data(runs_base=args.runs_dir, outdir=args.outdir)


if __name__ == "__main__":
    main()
