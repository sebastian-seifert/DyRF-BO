#!/usr/bin/env python3
"""CARP-S Data Gathering Script for BBSubset Held-Out Test Big Comparison Suite.

Collects and normalizes raw execution logs in `runs/sweep_bbsubset_test_big_comparison`
across all 11 evaluated optimizers:
- SMAC3_HPOFacade_lcb
- SMAC3_HPOFacade_ei
- SMAC20_Entropy_LCB
- SMAC20_ProximityA_LCB
- SMAC20_ProximityA_LCB_CV
- SMAC20_ProximityB_LCB
- SMAC20_ProximityB_LCB_CV
- SMAC20_ProximityAC_LCB
- SMAC20_ProximityAC_LCB_CV
- SMAC20_ProximityBC_LCB
- SMAC20_ProximityBC_LCB_CV

Aggregates all trial_logs.jsonl and telemetry JSON files into structured tables
in `results/sweep_bbsubset_test_big_comparison/logs.parquet` and `logs.csv`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


VALID_OPTIMIZER_IDS = [
    "SMAC3_HPOFacade_lcb",
    "SMAC3_HPOFacade_ei",
    "SMAC20_Entropy_LCB",
    "SMAC20_ProximityA_LCB",
    "SMAC20_ProximityA_LCB_CV",
    "SMAC20_ProximityB_LCB",
    "SMAC20_ProximityB_LCB_CV",
    "SMAC20_ProximityAC_LCB",
    "SMAC20_ProximityAC_LCB_CV",
    "SMAC20_ProximityBC_LCB",
    "SMAC20_ProximityBC_LCB_CV",
]


def gather_trial_logs(runs_base: Path, outdir: Path) -> pd.DataFrame:
    """Recursively parses all trial_logs.jsonl and telemetry files across the run directory."""
    trial_log_files = list(runs_base.glob("**/trial_logs.jsonl"))
    records = []

    if trial_log_files:
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
                        task_id = cfg_data.get("task", {}).get("name", task_id)
                        seed = cfg_data.get("seed", seed)
                except Exception:
                    pass

            # Fallback path inference if hydra config is missing
            if opt_id == "unknown" or task_id == "unknown":
                parts = run_dir.parts
                for p in parts:
                    for valid_id in VALID_OPTIMIZER_IDS:
                        if valid_id in p:
                            opt_id = valid_id
                            break
                    if "subset_" in p:
                        task_id = p
                    if p.isdigit() and seed == 1:
                        seed = int(p)

            with open(t_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    n_trials = row.get("n_trials", row.get("n_function_calls", row.get("trial", 1)))
                    trial_val = row.get("trial_value", {})
                    cost = trial_val.get("cost", np.nan)

                    records.append({
                        "task_id": task_id,
                        "optimizer_id": opt_id,
                        "seed": seed,
                        "n_trials": n_trials,
                        "trial_value__cost": cost,
                    })

    # Also check telemetry files in results directory
    telemetry_files = list(outdir.glob("telemetry_*.json"))
    if not records and telemetry_files:
        print(f"[INFO] Parsing {len(telemetry_files)} telemetry JSON files...")
        for telem_file in telemetry_files:
            try:
                with open(telem_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                opt_id = data.get("optimizer_id", "unknown")
                task_id = data.get("task_id", "unknown")
                seed = data.get("seed", 1)
                for tr in data.get("trials", []):
                    records.append({
                        "task_id": task_id,
                        "optimizer_id": opt_id,
                        "seed": seed,
                        "n_trials": tr.get("trial", 1),
                        "trial_value__cost": tr.get("cost", np.nan),
                    })
            except Exception:
                continue

    if not records:
        print(f"[WARN] No execution records found in {runs_base} or {outdir}.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    grouper = ["task_id", "optimizer_id", "seed"]
    df = df.sort_values(by=grouper + ["n_trials"]).reset_index(drop=True)
    df["trial_value__cost_inc"] = df.groupby(grouper)["trial_value__cost"].cummin()

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "logs.csv", index=False)
    df.to_parquet(outdir / "logs.parquet", index=False)
    print(f"[SUCCESS] Saved {len(df)} records across {df['optimizer_id'].nunique()} optimizers to {outdir / 'logs.parquet'}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Gather logs for CARP-S BBSubset Test Big Comparison Sweep."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs/sweep_bbsubset_test_big_comparison",
        help="Base runs directory containing CARP-S output folders.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/sweep_bbsubset_test_big_comparison",
        help="Destination directory for logs.parquet and logs.csv.",
    )
    args = parser.parse_args()

    gather_trial_logs(Path(args.runs_dir), Path(args.outdir))


if __name__ == "__main__":
    main()
