#!/usr/bin/env python3
"""CARP-S Data Gathering Script for BBsubset Held-Out Test Suite (Proximity LCB vs SMAC3 EI).

Collects and normalizes raw execution logs in `runs/sweep_bbsubset_test_proximity_vs_ei`
for `SMAC20_ProximityLCB_tuned` and `SMAC3_HPOFacade_ei` into parquet and CSV tables
in `results/sweep_bbsubset_test_proximity_vs_ei/logs.parquet` and `logs.csv`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def gather_bbsubset_test_proximity_vs_ei_logs(
    runs_dir: str = "runs/sweep_bbsubset_test_proximity_vs_ei",
    output_dir: str = "results/sweep_bbsubset_test_proximity_vs_ei",
) -> pd.DataFrame:
    """Gathers trial logs from all BBsubset test runs into a consolidated DataFrame."""
    runs_path = Path(runs_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    trial_log_files = list(runs_path.glob("**/trial_logs.jsonl"))
    if not trial_log_files:
        print(f"[WARNING] No trial_logs.jsonl found in {runs_dir}")
        return pd.DataFrame()

    print(f"Found {len(trial_log_files)} trial log files in {runs_dir}. Parsing...")

    records = []
    for t_file in trial_log_files:
        run_dir = t_file.parent
        hydra_cfg_file = run_dir / ".hydra" / "config.yaml"

        opt_id = "unknown"
        task_id = "unknown"
        seed = 1

        if hydra_cfg_file.exists():
            # 1. Attempt OmegaConf to resolve ${...} interpolations
            try:
                from omegaconf import OmegaConf
                cfg = OmegaConf.load(hydra_cfg_file)
                if "optimizer_id" in cfg and cfg.optimizer_id is not None:
                    opt_id = str(cfg.optimizer_id)
                if "task_id" in cfg and cfg.task_id is not None:
                    task_id = str(cfg.task_id)
                elif "task" in cfg and hasattr(cfg.task, "name") and cfg.task.name:
                    task_id = str(cfg.task.name)
                elif "task_name" in cfg and cfg.task_name is not None:
                    task_id = str(cfg.task_name)
                if "seed" in cfg and cfg.seed is not None:
                    seed = int(cfg.seed)
            except Exception:
                pass

            # 2. PyYAML fallback if OmegaConf unresolved
            if opt_id == "unknown" or task_id == "unknown" or task_id.startswith("${"):
                try:
                    import yaml
                    with open(hydra_cfg_file, "r") as f:
                        cfg_data = yaml.safe_load(f)
                    if isinstance(cfg_data, dict):
                        opt_id = cfg_data.get("optimizer_id", opt_id)
                        t_cand = cfg_data.get("task_id", cfg_data.get("task_name", task_id))
                        if not str(t_cand).startswith("${"):
                            task_id = t_cand
                        elif "task" in cfg_data and isinstance(cfg_data["task"], dict):
                            sub_name = cfg_data["task"].get("name")
                            if sub_name and not str(sub_name).startswith("${"):
                                task_id = sub_name
                        seed = int(cfg_data.get("seed", seed))
                except Exception:
                    pass

        # 3. Path heuristic fallback if still unresolved
        if opt_id == "unknown" or task_id == "unknown" or task_id.startswith("${"):
            for p in run_dir.parts:
                if "ProximityLCB" in p:
                    opt_id = "SMAC20_ProximityLCB_tuned"
                elif "HPOFacade_ei" in p or "SMAC3_HPOFacade" in p:
                    opt_id = "SMAC3_HPOFacade_ei"
                if p.startswith("subset_"):
                    task_id = p
                if p.isdigit() and len(p) <= 2:
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
    if df.empty:
        return df

    grouper = ["task_id", "optimizer_id", "seed"]
    df = df.sort_values(by=grouper + ["n_trials"]).reset_index(drop=True)
    df["trial_value__cost_inc"] = df.groupby(grouper)["trial_value__cost"].cummin()

    parquet_path = out_path / "logs.parquet"
    csv_path = out_path / "logs.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"Aggregated {len(df)} total rows across {df['task_id'].nunique()} tasks.")
    print(f"Saved to:\n  - {parquet_path}\n  - {csv_path}")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gather CARP-S logs for BBsubset Test Proximity vs EI.")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs/sweep_bbsubset_test_proximity_vs_ei",
        help="Root path containing raw run outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sweep_bbsubset_test_proximity_vs_ei",
        help="Directory to save aggregated parquet and CSV logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gather_bbsubset_test_proximity_vs_ei_logs(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
