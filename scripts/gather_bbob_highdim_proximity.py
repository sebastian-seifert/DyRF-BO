#!/usr/bin/env python3
"""Data Gathering Script for BBOB High-D & Extreme Proximity LCB Sweep.

Scans `runs/sweep_bbob_highdim_proximity` and consolidates trial logs into
`results/sweep_bbob_highdim_proximity/logs.parquet` and `logs.csv`.
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


def gather_bbob_highdim_logs(
    runs_dir: str = "runs/sweep_bbob_highdim_proximity",
    output_dir: str = "results/sweep_bbob_highdim_proximity",
) -> pd.DataFrame:
    """Gathers trial logs from all BBOB high-d runs into a consolidated DataFrame."""
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
            # 1. Attempt OmegaConf to resolve ${...} interpolations (e.g. task_id: ${task.name})
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

            # 2. PyYAML fallback if OmegaConf unavailable or task_id is still unresolved macro
            if opt_id == "unknown" or task_id == "unknown" or task_id.startswith("${"):
                try:
                    import yaml
                    with open(hydra_cfg_file, "r") as f:
                        cfg_data = yaml.safe_load(f)
                    if isinstance(cfg_data, dict):
                        opt_id = cfg_data.get("optimizer_id", opt_id)
                        raw_tid = cfg_data.get("task_id", cfg_data.get("task_name", task_id))
                        if isinstance(raw_tid, str) and raw_tid.startswith("${"):
                            task_dict = cfg_data.get("task")
                            if isinstance(task_dict, dict) and "name" in task_dict and str(task_dict["name"]).strip():
                                raw_tid = task_dict["name"]
                        task_id = str(raw_tid) if raw_tid is not None else task_id
                        seed = int(cfg_data.get("seed", seed))
                except Exception:
                    pass

        # 3. Path-based heuristic fallback if task_id / opt_id is still unresolved
        if opt_id == "unknown" or task_id == "unknown" or task_id.startswith("${"):
            parts = run_dir.parts
            for i, p in enumerate(parts):
                if "ProximityLCB" in p or "SMAC3_HPOFacade" in p:
                    opt_id = p
                if p == "BBOB" and i + 4 < len(parts):
                    # e.g. runs/<opt>/BBOB/bbob/16/1/0/<seed>
                    task_id = f"bbob/{parts[i+2]}/{parts[i+3]}/{parts[i+4]}"
                elif "cfg_16_" in p or "cfg_32_" in p:
                    task_id = p
                elif ("bbob" in p.lower() or "cfg_" in p) and task_id == "unknown":
                    task_id = p
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
    if df.empty:
        return df

    grouper = ["task_id", "optimizer_id", "seed"]
    df = df.sort_values(by=grouper + ["n_trials"]).reset_index(drop=True)
    df["trial_value__cost_inc"] = df.groupby(grouper)["trial_value__cost"].cummin()

    # Extract dimension
    df["dimension"] = df["task_id"].apply(lambda t: 16 if ("16_" in str(t) or "/16/" in str(t)) else (32 if ("32_" in str(t) or "/32/" in str(t)) else 0))

    csv_path = out_path / "logs.csv"
    parquet_path = out_path / "logs.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"[SUCCESS] Saved {len(df)} records across {len(trial_log_files)} runs to {parquet_path}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Gather BBOB High-D Proximity Sweep logs")
    parser.add_argument("--runs-dir", type=str, default="runs/sweep_bbob_highdim_proximity")
    parser.add_argument("--output-dir", type=str, default="results/sweep_bbob_highdim_proximity")
    args = parser.parse_args()

    gather_bbob_highdim_logs(runs_dir=args.runs_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
