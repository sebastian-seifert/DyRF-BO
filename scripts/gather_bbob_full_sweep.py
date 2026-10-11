#!/usr/bin/env python3
"""Data Gathering and Aggregation Script for Full BBOB Benchmark Sweeps.

Recursively gathers trial execution logs (`trial_logs.jsonl`) and fallback telemetry
files from `runs/sweep_bbob_full_comparison`, enriching every row with explicit
dimensionality, function ID, instance ID, and function category metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_bbob_full_registry import (
    BBOB_FUNCTION_METADATA,
    CarpsBBOBFullRegistry,
)


def extract_bbob_task_attributes(task_str: str) -> Dict[str, Any]:
    """Extracts dimension, fid, iid, name, and category from a task identifier."""
    match = re.search(r"cfg_(\d+)_(\d+)_(\d+)", task_str)
    if match:
        dim = int(match.group(1))
        fid = int(match.group(2))
        iid = int(match.group(3))
    else:
        # Fallback regex for alternative paths like bbob/16/12/0
        m2 = re.search(r"bbob[/-](\d+)[/-](\d+)[/-](\d+)", task_str, re.IGNORECASE)
        if m2:
            dim = int(m2.group(1))
            fid = int(m2.group(2))
            iid = int(m2.group(3))
        else:
            dim = -1
            fid = -1
            iid = -1

    if fid in BBOB_FUNCTION_METADATA:
        name, cat = BBOB_FUNCTION_METADATA[fid]
        category_str = cat.value
    else:
        name = "Unknown"
        category_str = "unknown"

    return {
        "dimension": dim,
        "fid": fid,
        "iid": iid,
        "function_name": name,
        "function_category": category_str,
    }


def gather_bbob_trial_logs(
    runs_base: Path | str = "runs/sweep_bbob_full_comparison",
    outdir: Path | str = "results/sweep_bbob_full_comparison",
) -> pd.DataFrame:
    """Recursively parses all trial logs or telemetry files and saves consolidated logs."""
    runs_path = Path(runs_base)
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    # 1. Search for trial_logs.jsonl
    trial_log_files = list(runs_path.glob("**/trial_logs.jsonl")) if runs_path.exists() else []

    if trial_log_files:
        for t_file in trial_log_files:
            run_dir = t_file.parent
            hydra_cfg = run_dir / ".hydra" / "config.yaml"

            opt_id = "unknown"
            task_id = "unknown"
            seed = 1

            if hydra_cfg.exists():
                try:
                    import yaml
                    with open(hydra_cfg, "r", encoding="utf-8") as f:
                        cfg_data = yaml.safe_load(f)
                    if isinstance(cfg_data, dict):
                        opt_id = cfg_data.get("optimizer_id", opt_id)
                        task_id = cfg_data.get("task", {}).get("name", task_id)
                        seed = cfg_data.get("seed", seed)
                except Exception:
                    pass

            if opt_id == "unknown" or task_id == "unknown":
                parts = run_dir.parts
                for p in parts:
                    if "SMAC" in p:
                        opt_id = p
                    if "cfg_" in p or "subset_" in p:
                        task_id = p
                    if p.isdigit():
                        seed = int(p)

            meta = extract_bbob_task_attributes(task_id)

            with open(t_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    trial_num = row.get("trial", row.get("n_trials", row.get("n_function_calls", 1)))
                    cost = row.get("trial_value", {}).get("cost", np.nan)

                    record = {
                        "task_id": task_id,
                        "optimizer_id": opt_id,
                        "seed": seed,
                        "trial": trial_num,
                        "cost": cost,
                        **meta,
                    }
                    records.append(record)

    # 2. Fallback to telemetry JSON files if no trial_logs found
    telemetry_files = list(out_path.glob("telemetry_*.json"))
    if not records and telemetry_files:
        print(f"[INFO] Parsing {len(telemetry_files)} telemetry JSON files...")
        for telem_file in telemetry_files:
            try:
                with open(telem_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                opt_id = data.get("optimizer_id", "unknown")
                task_id = data.get("task_id", "unknown")
                seed = data.get("seed", 1)
                meta = extract_bbob_task_attributes(task_id)

                history = data.get("history", [])
                for entry in history:
                    trial_num = entry.get("trial", entry.get("n_trials", 1))
                    cost = entry.get("cost", np.nan)
                    records.append({
                        "task_id": task_id,
                        "optimizer_id": opt_id,
                        "seed": seed,
                        "trial": trial_num,
                        "cost": cost,
                        **meta,
                    })
            except Exception:
                continue

    if not records:
        print(f"[WARN] No trial logs or telemetry records found in {runs_path} or {out_path}.")
        df = pd.DataFrame(columns=[
            "task_id", "optimizer_id", "seed", "trial", "cost", "trial_value__cost_inc",
            "dimension", "fid", "iid", "function_name", "function_category"
        ])
        return df

    df = pd.DataFrame(records)

    # Sort and compute running incumbent (best so far)
    grouper = ["task_id", "optimizer_id", "seed"]
    df = df.sort_values(grouper + ["trial"])
    df["trial_value__cost_inc"] = df.groupby(grouper)["cost"].cummin()

    # Save to parquet and csv
    parquet_path = out_path / "logs.parquet"
    csv_path = out_path / "logs.csv"
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"[WARN] Parquet export failed: {e}")
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Exported {len(df)} records to {parquet_path} and {csv_path}.")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather BBOB full sweep execution logs.")
    parser.add_argument("--runs-base", type=str, default="runs/sweep_bbob_full_comparison")
    parser.add_argument("--outdir", type=str, default="results/sweep_bbob_full_comparison")
    args = parser.parse_args()

    gather_bbob_trial_logs(runs_base=args.runs_base, outdir=args.outdir)
