#!/usr/bin/env python3
"""Data Gathering and Aggregation Script for YAHPO Gym rbv2_ranger Sweep.

Recursively gathers trial execution logs (`trial_logs.jsonl`) and fallback telemetry
files from `runs/sweep_yahpo_rbv2_ranger`, enriching every row with explicit metadata
and robustly handling CARP-S path hierarchies (.../rbv2_ranger/<dataset_id>/None/<seed>/trial_logs.jsonl).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

YAHPO_RANGER_RUNS_BASE = "runs/sweep_yahpo_rbv2_ranger"
YAHPO_RANGER_OUTDIR = "results/sweep_yahpo_rbv2_ranger"


def extract_yahpo_task_attributes(task_str: str) -> Dict[str, Any]:
    """Extracts task information from a YAHPO task identifier without capturing suite numbers."""
    cleaned = str(task_str).strip().rstrip("/")
    # Target digits specifically after ranger prefixes or whole digits if isolated
    match = re.search(r"(?:(?:cfg_)?rbv2_ranger[_\/]|ranger[_\/]|cfg_rbv2_ranger_|^)(\d+)(?:\D|$)", cleaned)
    if match:
        dataset_id = int(match.group(1))
    else:
        dataset_id = -1

    task_id = f"cfg_rbv2_ranger_{dataset_id}" if dataset_id != -1 else cleaned
    return {
        "dataset_id": dataset_id,
        "task_id": task_id,
        "suite": "yahpo_rbv2_ranger",
    }


def parse_run_dir_hierarchy(log_file: Path, runs_path: Path) -> Tuple[str, str, int]:
    """Parses (optimizer_id, task_id, seed) from CARP-S directory hierarchy."""
    rel_parts = log_file.relative_to(runs_path).parts
    optimizer_id = rel_parts[0]

    # Pattern A: CARP-S YAHPO standard: .../rbv2_ranger/<dataset_id>/None/<seed>/trial_logs.jsonl
    if len(rel_parts) >= 4 and rel_parts[-3].lower() == "none":
        try:
            seed = int(rel_parts[-2])
        except ValueError:
            seed = -1
        raw_dataset = rel_parts[-4]
    else:
        # Pattern B: Flattened or direct: .../{task_name}/{seed}/trial_logs.jsonl
        try:
            seed = int(rel_parts[-2])
            raw_dataset = rel_parts[-3]
        except (ValueError, IndexError):
            raw_dataset = rel_parts[-2] if len(rel_parts) >= 2 else "unknown"
            seed = -1

    attrs = extract_yahpo_task_attributes(raw_dataset)
    task_id = attrs["task_id"]
    return optimizer_id, task_id, seed


def gather_yahpo_rbv2_ranger_trial_logs(
    runs_base: Path | str = YAHPO_RANGER_RUNS_BASE,
    outdir: Path | str = YAHPO_RANGER_OUTDIR,
) -> pd.DataFrame:
    """Recursively parses all trial logs or telemetry files and saves consolidated logs."""
    runs_path = Path(runs_base)
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    # 1. Search for structured CARP-S trial_logs.jsonl in runs_path
    if runs_path.exists():
        jsonl_files = list(runs_path.glob("**/trial_logs.jsonl"))

        for log_file in jsonl_files:
            try:
                optimizer_id, task_id, seed = parse_run_dir_hierarchy(log_file, runs_path)
                task_attrs = extract_yahpo_task_attributes(task_id)

                min_cost = float("inf")
                with open(log_file, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except Exception:
                            continue

                        trial = entry.get(
                            "trial",
                            entry.get(
                                "trial_idx",
                                entry.get("n_trials", entry.get("n_function_calls", line_idx)),
                            ),
                        )
                        val_dict = entry.get("trial_value") or {}
                        cost = val_dict.get("cost", None)
                        if cost is None:
                            cost = entry.get("cost", None)

                        if cost is not None:
                            try:
                                cost = float(cost)
                            except (ValueError, TypeError):
                                cost = None

                        if cost is not None and not np.isnan(cost):
                            min_cost = min(min_cost, cost)

                        row = {
                            "optimizer_id": optimizer_id,
                            "task_id": task_id,
                            "seed": seed,
                            "trial": trial,
                            "cost": cost,
                            "trial_value__cost_inc": min_cost if min_cost != float("inf") else cost,
                            **task_attrs,
                        }
                        records.append(row)
            except Exception:
                continue

    # 2. Fallback to telemetry JSON files in runs_path or out_path
    if not records:
        search_paths = [runs_path]
        if out_path.exists() and out_path != runs_path:
            search_paths.append(out_path)

        for s_path in search_paths:
            if not s_path.exists():
                continue
            for t_file in s_path.glob("**/telemetry_*.json"):
                try:
                    with open(t_file, "r", encoding="utf-8") as f:
                        t_data = json.load(f)

                    opt_id = t_data.get("optimizer_id", "unknown")
                    raw_task = t_data.get(
                        "task_id",
                        t_data.get("task", t_data.get("task_name", "unknown")),
                    )
                    if raw_task == "unknown":
                        # Infer task from telemetry filename: telemetry_{opt}_{task}_seed{seed}.json
                        stem_parts = t_file.stem.split("_seed")[0].split("_")
                        for idx_p, p in enumerate(stem_parts):
                            if p == "ranger" and idx_p + 1 < len(stem_parts):
                                raw_task = f"cfg_rbv2_ranger_{stem_parts[idx_p + 1]}"
                                break

                    seed = t_data.get("seed", -1)
                    task_attrs = extract_yahpo_task_attributes(raw_task)
                    task_id = task_attrs["task_id"]

                    history = t_data.get("trials", t_data.get("history", []))
                    min_cost = float("inf")
                    for h_idx, item in enumerate(history, 1):
                        c = item.get("cost", item.get("value", None))
                        trial = item.get("trial", item.get("trial_idx", item.get("n_trials", h_idx)))
                        if c is not None:
                            try:
                                c = float(c)
                            except (ValueError, TypeError):
                                c = None

                        if c is not None and not np.isnan(c):
                            min_cost = min(min_cost, c)

                        records.append({
                            "optimizer_id": opt_id,
                            "task_id": task_id,
                            "seed": seed,
                            "trial": trial,
                            "cost": c,
                            "trial_value__cost_inc": min_cost if min_cost != float("inf") else c,
                            **task_attrs,
                        })
                except Exception:
                    continue

    if records:
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame(columns=[
            "optimizer_id", "task_id", "seed", "trial", "cost",
            "trial_value__cost_inc", "dataset_id", "suite"
        ])

    parquet_path = out_path / "logs.parquet"
    csv_path = out_path / "logs.csv"

    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass
    df.to_csv(csv_path, index=False)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather YAHPO rbv2_ranger execution logs.")
    parser.add_argument("--runs-base", type=str, default=YAHPO_RANGER_RUNS_BASE)
    parser.add_argument("--outdir", type=str, default=YAHPO_RANGER_OUTDIR)
    args = parser.parse_args()

    df = gather_yahpo_rbv2_ranger_trial_logs(runs_base=args.runs_base, outdir=args.outdir)
    print(f"Gathered {len(df)} records into {args.outdir}")
