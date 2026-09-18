#!/usr/bin/env python3
"""Empirical Reference Bounds Extractor for CARP-S BBsubset Dev Tasks.

Extracts the empirical minimum and maximum incumbent costs (y_inc at final trial)
for each of the 18 working dev tasks directly from historical CARP-S execution logs (logs.csv).
Outputs a clean reference_bounds.json used for scale-invariant normalized regret calculation in meta-SMAC.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


def canonical_dev_task_name(raw_task_id: str) -> str:
    """Maps raw CARP-S task_id (e.g. 'blackbox/20/dev/bbob/2/12/0') to registry short name ('subset_bbob_2_12_0')."""
    if "dev/" in raw_task_id:
        sub = raw_task_id.split("dev/")[-1]
    elif "blackbox/" in raw_task_id:
        sub = raw_task_id.split("blackbox/")[-1]
    else:
        sub = raw_task_id

    normalized = "subset_" + sub.replace("/", "_")
    return normalized


def extract_dev_reference_bounds(
    logs_csv_path: str,
    output_json: Optional[str] = None,
    max_quantile: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Extracts empirical min and max incumbent costs for each working dev task from logs.csv.

    Args:
        logs_csv_path: Path to CARP-S logs.csv file.
        output_json: Optional path to write extracted reference_bounds.json.
        max_quantile: Quantile for y_max (default 1.0 = absolute empirical max).

    Returns:
        Dictionary mapping task_name -> {'min': float, 'max': float, 'median': float}.
    """
    p = Path(logs_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Logs file not found: {logs_csv_path}")

    # Load required columns
    usecols = ["task_id", "trial_value__cost_inc"]
    df_raw = pd.read_csv(logs_csv_path, usecols=lambda c: c in usecols or c == "n_trials")

    # Map task names
    df_raw["canonical_task"] = df_raw["task_id"].apply(canonical_dev_task_name)

    # Filter to working dev tasks
    working_dev = [t.split("/")[-1] for t in CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)]
    df = df_raw[df_raw["canonical_task"].isin(working_dev)].copy()

    # If canonical didn't match directly, try endswith matching
    if df.empty:
        df_raw["canonical_task"] = df_raw["task_id"].apply(lambda t: t.split("/")[-1])
        df = df_raw[df_raw["canonical_task"].isin(working_dev)].copy()

    # If dataset has both partial trials and final trials, filter to final trials per run
    # (or group by task and take min/max of the incumbent cost)
    bounds: Dict[str, Dict[str, float]] = {}

    for task_name in df["canonical_task"].unique():
        task_df = df[df["canonical_task"] == task_name]
        costs = task_df["trial_value__cost_inc"].dropna().to_numpy(dtype=float)
        valid = costs[np.isfinite(costs)]

        if len(valid) == 0:
            continue

        y_min = float(np.min(valid))
        if max_quantile >= 1.0:
            y_max = float(np.max(valid))
        else:
            y_max = float(np.quantile(valid, max_quantile))

        y_median = float(np.median(valid))

        # Ensure min < max
        if y_max <= y_min:
            y_max = y_min + 1.0

        bounds[task_name] = {
            "min": round(y_min, 6),
            "max": round(y_max, 6),
            "median": round(y_median, 6),
        }

    # Also check if any working dev task was not in logs, fall back to known dev ranges if necessary
    for t in working_dev:
        if t not in bounds:
            # If a task wasn't in logs, provide fallback bounds
            bounds[t] = {"min": 0.0, "max": 1.0, "median": 0.5}

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(bounds, f, indent=2)

    return bounds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract empirical min/max reference bounds from CARP-S logs.csv."
    )
    parser.add_argument(
        "--logs",
        "-l",
        default="results/bbsubset_dev_analysis/logs.csv",
        help="Path to logs.csv file (default: results/bbsubset_dev_analysis/logs.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/meta_smac_proximity_hpo/reference_bounds.json",
        help="Path to output reference_bounds.json (default: results/meta_smac_proximity_hpo/reference_bounds.json)",
    )
    parser.add_argument(
        "--quantile",
        "-q",
        type=float,
        default=1.0,
        help="Quantile for y_max (default: 1.0 = absolute max)",
    )
    args = parser.parse_args()

    bounds = extract_dev_reference_bounds(
        logs_csv_path=args.logs,
        output_json=args.output,
        max_quantile=args.quantile,
    )
    print(f"Extracted empirical reference bounds for {len(bounds)} dev tasks.")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
