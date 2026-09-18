#!/usr/bin/env python3
"""Scale-Invariant Meta-Loss Engine for Multi-Task Meta-Optimization.

Computes stationary normalized regret across the 18 CARP-S BBsubset dev tasks:
- For each task m with empirical reference bounds [y_min_m, y_max_m]:
  r_m(theta) = (mean_s y_m(theta, s) - y_min_m) / (y_max_m - y_min_m)
- Missing or failed task runs receive worst-case normalized regret (1.0).
- Aggregate Meta-Loss:
  Loss(theta) = (1 / M) * sum_{m=1}^M r_m(theta)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_BOUNDS_JSON = os.path.join(
    PROJECT_ROOT, "results", "meta_smac_proximity_hpo", "reference_bounds.json"
)


def load_dev_reference_bounds(bounds_file: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Loads empirical reference bounds from JSON file or computes fallback."""
    file_path = bounds_file or DEFAULT_BOUNDS_JSON

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    # If JSON is not found, extract on the fly from logs.csv if available
    logs_csv = os.path.join(PROJECT_ROOT, "results", "bbsubset_dev_analysis", "logs.csv")
    if os.path.exists(logs_csv):
        from scripts.extract_dev_reference_bounds import extract_dev_reference_bounds
        return extract_dev_reference_bounds(logs_csv, output_json=file_path)

    # Minimal unit test fallback
    return {}


def compute_meta_loss(
    task_results: Dict[str, List[float]],
    ref_bounds: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """Computes the aggregate scale-invariant meta-loss across all benchmark tasks.

    Args:
        task_results: Mapping from task_name to list of evaluated costs across seeds.
        ref_bounds: Mapping from task_name to {'min': float, 'max': float}.
                    If None, loads empirical bounds via load_dev_reference_bounds().

    Returns:
        Scalar meta-loss in [0, 1] (or > 1 if severely exceeding worst baseline).
    """
    if ref_bounds is None:
        ref_bounds = load_dev_reference_bounds()

    if not ref_bounds:
        return 1.0

    task_regrets: List[float] = []

    for task_name, bounds in ref_bounds.items():
        y_min = float(bounds.get("min", 0.0))
        y_max = float(bounds.get("max", 1.0))
        spread = y_max - y_min

        costs = task_results.get(task_name, [])
        valid_costs = [c for c in costs if np.isfinite(c)]

        if len(valid_costs) == 0:
            # Penalize crashed, missing, or empty evaluations with maximum regret
            task_regrets.append(1.0)
            continue

        mean_cost = float(np.mean(valid_costs))

        if spread > 1e-12:
            norm_cost = (mean_cost - y_min) / spread
            task_regrets.append(norm_cost)
        else:
            task_regrets.append(0.0)

    return float(np.mean(task_regrets))
