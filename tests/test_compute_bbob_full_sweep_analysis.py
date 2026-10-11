#!/usr/bin/env python3
"""Tests for BBOB Full Sweep Dimensionality-Stratified Statistical Analysis."""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest

from scripts.compute_bbob_full_sweep_analysis import (
    apply_holm_bonferroni,
    calculate_cliffs_delta,
    compute_min_max_scaled_regret,
    run_bbob_statistical_analysis,
)


def test_holm_bonferroni():
    """Verify step-down Holm-Bonferroni p-value adjustment."""
    raw_p = [0.01, 0.04, 0.03]
    adj_p = apply_holm_bonferroni(raw_p)
    assert len(adj_p) == 3
    # Sorted: 0.01 * 3 = 0.03; 0.03 * 2 = 0.06; 0.04 * 1 = 0.04 -> monotonicity max(0.06, 0.04) = 0.06
    assert adj_p[0] == pytest.approx(0.03)
    assert adj_p[1] == pytest.approx(0.06)
    assert adj_p[2] == pytest.approx(0.06)


def test_cliffs_delta():
    """Verify Cliff's delta effect size calculation."""
    # When x is strictly lower (better for loss minimization) than y
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    delta = calculate_cliffs_delta(x, y)
    assert delta == -1.0

    # When equal
    delta_eq = calculate_cliffs_delta(x, x)
    assert delta_eq == 0.0


def test_min_max_scaled_regret():
    """Verify global per-task min-max normalization."""
    df = pd.DataFrame([
        {"task_id": "task_A", "optimizer_id": "opt1", "cost": 10.0},
        {"task_id": "task_A", "optimizer_id": "opt2", "cost": 20.0},
        {"task_id": "task_B", "optimizer_id": "opt1", "cost": 100.0},
        {"task_id": "task_B", "optimizer_id": "opt2", "cost": 200.0},
    ])
    scaled = compute_min_max_scaled_regret(df, cost_col="cost")
    assert "normalized_regret" in scaled.columns

    # For task_A: min=10, max=20 -> opt1 is 0.0, opt2 is 1.0
    task_a_opt1 = scaled[(scaled["task_id"] == "task_A") & (scaled["optimizer_id"] == "opt1")]["normalized_regret"].iloc[0]
    task_a_opt2 = scaled[(scaled["task_id"] == "task_A") & (scaled["optimizer_id"] == "opt2")]["normalized_regret"].iloc[0]
    assert task_a_opt1 == pytest.approx(0.0)
    assert task_a_opt2 == pytest.approx(1.0)


def test_run_bbob_statistical_analysis_with_dimension_stratification(tmp_path):
    """Verify analysis generates both global and dimension-stratified scorecards."""
    # Build synthetic multi-task, multi-dimension dataset
    records = []
    # 2 tasks in 2D, 2 tasks in 16D
    tasks = [
        ("cfg_2_1_0", 2),
        ("cfg_2_2_0", 2),
        ("cfg_16_1_0", 16),
        ("cfg_16_2_0", 16),
    ]
    opts = ["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei", "SMAC20_ProximityA_LCB"]
    for task_id, dim in tasks:
        for opt in opts:
            for seed in [1, 2, 3]:
                # Let ProximityA have lower cost in 16D, higher in 2D
                if opt == "SMAC20_ProximityA_LCB":
                    base_cost = 5.0 if dim == 16 else 15.0
                elif opt == "SMAC3_HPOFacade_ei":
                    base_cost = 10.0
                else:
                    base_cost = 12.0
                records.append({
                    "task_id": task_id,
                    "dimension": dim,
                    "optimizer_id": opt,
                    "seed": seed,
                    "trial": 100,
                    "cost": base_cost + 0.1 * seed,
                    "trial_value__cost_inc": base_cost + 0.1 * seed,
                })

    df = pd.DataFrame(records)
    results = run_bbob_statistical_analysis(df, cost_col="trial_value__cost_inc")

    assert "global_scorecard" in results
    assert "by_dimension" in results
    assert "by_tier" in results

    # Dimensions 2 and 16 should be in results['by_dimension']
    assert 2 in results["by_dimension"]
    assert 16 in results["by_dimension"]

    scorecard_2d = results["by_dimension"][2]
    scorecard_16d = results["by_dimension"][16]
    assert not scorecard_2d.empty
    assert not scorecard_16d.empty
    assert "optimizer_id" in scorecard_2d.columns
    assert "mean_norm_regret" in scorecard_2d.columns
