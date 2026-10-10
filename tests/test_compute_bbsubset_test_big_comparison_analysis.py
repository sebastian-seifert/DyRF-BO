"""TDD Tests for BBSubset Big Comparison Analysis Engine.

Verifies:
1. Min-max scaling computes normalized regret strictly in [0, 1].
2. Holm-Bonferroni correction strictly preserves monotonicity and bounds.
3. Cliff's Delta computation matches expected direction.
4. Paired Wilcoxon and Friedman omnibus statistics correctly execute.
"""

import os
import sys

# Ensure DyRF-BO root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest
from scripts.compute_bbsubset_test_big_comparison_analysis import (
    apply_holm_bonferroni,
    calculate_cliffs_delta,
    compute_min_max_scaled_regret,
    run_statistical_analysis,
    generate_markdown_scorecard,
)


def test_apply_holm_bonferroni():
    raw_p = [0.001, 0.04, 0.03, 0.20]
    adj_p = apply_holm_bonferroni(raw_p)
    assert len(adj_p) == len(raw_p)
    # Check bounds
    assert all(0.0 <= p <= 1.0 for p in adj_p)
    # Check step-down adjustments
    assert adj_p[0] == pytest.approx(0.001 * 4, abs=1e-5)
    # Monotonicity with sorted rank
    sorted_pairs = sorted(zip(raw_p, adj_p), key=lambda x: x[0])
    sorted_adjs = [p[1] for p in sorted_pairs]
    assert sorted_adjs == sorted(sorted_adjs)


def test_calculate_cliffs_delta():
    # x strictly smaller than y -> delta = -1.0
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    assert calculate_cliffs_delta(x, y) == -1.0

    # x strictly larger than y -> delta = +1.0
    assert calculate_cliffs_delta(y, x) == +1.0

    # identical -> delta = 0.0
    assert calculate_cliffs_delta(x, x) == 0.0


def test_compute_min_max_scaled_regret():
    df = pd.DataFrame({
        "task_id": ["task_a", "task_a", "task_b", "task_b"],
        "trial_value__cost_inc": [10.0, 20.0, 100.0, 200.0],
    })
    scaled = compute_min_max_scaled_regret(df, cost_col="trial_value__cost_inc")
    assert "normalized_regret" in scaled.columns
    # task_a: min=10, max=20 -> [0.0, 1.0]
    # task_b: min=100, max=200 -> [0.0, 1.0]
    np.testing.assert_allclose(scaled["normalized_regret"].values, [0.0, 1.0, 0.0, 1.0])


def test_run_statistical_analysis_mock():
    # Mock dataframe with 2 tasks, 2 seeds, 3 optimizers
    records = []
    opts = ["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei", "SMAC20_ProximityA_LCB"]
    for task in ["task_1", "task_2"]:
        for seed in [1, 2]:
            for opt in opts:
                base_val = 0.5 if opt == "SMAC3_HPOFacade_lcb" else (0.8 if opt == "SMAC3_HPOFacade_ei" else 0.2)
                records.append({
                    "task_id": task,
                    "optimizer_id": opt,
                    "seed": seed,
                    "n_trials": 100,
                    "trial_value__cost_inc": base_val + 0.01 * seed,
                })
    df = pd.DataFrame(records)
    res = run_statistical_analysis(df, cost_col="trial_value__cost_inc")
    assert "comparisons" in res
    assert "SMAC3_HPOFacade_lcb" in res["comparisons"]
    assert "SMAC3_HPOFacade_ei" in res["comparisons"]

    scorecard = generate_markdown_scorecard(res)
    assert "# CARP-S BBSubset Held-Out Test Big Comparison Scorecard" in scorecard
    assert "SMAC20_ProximityA_LCB" in scorecard
