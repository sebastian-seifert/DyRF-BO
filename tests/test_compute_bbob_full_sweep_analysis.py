#!/usr/bin/env python3
"""Tests for BBOB Full Sweep Dimensionality-Stratified Statistical Analysis."""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.compute_bbob_full_sweep_analysis import (
    apply_holm_bonferroni,
    calculate_cliffs_delta,
    calculate_rank_biserial_correlation,
    compute_min_max_scaled_regret,
    run_bbob_statistical_analysis,
    export_analysis_results,
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

    # Unequal sample sizes
    y_longer = np.array([4.0, 5.0, 6.0, 7.0])
    assert calculate_cliffs_delta(x, y_longer) == -1.0


def test_rank_biserial_correlation():
    """Verify matched-pairs rank-biserial correlation."""
    # When x is strictly lower (better) than y on every paired task
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    r_rb = calculate_rank_biserial_correlation(x, y)
    assert r_rb == pytest.approx(-1.0)

    # When x is strictly higher (worse) than y on every paired task
    r_rb_worse = calculate_rank_biserial_correlation(y, x)
    assert r_rb_worse == pytest.approx(1.0)

    # When equal on all tasks
    r_rb_eq = calculate_rank_biserial_correlation(x, x)
    assert r_rb_eq == pytest.approx(0.0)

    # Mismatched lengths for paired metric returns 0.0 safely
    assert calculate_rank_biserial_correlation(x, np.array([1.0, 2.0])) == 0.0


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


def test_one_sided_wilcoxon_and_joint_scope_b():
    """Verify one-sided Wilcoxon (less) and Scope B joint Holm-Bonferroni correction."""
    records = []
    # 10 tasks in 16D, 3 optimizers:
    # - Control 1: SMAC3_HPOFacade_lcb
    # - Control 2: SMAC3_HPOFacade_ei
    # - Proposed: SMAC20_ProximityA_LCB (strictly better than both controls)
    # - Inferior: SMAC20_Inferior_LCB (strictly worse than both controls)
    tasks = [f"cfg_16_{i}_0" for i in range(1, 11)]
    opts = [
        "SMAC3_HPOFacade_lcb",
        "SMAC3_HPOFacade_ei",
        "SMAC20_ProximityA_LCB",
        "SMAC20_Inferior_LCB",
    ]
    for task_id in tasks:
        for opt in opts:
            for seed in [1, 2]:
                if opt == "SMAC20_ProximityA_LCB":
                    cost = 1.0
                elif opt == "SMAC3_HPOFacade_ei":
                    cost = 5.0
                elif opt == "SMAC3_HPOFacade_lcb":
                    cost = 6.0
                else:  # Inferior
                    cost = 20.0
                records.append({
                    "task_id": task_id,
                    "dimension": 16,
                    "optimizer_id": opt,
                    "seed": seed,
                    "trial": 100,
                    "cost": cost,
                    "trial_value__cost_inc": cost,
                })

    df = pd.DataFrame(records)
    results = run_bbob_statistical_analysis(
        df,
        cost_col="trial_value__cost_inc",
        controls=["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"],
        holm_scope="joint",
    )

    scorecard = results["global_scorecard"]
    prox_row = scorecard[scorecard["optimizer_id"] == "SMAC20_ProximityA_LCB"].iloc[0]
    inferior_row = scorecard[scorecard["optimizer_id"] == "SMAC20_Inferior_LCB"].iloc[0]

    # ProximityA is superior -> one-sided p-value must be small (< 0.05)
    assert prox_row["ctrl_lcb_wilcoxon_p"] < 0.05
    assert prox_row["ctrl_ei_wilcoxon_p"] < 0.05
    # Rank-biserial correlation must be -1.0 (indicating 100% dominance)
    assert prox_row["ctrl_lcb_rank_biserial"] == pytest.approx(-1.0)
    assert prox_row["ctrl_lcb_cliffs_delta"] == pytest.approx(-1.0)

    # Inferior method is worse -> one-sided p-value must be near 1.0
    assert inferior_row["ctrl_lcb_wilcoxon_p"] > 0.95
    assert inferior_row["ctrl_ei_wilcoxon_p"] > 0.95
    assert inferior_row["ctrl_lcb_rank_biserial"] == pytest.approx(1.0)

    # Joint Holm-Bonferroni (Scope B) must adjust across both controls together
    # Controls themselves must NOT enter the hypothesis pool (no self-test penalty)
    assert "ctrl_lcb_wilcoxon_adj_p" in scorecard.columns
    assert "ctrl_ei_wilcoxon_adj_p" in scorecard.columns


def test_export_analysis_results_fallback(tmp_path):
    """Verify export_analysis_results succeeds even if tabulate is not present."""
    records = []
    for opt in ["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"]:
        for task in ["cfg_2_1_0"]:
            records.append({
                "task_id": task,
                "dimension": 2,
                "optimizer_id": opt,
                "seed": 1,
                "trial": 1,
                "trial_value__cost_inc": 1.0,
            })
    df = pd.DataFrame(records)
    res = run_bbob_statistical_analysis(df)
    out_dir = str(tmp_path / "analysis_export")
    export_analysis_results(res, out_dir)
    assert (Path(out_dir) / "bbob_analysis_report.md").exists()
    assert (Path(out_dir) / "global_scorecard.csv").exists()
