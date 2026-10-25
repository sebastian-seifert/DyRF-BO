#!/usr/bin/env python3
"""TDD Test Suite for Unified Sweep Scorecard Engine."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure DyRF-BO project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compute_sweep_scorecard import (
    apply_holm_bonferroni,
    calculate_rank_biserial_correlation,
    compute_min_max_scaled_regret,
    compute_sweep_scorecard,
    export_scorecard,
    format_scorecard_markdown,
)


def test_statistical_primitives_holm_bonferroni():
    """Verify Holm-Bonferroni step-down adjustment."""
    raw_p = [0.01, 0.04, 0.03]
    adj_p = apply_holm_bonferroni(raw_p)
    assert len(adj_p) == 3
    # Sorted: 0.01 * 3 = 0.03; 0.03 * 2 = 0.06; 0.04 * 1 = 0.04 -> max(0.06, 0.04) = 0.06
    assert adj_p[0] == pytest.approx(0.03)
    assert adj_p[1] == pytest.approx(0.06)
    assert adj_p[2] == pytest.approx(0.06)

    # Empty and single-element edge cases
    assert apply_holm_bonferroni([]) == []
    assert apply_holm_bonferroni([0.05]) == [0.05]


def test_statistical_primitives_rank_biserial_clean():
    """Verify matched-pairs rank-biserial correlation under pure win/loss."""
    # When proposed (x) has strictly lower regret than baseline (y) on all tasks
    x_wins = np.array([0.1, 0.2, 0.3])
    y_base = np.array([0.5, 0.6, 0.7])
    r_rb_win = calculate_rank_biserial_correlation(x_wins, y_base, tie_tolerance=1e-6)
    assert r_rb_win == pytest.approx(-1.0)

    # When baseline has strictly lower regret on all tasks
    r_rb_loss = calculate_rank_biserial_correlation(y_base, x_wins, tie_tolerance=1e-6)
    assert r_rb_loss == pytest.approx(1.0)


def test_statistical_primitives_rank_biserial_tie_filtering():
    """Verify that ties within tie_tolerance are strictly filtered out prior to ranking."""
    # Pairs:
    # Task 1: x=0.20, y=0.50 -> diff = -0.30 (Proposed wins, rank 2)
    # Task 2: x=0.3000001, y=0.3000000 -> diff = 1e-7 (TIE <= 1e-6, MUST be filtered out!)
    # Task 3: x=0.40, y=0.30 -> diff = +0.10 (Baseline wins, rank 1)
    #
    # With tie filtered out, remaining non-zero absolute diffs are:
    # 0.10 (rank 1, positive diff -> baseline win, W+ = 1)
    # 0.30 (rank 2, negative diff -> proposed win, W- = 2)
    # r_rb = (W+ - W-) / (W+ + W-) = (1 - 2) / 3 = -1/3 = -0.333333
    x = np.array([0.20, 0.3000001, 0.40])
    y = np.array([0.50, 0.3000000, 0.30])
    r_rb = calculate_rank_biserial_correlation(x, y, tie_tolerance=1e-6)
    assert r_rb == pytest.approx(-1.0 / 3.0)


def test_statistical_primitives_rank_biserial_all_ties():
    """Verify that when all pairs are ties within tolerance, r_rb returns 0.0."""
    x = np.array([0.10, 0.20, 0.30])
    y = np.array([0.10, 0.20 + 5e-7, 0.30 - 5e-7])
    r_rb = calculate_rank_biserial_correlation(x, y, tie_tolerance=1e-6)
    assert r_rb == pytest.approx(0.0)


def test_statistical_primitives_min_max_scaling():
    """Verify task-level min-max normalization to [0, 1]."""
    df = pd.DataFrame([
        {"task_id": "T1", "optimizer_id": "opt1", "cost": 10.0},
        {"task_id": "T1", "optimizer_id": "opt2", "cost": 20.0},
        {"task_id": "T2", "optimizer_id": "opt1", "cost": 100.0},
        {"task_id": "T2", "optimizer_id": "opt2", "cost": 300.0},
    ])
    scaled = compute_min_max_scaled_regret(df, cost_col="cost")
    assert "normalized_regret" in scaled.columns
    t1_opt1 = scaled[(scaled["task_id"] == "T1") & (scaled["optimizer_id"] == "opt1")]["normalized_regret"].iloc[0]
    t1_opt2 = scaled[(scaled["task_id"] == "T1") & (scaled["optimizer_id"] == "opt2")]["normalized_regret"].iloc[0]
    assert t1_opt1 == pytest.approx(0.0)
    assert t1_opt2 == pytest.approx(1.0)


def test_scorecard_schema_and_exclusion():
    """Verify that compute_sweep_scorecard strictly excludes baselines from evaluated rows
    and retains only corrected p-value, winrate, and biserial correlation per baseline.
    """
    records = []
    # 6 tasks, 2 seeds per task, 100 trials
    # 2 baselines: SMAC3_HPOFacade_lcb, SMAC3_HPOFacade_ei
    # 2 custom approaches: SMAC20_ProximityA_LCB, SMAC20_ProximityB_LCB
    tasks = [f"task_{i}" for i in range(1, 7)]
    opts = [
        "SMAC3_HPOFacade_lcb",
        "SMAC3_HPOFacade_ei",
        "SMAC20_ProximityA_LCB",
        "SMAC20_ProximityB_LCB",
    ]
    for task_id in tasks:
        for opt in opts:
            for seed in [1, 2]:
                for trial in [10, 100]:
                    if opt == "SMAC20_ProximityA_LCB":
                        cost = 0.10 if trial == 100 else 0.50
                    elif opt == "SMAC20_ProximityB_LCB":
                        cost = 0.20 if trial == 100 else 0.60
                    elif opt == "SMAC3_HPOFacade_ei":
                        cost = 0.30 if trial == 100 else 0.70
                    else:  # SMAC3_HPOFacade_lcb
                        cost = 0.40 if trial == 100 else 0.80

                    records.append({
                        "task_id": task_id,
                        "optimizer_id": opt,
                        "seed": seed,
                        "trial": trial,
                        "trial_value__cost_inc": cost,
                    })

    df = pd.DataFrame(records)
    results = compute_sweep_scorecard(
        df,
        controls=["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"],
        cost_col="trial_value__cost_inc",
        tie_tolerance=1e-6,
    )

    assert "scorecard" in results
    scorecard = results["scorecard"]

    # 1. Exclusion assertion: The baselines MUST NOT be rows in the scorecard!
    assert "SMAC3_HPOFacade_lcb" not in scorecard["optimizer_id"].values
    assert "SMAC3_HPOFacade_ei" not in scorecard["optimizer_id"].values
    assert set(scorecard["optimizer_id"].values) == {
        "SMAC20_ProximityA_LCB",
        "SMAC20_ProximityB_LCB",
    }

    # 2. Schema assertion: strictly optimizer_id and [win_rate, wilcoxon_adj_p, rank_biserial] per baseline
    expected_cols = [
        "optimizer_id",
        "ctrl_lcb_win_rate",
        "ctrl_lcb_wilcoxon_adj_p",
        "ctrl_lcb_rank_biserial",
        "ctrl_ei_win_rate",
        "ctrl_ei_wilcoxon_adj_p",
        "ctrl_ei_rank_biserial",
    ]
    assert list(scorecard.columns) == expected_cols

    # 3. Value check: ProximityA beats both baselines on all tasks
    row_a = scorecard[scorecard["optimizer_id"] == "SMAC20_ProximityA_LCB"].iloc[0]
    assert row_a["ctrl_lcb_win_rate"] == pytest.approx(1.0)
    assert row_a["ctrl_lcb_rank_biserial"] == pytest.approx(-1.0)
    assert row_a["ctrl_lcb_wilcoxon_adj_p"] < 0.05

    assert row_a["ctrl_ei_win_rate"] == pytest.approx(1.0)
    assert row_a["ctrl_ei_rank_biserial"] == pytest.approx(-1.0)
    assert row_a["ctrl_ei_wilcoxon_adj_p"] < 0.05


def test_export_scorecard_and_markdown(tmp_path):
    """Verify that export_scorecard writes valid CSV and Markdown files containing only requested columns."""
    df_scorecard = pd.DataFrame([
        {
            "optimizer_id": "SMAC20_ProximityAC_LCB",
            "ctrl_lcb_win_rate": 0.95,
            "ctrl_lcb_wilcoxon_adj_p": 1.2e-18,
            "ctrl_lcb_rank_biserial": -0.972,
            "ctrl_ei_win_rate": 0.836,
            "ctrl_ei_wilcoxon_adj_p": 5.3e-14,
            "ctrl_ei_rank_biserial": -0.849,
        }
    ])
    results = {
        "scorecard": df_scorecard,
        "full_scorecard": df_scorecard,
    }

    csv_path, md_path = export_scorecard(
        results,
        output_dir=tmp_path,
        title="Test YAHPO Scorecard",
        controls=["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"],
    )

    assert csv_path.exists()
    assert md_path.exists()

    csv_content = pd.read_csv(csv_path)
    assert list(csv_content.columns) == list(df_scorecard.columns)

    md_text = md_path.read_text(encoding="utf-8")
    assert "# Test YAHPO Scorecard" in md_text
    assert "LCB Win Rate" in md_text
    assert "LCB Adj. p" in md_text
    assert "LCB $r_{rb}$" in md_text
    assert "EI Win Rate" in md_text
    assert "EI Adj. p" in md_text
    assert "EI $r_{rb}$" in md_text
    assert "SMAC20_ProximityAC_LCB" in md_text
    assert "-0.972" in md_text


