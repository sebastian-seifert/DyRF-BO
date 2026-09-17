#!/usr/bin/env python3
"""Tests for compute_sweep_median_wilcoxon script.

Strict TDD verification:
- Test task-level min-max normalization (scale invariance across tasks with different dynamic ranges)
- Test paired seed extraction and incumbent resolution per run
- Test task median of normalized seed differences calculation
- Test one-sided Wilcoxon signed-rank test (alternative='less')
- Test handling of ties, zero-spread edge cases, and optimizer auto-detection
- Test end-to-end processing of mock sweep CSV
"""

import numpy as np
import pandas as pd
import pytest

from scripts.compute_sweep_median_wilcoxon import (
    auto_detect_optimizers,
    compute_task_median_differences,
    run_one_sided_wilcoxon,
    process_sweep_logs,
)


@pytest.fixture
def mock_multiscale_sweep_df():
    """Generates synthetic sweep data with vastly different scales across tasks.
    
    Task 1: Large scale [0, 100,000] - proposed is slightly better.
    Task 2: Small scale [0, 0.01] - proposed is moderately better.
    Task 3: Unit scale [0, 1] - proposed is slightly worse.
    """
    records = []
    # Task 1: Scale ~100,000. Proposed wins by 10,000 (10% normalized)
    for s in range(1, 6):
        records.append({
            "task_id": "bbob/huge_scale",
            "optimizer_id": "SMAC20_ProximityLCB",
            "seed": s,
            "n_trials": 100,
            "trial_value__cost_inc": 20000.0 + s * 1000.0,
        })
        records.append({
            "task_id": "bbob/huge_scale",
            "optimizer_id": "SMAC3_HPOFacade_ei",
            "seed": s,
            "n_trials": 100,
            "trial_value__cost_inc": 30000.0 + s * 1000.0,
        })

    # Task 2: Small scale [0, 0.01]. Proposed wins by 0.005 (50% normalized)
    for s in range(1, 6):
        records.append({
            "task_id": "yahpo/tiny_scale",
            "optimizer_id": "SMAC20_ProximityLCB",
            "seed": s,
            "n_trials": 100,
            "trial_value__cost_inc": 0.002,
        })
        records.append({
            "task_id": "yahpo/tiny_scale",
            "optimizer_id": "SMAC3_HPOFacade_ei",
            "seed": s,
            "n_trials": 100,
            "trial_value__cost_inc": 0.007,
        })

    # Task 3: Unit scale [0, 1.0]. Proposed loses by 0.2 (20% normalized)
    for s in range(1, 6):
        records.append({
            "task_id": "bbob/unit_scale",
            "optimizer_id": "SMAC20_ProximityLCB",
            "seed": s,
            "n_trials": 100,
            "trial_value__cost_inc": 0.6,
        })
        records.append({
            "task_id": "bbob/unit_scale",
            "optimizer_id": "SMAC3_HPOFacade_ei",
            "seed": s,
            "n_trials": 100,
            "trial_value__cost_inc": 0.4,
        })

    return pd.DataFrame(records)


def test_auto_detect_optimizers():
    """Verifies that Proximity and baseline optimizers are correctly detected."""
    opts_1 = ["SMAC20_ProximityLCB_tuned", "SMAC3_HPOFacade_lcb"]
    prop, base = auto_detect_optimizers(opts_1)
    assert prop == "SMAC20_ProximityLCB_tuned"
    assert base == "SMAC3_HPOFacade_lcb"

    opts_2 = ["SMAC3_HPOFacade_ei", "SMAC20_ProximityLCB"]
    prop, base = auto_detect_optimizers(opts_2)
    assert prop == "SMAC20_ProximityLCB"
    assert base == "SMAC3_HPOFacade_ei"


def test_scale_normalization(mock_multiscale_sweep_df):
    """Verifies that min-max normalization bounds task differences into [-1, 1]."""
    task_medians, seed_diffs, _ = compute_task_median_differences(
        mock_multiscale_sweep_df,
        proposed_id="SMAC20_ProximityLCB",
        baseline_id="SMAC3_HPOFacade_ei",
        normalize=True,
    )

    # Both Task 1 and Task 2 differences must be bounded in [-1, 0] despite 10^7 scale difference
    assert -1.0 <= task_medians["bbob/huge_scale"] <= 0.0
    assert -1.0 <= task_medians["yahpo/tiny_scale"] <= 0.0
    assert 0.0 <= task_medians["bbob/unit_scale"] <= 1.0

    # Both tasks maintain their normalized impact without high-scale dominance
    assert abs(task_medians["yahpo/tiny_scale"]) > abs(task_medians["bbob/huge_scale"])


def test_run_one_sided_wilcoxon():
    """Verifies one-sided Wilcoxon signed-rank test (testing proposed < baseline)."""
    # Negative medians (proposed better than baseline)
    medians = np.array([-0.5, -0.2, -0.3, -0.4, -0.1])
    w_stat, p_val = run_one_sided_wilcoxon(medians)
    assert p_val < 0.05
    assert w_stat == 0.0

    # All zeros / ties
    zero_medians = np.zeros(10)
    w_stat, p_val = run_one_sided_wilcoxon(zero_medians)
    assert p_val == 1.0


def test_process_sweep_logs_end_to_end(mock_multiscale_sweep_df, tmp_path):
    """Verifies end-to-end execution on a CSV file with scale normalization."""
    csv_file = tmp_path / "logs.csv"
    mock_multiscale_sweep_df.to_csv(csv_file, index=False)

    results = process_sweep_logs(str(csv_file), normalize=True)
    assert results["n_tasks"] == 3
    assert results["n_paired_runs"] == 15
    assert results["proposed_id"] == "SMAC20_ProximityLCB"
    assert results["baseline_id"] == "SMAC3_HPOFacade_ei"
    assert "w_stat" in results
    assert "p_value" in results
    assert len(results["task_medians"]) == 3
