#!/usr/bin/env python3
"""TDD Test Suite for Multi-Budget Horizon Evaluation & Statistical Rank Testing.

Tests data slicing at various budget horizons (e.g. T=15, 20, 25, 50),
incumbent recalculation, task-level ranking, Friedman omnibus test,
and Holm-Bonferroni corrected paired Wilcoxon post-hoc comparisons.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_multi_budget_horizons import (
    slice_logs_at_budget,
    compute_ranks_at_budget,
    evaluate_multi_budget_horizons,
    format_multi_budget_markdown,
    run_statistical_tests_at_budget,
    evaluate_multi_budget_statistical_matrix
)


@pytest.fixture
def synthetic_multibudget_logs():
    """Generates synthetic logs where Optimizer A is better early (T<=20) and Optimizer B is better late (T=50)."""
    rows = []
    # 10 tasks, 5 seeds, 50 trials per run
    for t_idx in range(1, 11):
        task_id = f"synthetic_task_{t_idx}"
        for seed in range(1, 6):
            for trial in range(1, 51):
                # Optimizer A: drops fast to cost 10 by trial 15, then stays at 10
                cost_a = max(10.0, 100.0 - trial * 6.0) + (seed * 0.1) + (t_idx * 0.05)
                
                # Optimizer B: drops slowly, at trial 15 is 40, but reaches 2.0 at trial 50
                cost_b = max(2.0, 100.0 - trial * 2.0) + (seed * 0.1) + (t_idx * 0.05)

                # Baseline: medium constant performance
                cost_base = max(25.0, 100.0 - trial * 3.0) + (seed * 0.1) + (t_idx * 0.05)
                
                rows.append({
                    "task_id": task_id,
                    "optimizer_id": "Opt_EarlyExplorer",
                    "seed": seed,
                    "n_trials": trial,
                    "trial_value__cost": cost_a,
                    "trial_value__cost_inc": cost_a,
                    "task_type": "blackbox",
                    "set": "test"
                })
                rows.append({
                    "task_id": task_id,
                    "optimizer_id": "Opt_LateRefiner",
                    "seed": seed,
                    "n_trials": trial,
                    "trial_value__cost": cost_b,
                    "trial_value__cost_inc": cost_b,
                    "task_type": "blackbox",
                    "set": "test"
                })
                rows.append({
                    "task_id": task_id,
                    "optimizer_id": "Baseline_Control",
                    "seed": seed,
                    "n_trials": trial,
                    "trial_value__cost": cost_base,
                    "trial_value__cost_inc": cost_base,
                    "task_type": "blackbox",
                    "set": "test"
                })
    return pd.DataFrame(rows)


def test_slice_logs_at_budget(synthetic_multibudget_logs):
    """Test that slicing at T=15 only keeps trials <= 15 and recomputes incumbents."""
    df_sliced = slice_logs_at_budget(synthetic_multibudget_logs, max_trial=15)
    assert df_sliced["n_trials"].max() == 15
    assert len(df_sliced) == 10 * 5 * 15 * 3  # 10 tasks * 5 seeds * 15 trials * 3 optimizers


def test_compute_ranks_at_budget(synthetic_multibudget_logs):
    """Test ranking dynamics at early vs late budgets."""
    # At T=15, Opt_EarlyExplorer should have rank 1.0 (cost 10 < 25 < 70)
    df_t15 = slice_logs_at_budget(synthetic_multibudget_logs, max_trial=15)
    ranks_t15 = compute_ranks_at_budget(df_t15)
    
    rank_a_15 = ranks_t15.loc[ranks_t15["optimizer_id"] == "Opt_EarlyExplorer", "mean_rank"].values[0]
    rank_base_15 = ranks_t15.loc[ranks_t15["optimizer_id"] == "Baseline_Control", "mean_rank"].values[0]
    rank_b_15 = ranks_t15.loc[ranks_t15["optimizer_id"] == "Opt_LateRefiner", "mean_rank"].values[0]
    assert rank_a_15 < rank_base_15 < rank_b_15
    assert np.isclose(rank_a_15, 1.0)
    assert np.isclose(rank_base_15, 2.0)
    assert np.isclose(rank_b_15, 3.0)

    # At T=50, Opt_LateRefiner should have rank 1.0 (cost 2 < 10 < 25)
    df_t50 = slice_logs_at_budget(synthetic_multibudget_logs, max_trial=50)
    ranks_t50 = compute_ranks_at_budget(df_t50)
    rank_b_50 = ranks_t50.loc[ranks_t50["optimizer_id"] == "Opt_LateRefiner", "mean_rank"].values[0]
    rank_a_50 = ranks_t50.loc[ranks_t50["optimizer_id"] == "Opt_EarlyExplorer", "mean_rank"].values[0]
    rank_base_50 = ranks_t50.loc[ranks_t50["optimizer_id"] == "Baseline_Control", "mean_rank"].values[0]
    assert rank_b_50 < rank_a_50 < rank_base_50
    assert np.isclose(rank_b_50, 1.0)
    assert np.isclose(rank_a_50, 2.0)
    assert np.isclose(rank_base_50, 3.0)


def test_evaluate_multi_budget_horizons(synthetic_multibudget_logs):
    """Test cross-horizon summary matrix evaluation."""
    horizons = [15, 25, 50]
    summary_df = evaluate_multi_budget_horizons(synthetic_multibudget_logs, budgets=horizons)
    
    assert "optimizer_id" in summary_df.columns
    assert "Rank (T=15)" in summary_df.columns
    assert "Rank (T=25)" in summary_df.columns
    assert "Rank (T=50)" in summary_df.columns
    assert len(summary_df) == 3


def test_run_statistical_tests_at_budget(synthetic_multibudget_logs):
    """Test omnibus Friedman test and Holm-corrected pairwise Wilcoxon tests."""
    df_t15 = slice_logs_at_budget(synthetic_multibudget_logs, max_trial=15)
    stat_res = run_statistical_tests_at_budget(df_t15, baseline_id="Baseline_Control")
    
    assert "friedman_p" in stat_res
    assert stat_res["friedman_p"] < 0.05  # Strong difference across approaches
    assert "pairwise_results" in stat_res
    
    # Check pairwise results against Baseline_Control
    pw_df = stat_res["pairwise_results"]
    assert len(pw_df) == 2  # 2 candidates against 1 baseline
    assert "p_holm" in pw_df.columns
    assert "r_rb" in pw_df.columns
    assert "verdict" in pw_df.columns
    
    # Opt_EarlyExplorer should be significantly better at T=15 (WIN)
    early_row = pw_df[pw_df["optimizer_id"] == "Opt_EarlyExplorer"].iloc[0]
    assert early_row["p_holm"] < 0.05
    assert early_row["r_rb"] < 0.0
    assert early_row["verdict"] == "WIN"
    
    # Opt_LateRefiner should be significantly worse at T=15 (LOSS)
    late_row = pw_df[pw_df["optimizer_id"] == "Opt_LateRefiner"].iloc[0]
    assert late_row["p_holm"] < 0.05
    assert late_row["r_rb"] > 0.0
    assert late_row["verdict"] == "LOSS"


def test_evaluate_multi_budget_statistical_matrix(synthetic_multibudget_logs):
    """Test full multi-budget statistical matrix generation across multiple horizons."""
    horizons = [15, 50]
    matrix_report = evaluate_multi_budget_statistical_matrix(
        synthetic_multibudget_logs,
        budgets=horizons,
        baseline_id="Baseline_Control"
    )
    assert 15 in matrix_report
    assert 50 in matrix_report
    assert matrix_report[15]["friedman_p"] < 0.05
    assert matrix_report[50]["friedman_p"] < 0.05


def test_cli_argument_validators_and_error_handling(tmp_path):
    """Verify CLI parser for multi-budget evaluation validates input constraints and fails on invalid inputs."""
    from scripts.evaluate_multi_budget_horizons import (
        build_parser,
        positive_int,
        non_empty_path,
        non_empty_str
    )
    import argparse

    # Test validators directly
    assert positive_int("25") == 25
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        positive_int("-10")
    with pytest.raises(argparse.ArgumentTypeError, match="Invalid integer"):
        positive_int("abc")

    assert non_empty_path("results/logs.parquet") == "results/logs.parquet"
    with pytest.raises(argparse.ArgumentTypeError, match="cannot be empty"):
        non_empty_path("")
    with pytest.raises(argparse.ArgumentTypeError, match="cannot be empty"):
        non_empty_path("   ")

    assert non_empty_str("SMAC3_HPOFacade_ei") == "SMAC3_HPOFacade_ei"
    with pytest.raises(argparse.ArgumentTypeError, match="cannot be empty"):
        non_empty_str("")
    with pytest.raises(argparse.ArgumentTypeError, match="cannot be empty"):
        non_empty_str("   ")

    parser = build_parser()

    # Invalid budget <= 0
    with pytest.raises(SystemExit):
        parser.parse_args(["--budgets", "10", "0", "50"])

    # Empty outdir
    with pytest.raises(SystemExit):
        parser.parse_args(["--outdir", ""])

    # Empty baseline
    with pytest.raises(SystemExit):
        parser.parse_args(["--baseline", ""])

    # Valid CLI arguments parsing
    args = parser.parse_args([
        "my_logs.parquet",
        "--budgets", "15", "30", "50",
        "--baseline", "MyBaseline",
        "--outdir", str(tmp_path / "reports")
    ])
    assert args.logs_path == "my_logs.parquet"
    assert args.budgets == [15, 30, 50]
    assert args.baseline == "MyBaseline"
    assert args.outdir == str(tmp_path / "reports")

    # Help text verification
    help_text = parser.format_help()
    assert "Examples:" in help_text or "usage" in help_text.lower()
    assert "--budgets" in help_text
    assert "--baseline" in help_text

