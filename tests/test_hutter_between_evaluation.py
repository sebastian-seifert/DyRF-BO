"""Test suite for between-tree disagreement evaluation and cube aggregation."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq.cube_analysis import (
    METRIC_COLUMNS,
    evaluate_single_run_between_tree,
    aggregate_between_tree_slices,
)


@pytest.fixture
def sample_raw_run_df():
    """Generates a synthetic single-run DataFrame matching the runner.py raw parquet schema."""
    rng = np.random.default_rng(123)
    n_per_stratum = 20
    n_total = n_per_stratum * 4

    strata = np.repeat([0, 1, 2, 3], n_per_stratum)
    y_true = rng.normal(0, 1, size=n_total)
    y_hat = y_true + rng.normal(0, 0.4, size=n_total)
    abs_error = np.abs(y_true - y_hat)
    u_between = np.maximum(0.05, abs_error + rng.uniform(0.1, 0.3, size=n_total))

    data = {
        "point_id": np.arange(n_total, dtype=np.int64),
        "dimension": np.full(n_total, 2, dtype=np.int64),
        "n_train": np.full(n_total, 34, dtype=np.int64),
        "function_name": np.full(n_total, "sphere", dtype=object),
        "seed": np.full(n_total, 0, dtype=np.int64),
        "surrogate": np.full(n_total, "smac_default", dtype=object),
        "stratum": strata,
        "d_norm": strata * 0.25,
        "d_rel": strata * 0.25,
        "d_inf": strata * 0.25,
        "is_interpolating": strata == 0,
        "y_true": y_true,
        "y_hat": y_hat,
        "abs_error": abs_error,
        "u_hutter_between": u_between,
    }
    return pd.DataFrame(data)


def test_evaluate_single_run_between_tree(sample_raw_run_df):
    eval_df = evaluate_single_run_between_tree(sample_raw_run_df)
    
    # 5 stratum scopes: 'all', 0, 1, 2, 3 -> exactly 5 rows
    assert len(eval_df) == 5
    assert set(eval_df["approach"].unique()) == {"hutter_between"}
    assert set(eval_df["stratum_scope"].unique()) == {"all", "0", "1", "2", "3"}

    # Check all metric columns exist and are finite numbers
    for col in METRIC_COLUMNS:
        assert col in eval_df.columns
        assert not eval_df[col].isna().any()
        assert np.all(np.isfinite(eval_df[col].to_numpy()))


def test_aggregate_between_tree_slices(sample_raw_run_df):
    eval_df1 = evaluate_single_run_between_tree(sample_raw_run_df)
    
    # Create second run with different dimension and surrogate
    df2 = sample_raw_run_df.copy()
    df2["dimension"] = 5
    df2["surrogate"] = "mature"
    df2["seed"] = 1
    eval_df2 = evaluate_single_run_between_tree(df2)

    combined = pd.concat([eval_df1, eval_df2], ignore_index=True)
    slices = aggregate_between_tree_slices(combined)

    expected_slices = [
        "main_global",
        "stratum_global",
        "main_global_by_dimension",
        "main_global_by_surrogate",
        "stratum_by_dimension",
        "stratum_by_surrogate",
        "main_global_rf_by_dimension",
    ]
    for s_name in expected_slices:
        assert s_name in slices
        sdf = slices[s_name]
        assert len(sdf) > 0
        assert (sdf["approach"] == "hutter_between").all()
        assert "ece_mean" in sdf.columns
        assert "winkler_mean_mean" in sdf.columns
        assert "e_aurc_mean" in sdf.columns
        assert "n_runs" in sdf.columns
