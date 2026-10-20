"""Tests for 3D Cube Aggregator (Stratum x Surrogate x Dimension)."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq.uq_evaluator import APPROACH_NAMES


@pytest.fixture
def synthetic_single_run_df():
    """Generates a synthetic single-run DataFrame matching the runner.py parquet schema."""
    rng = np.random.default_rng(42)
    n_per_stratum = 25
    n_total = n_per_stratum * 4

    strata = np.repeat([0, 1, 2, 3], n_per_stratum)
    y_true = rng.normal(0, 1, size=n_total)
    y_hat = y_true + rng.normal(0, 0.5, size=n_total)
    abs_error = np.abs(y_true - y_hat)

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
    }

    for app in APPROACH_NAMES:
        u = np.maximum(0.1, abs_error + rng.uniform(0.1, 0.5, size=n_total))
        data[f"u_{app}"] = u
        data[f"sigma_{app}"] = u / 1.96
        data[f"int_lwr_{app}"] = y_hat - u
        data[f"int_upr_{app}"] = y_hat + u
        for sfx, scale in [("50", 0.67), ("80", 1.28), ("90", 1.64), ("95", 1.96)]:
            data[f"int_lwr_{app}_{sfx}"] = y_hat - (u / 1.96) * scale
            data[f"int_upr_{app}_{sfx}"] = y_hat + (u / 1.96) * scale

    return pd.DataFrame(data)


class TestExtrapolationCubeAnalysis:
    """Verifies per-run extraction and full 3D cube aggregation."""

    def test_evaluate_single_run_df_output_shape_and_metrics(self, synthetic_single_run_df):
        from dyrf_bo.extrapolation_uq.cube_analysis import evaluate_single_run_df

        res_df = evaluate_single_run_df(synthetic_single_run_df)
        assert isinstance(res_df, pd.DataFrame)

        # 10 approaches x 5 stratum scopes ('all', 0, 1, 2, 3) = 50 rows
        assert len(res_df) == 10 * 5

        # Check required columns
        expected_cols = [
            "dimension", "n_train", "function_name", "surrogate", "seed",
            "stratum_scope", "approach",
            "ece", "winkler_mean", "winkler_95", "winkler_90", "winkler_80", "winkler_50",
            "e_aurc", "aurc", "oracle_aurc", "random_aurc", "naurc",
        ]
        for col in expected_cols:
            assert col in res_df.columns, f"Missing column {col} in run evaluation"

        # Check all metrics are non-negative and finite
        metric_cols = ["ece", "winkler_mean", "e_aurc", "aurc", "oracle_aurc", "random_aurc", "naurc"]
        for col in metric_cols:
            assert np.all(np.isfinite(res_df[col])), f"Non-finite values found in {col}"
            assert np.all(res_df[col] >= 0.0), f"Negative values found in {col}"

    def test_aggregate_3d_cube_views(self, synthetic_single_run_df):
        from dyrf_bo.extrapolation_uq.cube_analysis import evaluate_single_run_df, aggregate_3d_cube

        # Create 4 synthetic runs: 2 surrogates x 2 dimensions
        runs = []
        for d in [2, 16]:
            for surr in ["smac_default", "mature"]:
                df = synthetic_single_run_df.copy()
                df["dimension"] = d
                df["surrogate"] = surr
                runs.append(evaluate_single_run_df(df))

        all_runs_df = pd.concat(runs, ignore_index=True)
        cube_views = aggregate_3d_cube(all_runs_df)

        assert isinstance(cube_views, dict)
        expected_views = [
            "main_global",
            "stratum_global",
            "main_global_by_dimension",
            "main_global_by_surrogate",
            "stratum_by_dimension",
            "stratum_by_surrogate",
            "main_global_rf_by_dimension",
            "full_cube",
        ]
        for v in expected_views:
            assert v in cube_views, f"Missing cube view: {v}"
            view_df = cube_views[v]
            assert isinstance(view_df, pd.DataFrame)
            assert len(view_df) > 0

        # View 1: Main Global (stratum=all, surrogate=all, dimension=all) -> exactly 10 rows (1 per approach)
        mg = cube_views["main_global"]
        assert len(mg) == 10
        assert set(mg["approach"]) == set(APPROACH_NAMES)
        assert "ece_mean" in mg.columns
        assert "ece_sem" in mg.columns
        assert "winkler_mean_mean" in mg.columns
        assert "e_aurc_mean" in mg.columns
        assert "n_runs" in mg.columns
        assert (mg["n_runs"] == 4).all()
