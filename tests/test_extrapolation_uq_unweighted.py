"""Test suite for Milestone 2: Extrapolation UQ Evaluator, MultiUQResult, Runner & Metrics Pipeline.

Validates:
1. MultiUQResult exposure of 8 new fields:
   - u_prox_a_unweighted_half, u_prox_a_unweighted_lower
   - u_plcb_unweighted_half, u_plcb_unweighted_lower
   - u_prox_a_weighted_half, u_prox_a_weighted_lower
   - u_plcb_weighted_half, u_plcb_weighted_lower
2. 100% backward compatibility parity:
   - u_prox_a_half == u_prox_a_weighted_half
   - u_prox_a_lower == u_prox_a_weighted_lower
   - u_plcb_half == u_plcb_weighted_half
   - u_plcb_lower == u_plcb_weighted_lower
3. Mathematical invariants for unweighted proximities:
   - u_prox_a_unweighted_half >= 0.0, u_prox_a_unweighted_lower >= 0.0
   - u_plcb_unweighted_lower >= delta_floor - 1e-12
   - u_plcb_unweighted_lower >= u_prox_a_unweighted_lower - 1e-12
   - u_plcb_unweighted_half == u_prox_a_unweighted_half
4. Serialization, mapping, to_dict, get_uncertainties_dict contracts.
5. runner.py DataFrame export and Parquet schema parity.
6. scripts/aggregate_extrapolation_results.py standard_methods and method_column_map inclusion.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import is_dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq.uq_evaluator import (
    DualUQEvaluator,
    DualUQResult,
    MultiUQEvaluator,
    MultiUQResult,
)
from dyrf_bo.extrapolation_uq.runner import ExtrapolationRunConfig, run_single_experiment


NEW_UQ_FIELDS = (
    "u_prox_a_unweighted_half",
    "u_prox_a_unweighted_lower",
    "u_plcb_unweighted_half",
    "u_plcb_unweighted_lower",
    "u_prox_a_weighted_half",
    "u_prox_a_weighted_lower",
    "u_plcb_weighted_half",
    "u_plcb_weighted_lower",
)


@pytest.fixture
def fitted_evaluator_bundle():
    np.random.seed(42)
    N, D, M = 50, 3, 25
    X_train = np.random.uniform(-0.5, 0.5, size=(N, D))
    y_train = np.sin(X_train[:, 0]) + 0.5 * (X_train[:, 1] ** 2) + 0.05 * np.random.randn(N)
    evaluator = MultiUQEvaluator(
        seed=42,
        n_trees=10,
        k=28,
        epsilon=0.080791,
        topological_decay_lambda=0.20486,
        device="cpu",
    )
    evaluator.fit(X_train, y_train)
    X_test = np.random.uniform(-1.0, 1.0, size=(M, D))
    res = evaluator.evaluate(X_test)
    return evaluator, X_test, res, M


class TestMultiUQResultNewFields:
    """Verifies that MultiUQResult exposes the 8 new fields with valid shapes and dtypes."""

    def test_dataclass_contract_and_attributes(self, fitted_evaluator_bundle):
        _, _, res, M = fitted_evaluator_bundle
        assert is_dataclass(res)

        for field in NEW_UQ_FIELDS:
            assert hasattr(res, field), f"Missing attribute {field} on MultiUQResult"
            arr = getattr(res, field)
            assert isinstance(arr, np.ndarray), f"{field} must be an np.ndarray"
            assert arr.shape == (M,), f"{field} shape {arr.shape} != ({M},)"
            assert np.issubdtype(arr.dtype, np.floating), f"{field} dtype {arr.dtype} not floating"
            assert np.all(np.isfinite(arr)), f"{field} contains non-finite values"

    def test_mapping_and_dictionary_access(self, fitted_evaluator_bundle):
        _, _, res, _ = fitted_evaluator_bundle
        # Dictionary style __getitem__
        for field in NEW_UQ_FIELDS:
            assert field in res, f"{field} not found in container"
            assert np.array_equal(res[field], getattr(res, field))

        # keys() method
        keys = res.keys()
        for field in NEW_UQ_FIELDS:
            assert field in keys, f"{field} missing from res.keys()"

        # to_dict() method
        d = res.to_dict()
        assert isinstance(d, dict)
        for field in NEW_UQ_FIELDS:
            assert field in d, f"{field} missing from res.to_dict()"
            assert np.array_equal(d[field], getattr(res, field))

        # get_uncertainties_dict() method
        u_dict = res.get_uncertainties_dict()
        assert isinstance(u_dict, dict)
        for field in NEW_UQ_FIELDS:
            assert field in u_dict, f"{field} missing from res.get_uncertainties_dict()"
            assert np.array_equal(u_dict[field], getattr(res, field))


class TestBackwardCompatibilityParity:
    """Verifies strict 100% backward compatibility parity between legacy and weighted fields."""

    def test_exact_weighted_parity(self, fitted_evaluator_bundle):
        _, _, res, _ = fitted_evaluator_bundle

        # u_prox_a_half == u_prox_a_weighted_half
        np.testing.assert_allclose(
            res.u_prox_a_half,
            res.u_prox_a_weighted_half,
            atol=1e-12,
            err_msg="u_prox_a_half does not match u_prox_a_weighted_half",
        )

        # u_prox_a_lower == u_prox_a_weighted_lower
        np.testing.assert_allclose(
            res.u_prox_a_lower,
            res.u_prox_a_weighted_lower,
            atol=1e-12,
            err_msg="u_prox_a_lower does not match u_prox_a_weighted_lower",
        )

        # u_plcb_half == u_plcb_weighted_half
        np.testing.assert_allclose(
            res.u_plcb_half,
            res.u_plcb_weighted_half,
            atol=1e-12,
            err_msg="u_plcb_half does not match u_plcb_weighted_half",
        )

        # u_plcb_lower == u_plcb_weighted_lower
        np.testing.assert_allclose(
            res.u_plcb_lower,
            res.u_plcb_weighted_lower,
            atol=1e-12,
            err_msg="u_plcb_lower does not match u_plcb_weighted_lower",
        )

        # Legacy alias u_plcb == u_plcb_lower == u_plcb_weighted_lower
        np.testing.assert_allclose(res.u_plcb, res.u_plcb_weighted_lower, atol=1e-12)

    def test_dual_uq_evaluator_compatibility(self, fitted_evaluator_bundle):
        evaluator, X_test, multi_res, _ = fitted_evaluator_bundle
        dual_eval = DualUQEvaluator(
            seed=evaluator.seed,
            n_trees=evaluator.n_trees,
            k=evaluator.k,
            epsilon=evaluator.epsilon,
            topological_decay_lambda=evaluator.topological_decay_lambda,
            device="cpu",
        )
        dual_eval.fit(evaluator.uq_model.X_train, evaluator.uq_model.y_train)
        dual_res = dual_eval.evaluate(X_test)

        assert isinstance(dual_res, DualUQResult)
        for field in NEW_UQ_FIELDS:
            assert hasattr(dual_res, field)
            np.testing.assert_allclose(getattr(dual_res, field), getattr(multi_res, field), atol=1e-10)


class TestUnweightedProximityInvariants:
    """Verifies theoretical mathematical invariants for unweighted proximities."""

    def test_unweighted_non_negativity_and_flooring(self, fitted_evaluator_bundle):
        _, _, res, _ = fitted_evaluator_bundle

        # Non-negativity
        assert np.all(res.u_prox_a_unweighted_half >= 0.0)
        assert np.all(res.u_prox_a_unweighted_lower >= 0.0)
        assert np.all(res.u_plcb_unweighted_half >= 0.0)
        assert np.all(res.u_plcb_unweighted_lower >= 0.0)

        # Half-width identity
        np.testing.assert_allclose(
            res.u_plcb_unweighted_half,
            res.u_prox_a_unweighted_half,
            atol=1e-12,
            err_msg="u_plcb_unweighted_half must equal u_prox_a_unweighted_half",
        )

        # Exploration floor enforcement: u_plcb_unweighted_lower >= delta_floor
        assert np.all(
            res.u_plcb_unweighted_lower >= res.delta_floor - 1e-12
        ), "u_plcb_unweighted_lower failed delta_floor lower bound!"

        # Lower bound dominance: u_plcb_unweighted_lower >= u_prox_a_unweighted_lower
        assert np.all(
            res.u_plcb_unweighted_lower >= res.u_prox_a_unweighted_lower - 1e-12
        ), "u_plcb_unweighted_lower must dominate u_prox_a_unweighted_lower everywhere!"

    def test_fallback_branch_coverage(self):
        """Verify fallback branch in evaluate() when uq_model does not have tree_leaf_distances."""
        evaluator = MultiUQEvaluator(
            seed=42, n_trees=5, k=10, topological_decay_lambda=None, device="cpu"
        )
        X_train = np.random.uniform(-0.5, 0.5, size=(30, 2))
        y_train = np.sin(X_train[:, 0]) + 0.1 * np.random.randn(30)
        evaluator.fit(X_train, y_train)

        X_test = np.random.uniform(-1.0, 1.0, size=(10, 2))
        res = evaluator.evaluate(X_test)
        for field in NEW_UQ_FIELDS:
            assert hasattr(res, field)
            arr = getattr(res, field)
            assert arr.shape == (10,)
            assert np.all(np.isfinite(arr))
            assert np.all(arr >= 0.0)


class TestRunnerDataFrameExportNewColumns:
    """Verifies that runner.py exports all 8 new fields in data_dict and Parquet."""

    def test_run_single_experiment_dataframe_contains_new_columns(self):
        cfg = ExtrapolationRunConfig(
            dimension=2,
            n_train=25,
            n_test=12,
            function_name="sphere",
            sampling_strategy="natural",
            n_trees=5,
            surrogate_type="smac_default",
            seed=42,
        )
        summary, point_df = run_single_experiment(cfg, save_parquet=False)

        assert hasattr(point_df, "columns")
        for field in NEW_UQ_FIELDS:
            assert field in point_df.columns, f"Column '{field}' missing from runner point_df!"
            series = point_df[field]
            assert np.all(np.isfinite(series)), f"Non-finite values in runner column '{field}'"
            assert np.all(series >= 0.0), f"Negative values in runner column '{field}'"

        # Parquet round-trip verification
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "test_run.parquet"
            summary2, point_df2 = run_single_experiment(cfg, output_dir=out_path, save_parquet=True)
            assert out_path.exists()
            import pandas as pd
            loaded_df = pd.read_parquet(out_path)
            for field in NEW_UQ_FIELDS:
                assert field in loaded_df.columns
                np.testing.assert_allclose(loaded_df[field].values, point_df2[field].values, atol=1e-12)


class TestAggregateExtrapolationResultsConfiguration:
    """Verifies scripts/aggregate_extrapolation_results.py has unweighted methods configured."""

    def test_aggregate_script_contains_unweighted_methods(self):
        import scripts.aggregate_extrapolation_results as agg

        # Verify SUPPORTED_ESTIMATORS
        assert hasattr(agg, "SUPPORTED_ESTIMATORS")
        assert "prox_a_unweighted" in agg.SUPPORTED_ESTIMATORS
        assert "plcb_unweighted" in agg.SUPPORTED_ESTIMATORS

        # Verify ESTIMATOR_ALIASES and method_column_map
        assert hasattr(agg, "ESTIMATOR_ALIASES")
        assert "prox_a_unweighted" in agg.ESTIMATOR_ALIASES
        assert "plcb_unweighted" in agg.ESTIMATOR_ALIASES

        assert hasattr(agg, "method_column_map")
        assert agg.method_column_map is agg.ESTIMATOR_ALIASES

        # Verify build_uq_ablation_scorecard_dataframe standard_methods
        import inspect
        src = inspect.getsource(agg.build_uq_ablation_scorecard_dataframe)
        assert "prox_a_unweighted" in src
        assert "plcb_unweighted" in src


class TestEvalModeUnweightedProximityOnly:
    """Verifies the fast evaluation mode skipping Shaker and focusing on unweighted proximity."""

    def test_unweighted_proximity_only_skips_shaker_and_evaluates(self):
        np.random.seed(42)
        N, D, M = 40, 2, 15
        X_train = np.random.uniform(-0.5, 0.5, size=(N, D))
        y_train = np.sin(X_train[:, 0]) + 0.1 * np.random.randn(N)
        evaluator = MultiUQEvaluator(
            seed=42,
            n_trees=5,
            k=10,
            eval_mode="unweighted_proximity_only",
            device="cpu",
        )
        assert evaluator.eval_mode == "unweighted_proximity_only"
        evaluator.fit(X_train, y_train)

        # In unweighted_proximity_only mode, Shaker is NOT fitted or instantiated
        assert evaluator.shaker is None

        X_test = np.random.uniform(-1.0, 1.0, size=(M, D))
        res = evaluator.evaluate(X_test)

        # Shaker fields must be zeros
        np.testing.assert_array_equal(res.u_shaker_epistemic, np.zeros(M))
        np.testing.assert_array_equal(res.u_shaker_total, np.zeros(M))
        np.testing.assert_array_equal(res.shaker_mi, np.zeros(M))
        np.testing.assert_array_equal(res.shaker_total_entropy, np.zeros(M))

        # Hutter fields must be computed and finite
        assert np.all(np.isfinite(res.u_hutter_total))
        assert np.all(res.u_hutter_total > 0.0)
        assert np.all(np.isfinite(res.y_hat))

        # Unweighted & weighted proximity fields must be computed and finite
        for f in (
            "u_prox_a_unweighted_half",
            "u_prox_a_unweighted_lower",
            "u_plcb_unweighted_half",
            "u_plcb_unweighted_lower",
            "u_prox_a_weighted_half",
            "u_prox_a_weighted_lower",
            "u_plcb_weighted_half",
            "u_plcb_weighted_lower",
            "u_prox_a_half",
            "u_prox_a_lower",
            "u_plcb_half",
            "u_plcb_lower",
        ):
            arr = getattr(res, f)
            assert np.all(np.isfinite(arr)), f"{f} contains non-finite values"
            assert np.all(arr >= 0.0), f"{f} contains negative values"

        # u_slcb = u_hutter_total and u_plcb = u_plcb_unweighted_lower
        np.testing.assert_allclose(res.u_slcb, res.u_hutter_total, atol=1e-12)
        np.testing.assert_allclose(res.u_plcb, res.u_plcb_unweighted_lower, atol=1e-12)

    def test_dual_uq_evaluator_unweighted_proximity_only(self):
        np.random.seed(42)
        N, D, M = 30, 2, 10
        X_train = np.random.uniform(-0.5, 0.5, size=(N, D))
        y_train = np.sin(X_train[:, 0]) + 0.1 * np.random.randn(N)
        dual_eval = DualUQEvaluator(
            seed=42,
            n_trees=5,
            k=10,
            eval_mode="unweighted_proximity_only",
            device="cpu",
        )
        assert dual_eval.eval_mode == "unweighted_proximity_only"
        dual_eval.fit(X_train, y_train)
        assert dual_eval.shaker is None

        X_test = np.random.uniform(-1.0, 1.0, size=(M, D))
        res = dual_eval.evaluate(X_test)
        assert isinstance(res, DualUQResult)
        np.testing.assert_allclose(res.u_slcb, res.u_hutter_total, atol=1e-12)
        np.testing.assert_allclose(res.u_plcb, res.u_plcb_unweighted_lower, atol=1e-12)

    def test_invalid_eval_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown eval_mode"):
            MultiUQEvaluator(eval_mode="invalid_mode")

    def test_runner_and_cli_eval_mode(self):
        from scripts.run_extrapolation_experiment import build_parser

        parser = build_parser()
        args1 = parser.parse_args([
            "--dimension", "2",
            "--n-train", "25",
            "--function", "sphere",
            "--strategy", "natural",
            "--seed", "42",
            "--eval-mode", "unweighted_proximity_only",
        ])
        assert args1.eval_mode == "unweighted_proximity_only"

        args2 = parser.parse_args([
            "--dimension", "2",
            "--n-train", "25",
            "--function", "sphere",
            "--strategy", "natural",
            "--seed", "42",
            "--unweighted-proximity-only",
        ])
        assert args2.unweighted_proximity_only is True

        cfg = ExtrapolationRunConfig(
            dimension=2,
            n_train=25,
            n_test=10,
            function_name="sphere",
            sampling_strategy="natural",
            n_trees=5,
            surrogate_type="smac_default",
            seed=42,
            eval_mode="unweighted_proximity_only",
        )
        assert cfg.eval_mode == "unweighted_proximity_only"
        summary, point_df = run_single_experiment(cfg, save_parquet=False)
        assert summary["eval_mode"] == "unweighted_proximity_only"
        np.testing.assert_array_equal(point_df["u_shaker_epistemic"].values, np.zeros(10))
        np.testing.assert_allclose(
            point_df["u_plcb"].values,
            point_df["u_plcb_unweighted_lower"].values,
            atol=1e-12,
        )

