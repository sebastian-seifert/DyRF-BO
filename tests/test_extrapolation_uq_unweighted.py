"""Test suite for Unweighted Proximity Variants and Evaluator Modes.

Validates:
1. MultiUQResult exposure of the 8 proximity fields:
   - u_prox_a_oob, u_prox_a_cv
   - u_prox_b_oob, u_prox_b_cv
   - u_prox_ac_oob, u_prox_ac_cv
   - u_prox_bc_oob, u_prox_bc_cv
2. Backward compatibility parity:
   - u_prox_a_lower == u_prox_a_oob
   - u_prox_b_lower == u_prox_b_oob
   - u_prox_ac_lower == u_prox_ac_oob
   - u_prox_bc_lower == u_prox_bc_oob
   - u_plcb == u_prox_a_oob
   - u_plcb_lower == u_prox_a_oob
3. Mathematical invariants for unweighted proximities:
   - All proximity uncertainties >= 0.0
   - u_plcb_lower >= delta_floor - 1e-12
4. Container mapping, to_dict, and get_uncertainties_dict contracts.
5. Fast eval_mode='unweighted_proximity_only' skipping Shaker computation.
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


PROXIMITY_UQ_FIELDS = (
    "u_prox_a_oob",
    "u_prox_a_cv",
    "u_prox_b_oob",
    "u_prox_b_cv",
    "u_prox_ac_oob",
    "u_prox_ac_cv",
    "u_prox_bc_oob",
    "u_prox_bc_cv",
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


class TestMultiUQResultProximityFields:
    """Verifies that MultiUQResult exposes all proximity fields with valid shapes and dtypes."""

    def test_dataclass_contract_and_attributes(self, fitted_evaluator_bundle):
        _, _, res, M = fitted_evaluator_bundle
        assert is_dataclass(res)

        for field in PROXIMITY_UQ_FIELDS:
            assert hasattr(res, field), f"Missing attribute {field} on MultiUQResult"
            arr = getattr(res, field)
            assert isinstance(arr, np.ndarray), f"{field} must be an np.ndarray"
            assert arr.shape == (M,), f"{field} shape {arr.shape} != ({M},)"
            assert np.issubdtype(arr.dtype, np.floating), f"{field} dtype {arr.dtype} not floating"
            assert np.all(np.isfinite(arr)), f"{field} contains non-finite values"

    def test_mapping_and_dictionary_access(self, fitted_evaluator_bundle):
        _, _, res, _ = fitted_evaluator_bundle
        for field in PROXIMITY_UQ_FIELDS:
            assert field in res, f"{field} not found in container"
            assert np.array_equal(res[field], getattr(res, field))

        keys = res.keys()
        for field in PROXIMITY_UQ_FIELDS:
            assert field in keys, f"{field} missing from res.keys()"

        d = res.to_dict()
        assert isinstance(d, dict)
        for field in PROXIMITY_UQ_FIELDS:
            assert field in d, f"{field} missing from res.to_dict()"
            assert np.array_equal(d[field], getattr(res, field))

        u_dict = res.get_uncertainties_dict()
        assert isinstance(u_dict, dict)
        for field in PROXIMITY_UQ_FIELDS:
            assert field in u_dict, f"{field} missing from res.get_uncertainties_dict()"
            assert np.array_equal(u_dict[field], getattr(res, field))


class TestBackwardCompatibilityParity:
    """Verifies proximity lower quantile aliases parity."""

    def test_default_unweighted_parity(self, fitted_evaluator_bundle):
        evaluator, _, res, _ = fitted_evaluator_bundle
        assert getattr(evaluator, "weighted_proximity", False) is False

        np.testing.assert_allclose(res.u_prox_a_lower, res.u_prox_a_oob, atol=1e-12)
        np.testing.assert_allclose(res.u_prox_b_lower, res.u_prox_b_oob, atol=1e-12)
        np.testing.assert_allclose(res.u_prox_ac_lower, res.u_prox_ac_oob, atol=1e-12)
        np.testing.assert_allclose(res.u_prox_bc_lower, res.u_prox_bc_oob, atol=1e-12)
        np.testing.assert_allclose(res.u_plcb, res.u_prox_a_oob, atol=1e-12)
        np.testing.assert_allclose(res.u_plcb_lower, res.u_prox_a_oob, atol=1e-12)

    def test_dual_uq_evaluator_compatibility(self, fitted_evaluator_bundle):
        evaluator, X_test, multi_res, _ = fitted_evaluator_bundle
        dual_eval = DualUQEvaluator(
            seed=evaluator.seed,
            n_trees=evaluator.n_trees,
            k=evaluator.k,
            epsilon=evaluator.epsilon,
            topological_decay_lambda=evaluator.topological_decay_lambda,
            device="cpu",
            weighted_proximity=False,
        )
        dual_eval.fit(evaluator.uq_model.X_train, evaluator.uq_model.y_train)
        dual_res = dual_eval.evaluate(X_test)

        assert isinstance(dual_res, DualUQResult)
        for field in PROXIMITY_UQ_FIELDS:
            assert hasattr(dual_res, field)
            np.testing.assert_allclose(getattr(dual_res, field), getattr(multi_res, field), atol=1e-10)

        np.testing.assert_allclose(dual_res.u_plcb, dual_res.u_prox_a_oob, atol=1e-12)


class TestUnweightedProximityInvariants:
    """Verifies theoretical mathematical invariants for unweighted proximities."""

    def test_unweighted_non_negativity_and_flooring(self, fitted_evaluator_bundle):
        _, _, res, _ = fitted_evaluator_bundle

        for field in PROXIMITY_UQ_FIELDS:
            assert np.all(getattr(res, field) >= 0.0), f"{field} has negative values"

        assert np.all(
            res.u_plcb_lower >= res.delta_floor - 1e-12
        ), "u_plcb_lower failed delta_floor lower bound!"

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
        for field in PROXIMITY_UQ_FIELDS:
            assert hasattr(res, field)
            arr = getattr(res, field)
            assert arr.shape == (10,)
            assert np.all(np.isfinite(arr))
            assert np.all(arr >= 0.0)


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

        # Proximity fields must be computed and finite
        for f in PROXIMITY_UQ_FIELDS:
            arr = getattr(res, f)
            assert np.all(np.isfinite(arr)), f"{f} contains non-finite values"
            assert np.all(arr >= 0.0), f"{f} contains negative values"

        np.testing.assert_allclose(res.u_slcb, res.u_hutter_total, atol=1e-12)
        np.testing.assert_allclose(res.u_plcb, res.u_prox_a_oob, atol=1e-12)

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
        np.testing.assert_allclose(res.u_plcb, res.u_prox_a_oob, atol=1e-12)

    def test_invalid_eval_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown eval_mode"):
            MultiUQEvaluator(eval_mode="invalid_mode")
