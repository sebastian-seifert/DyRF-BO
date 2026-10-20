"""Tests for the Extrapolation UQ script redesign.

Tests the 10 evaluated approaches:
- 2 Non-Proximity: hutter_total, shaker_epistemic
- 8 Proximity: prox_a_oob, prox_a_cv, prox_b_oob, prox_b_cv,
               prox_ac_oob, prox_ac_cv, prox_bc_oob, prox_bc_cv

Validates mathematical invariants, prediction intervals, symmetry,
non-negativity, 5-fold CV residual extraction and fallback, and unit consistency.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import List

import numpy as np
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq.uq_evaluator import (
    MultiUQEvaluator,
    MultiUQResult,
)


EXPECTED_APPROACHES: List[str] = [
    "hutter_total",
    "shaker_epistemic",
    "prox_a_oob",
    "prox_a_cv",
    "prox_b_oob",
    "prox_b_cv",
    "prox_ac_oob",
    "prox_ac_cv",
    "prox_bc_oob",
    "prox_bc_cv",
]

NON_PROXIMITY_APPROACHES: List[str] = [
    "hutter_total",
    "shaker_epistemic",
]

PROXIMITY_APPROACHES: List[str] = [
    "prox_a_oob",
    "prox_a_cv",
    "prox_b_oob",
    "prox_b_cv",
    "prox_ac_oob",
    "prox_ac_cv",
    "prox_bc_oob",
    "prox_bc_cv",
]


@pytest.fixture
def synthetic_data():
    """Generates synthetic 2D regression dataset."""
    rng = np.random.default_rng(42)
    N_train, D, M_test = 50, 2, 20
    X_train = rng.uniform(-1.0, 1.0, size=(N_train, D))
    y_train = np.sin(2.0 * X_train[:, 0]) + 0.5 * (X_train[:, 1] ** 2) + rng.normal(0, 0.1, size=N_train)
    X_test = rng.uniform(-1.5, 1.5, size=(M_test, D))
    return X_train, y_train, X_test


class TestExtrapolationUQRedesign:
    """Verifies all redesign requirements for MultiUQEvaluator and MultiUQResult."""

    def test_all_10_approaches_present_with_intervals_and_uncertainties(self, synthetic_data):
        X_train, y_train, X_test = synthetic_data
        M = X_test.shape[0]

        evaluator = MultiUQEvaluator(seed=42, n_trees=10, k=20, device="cpu")
        evaluator.fit(X_train, y_train)
        res = evaluator.evaluate(X_test)

        assert isinstance(res, MultiUQResult)
        assert res.y_hat.shape == (M,)
        assert np.all(np.isfinite(res.y_hat))

        # Check get_approach_names
        assert hasattr(res, "get_approach_names")
        approaches = res.get_approach_names()
        assert set(approaches) == set(EXPECTED_APPROACHES)
        assert len(approaches) == 10

        # Check all 10 approaches have u_*, int_lwr_*, int_upr_*
        for app in EXPECTED_APPROACHES:
            u_field = f"u_{app}"
            lwr_field = f"int_lwr_{app}"
            upr_field = f"int_upr_{app}"

            assert hasattr(res, u_field), f"Missing {u_field} on MultiUQResult"
            assert hasattr(res, lwr_field), f"Missing {lwr_field} on MultiUQResult"
            assert hasattr(res, upr_field), f"Missing {upr_field} on MultiUQResult"

            u_arr = getattr(res, u_field)
            lwr_arr = getattr(res, lwr_field)
            upr_arr = getattr(res, upr_field)

            assert isinstance(u_arr, np.ndarray), f"{u_field} must be np.ndarray"
            assert isinstance(lwr_arr, np.ndarray), f"{lwr_field} must be np.ndarray"
            assert isinstance(upr_arr, np.ndarray), f"{upr_field} must be np.ndarray"

            assert u_arr.shape == (M,), f"{u_field} shape mismatch: {u_arr.shape} != ({M},)"
            assert lwr_arr.shape == (M,), f"{lwr_field} shape mismatch: {lwr_arr.shape} != ({M},)"
            assert upr_arr.shape == (M,), f"{upr_field} shape mismatch: {upr_arr.shape} != ({M},)"

            assert np.all(np.isfinite(u_arr)), f"{u_field} has non-finite values"
            assert np.all(np.isfinite(lwr_arr)), f"{lwr_field} has non-finite values"
            assert np.all(np.isfinite(upr_arr)), f"{upr_field} has non-finite values"

            # Check convenience methods
            u_method = res.get_uncertainty(app)
            lwr_method, upr_method = res.get_prediction_interval(app)
            np.testing.assert_array_equal(u_method, u_arr)
            np.testing.assert_array_equal(lwr_method, lwr_arr)
            np.testing.assert_array_equal(upr_method, upr_arr)

    def test_intervals_ordering_lwr_le_upr(self, synthetic_data):
        """Assert that L_m(x) <= U_m(x) for all 10 approaches."""
        X_train, y_train, X_test = synthetic_data
        evaluator = MultiUQEvaluator(seed=42, n_trees=10, k=20, device="cpu").fit(X_train, y_train)
        res = evaluator.evaluate(X_test)

        for app in EXPECTED_APPROACHES:
            lwr, upr = res.get_prediction_interval(app)
            assert np.all(lwr <= upr + 1e-10), f"Interval violated L <= U for approach {app}"

    def test_non_proximity_intervals_symmetric(self, synthetic_data):
        """Assert that non-proximity intervals are symmetric: L = mu - u and U = mu + u."""
        X_train, y_train, X_test = synthetic_data
        evaluator = MultiUQEvaluator(seed=42, n_trees=10, k=20, device="cpu").fit(X_train, y_train)
        res = evaluator.evaluate(X_test)

        for app in NON_PROXIMITY_APPROACHES:
            u_m = res.get_uncertainty(app)
            lwr, upr = res.get_prediction_interval(app)

            np.testing.assert_allclose(
                lwr,
                res.y_hat - u_m,
                rtol=1e-6,
                atol=1e-7,
                err_msg=f"Symmetry failed for lower bound in non-proximity approach {app}",
            )
            np.testing.assert_allclose(
                upr,
                res.y_hat + u_m,
                rtol=1e-6,
                atol=1e-7,
                err_msg=f"Symmetry failed for upper bound in non-proximity approach {app}",
            )

    def test_proximity_uncertainty_equals_lower_interval_value(self, synthetic_data):
        """Assert that proximity uncertainty equals the lower interval value: u_m = mu - L_m >= 0."""
        X_train, y_train, X_test = synthetic_data
        evaluator = MultiUQEvaluator(seed=42, n_trees=10, k=20, device="cpu").fit(X_train, y_train)
        res = evaluator.evaluate(X_test)

        for app in PROXIMITY_APPROACHES:
            u_m = res.get_uncertainty(app)
            lwr, _ = res.get_prediction_interval(app)

            # u_m must be non-negative
            assert np.all(u_m >= -1e-12), f"Negative uncertainty found in proximity approach {app}: min={np.min(u_m)}"

            # u_m must equal y_hat - L_m
            lower_interval_value = res.y_hat - lwr
            np.testing.assert_allclose(
                u_m,
                lower_interval_value,
                rtol=1e-6,
                atol=1e-7,
                err_msg=f"Proximity uncertainty u_m != mu - L_m for approach {app}",
            )

    def test_five_fold_cv_residual_extraction(self, synthetic_data):
        """Assert that 5-fold CV produces valid out-of-fold residuals."""
        X_train, y_train, _ = synthetic_data
        evaluator = MultiUQEvaluator(seed=42, n_trees=10, k=20, device="cpu")
        evaluator.fit(X_train, y_train)

        assert hasattr(evaluator, "cv_res_"), "MultiUQEvaluator must store cv_res_"
        assert hasattr(evaluator, "oob_res_"), "MultiUQEvaluator must store oob_res_"

        cv_res = evaluator.cv_res_
        oob_res = evaluator.oob_res_

        assert cv_res.shape == (len(y_train),)
        assert np.all(np.isfinite(cv_res)), "cv_res_ contains non-finite values"
        assert np.var(cv_res) > 0.0, "cv_res_ must have positive variance"

        assert oob_res.shape == (len(y_train),)
        assert np.all(np.isfinite(oob_res)), "oob_res_ contains non-finite values"

    def test_cv_residual_fallback_when_n_less_than_five(self):
        """Assert that if N < 5, CV issues a warning and falls back gracefully to OOB residuals."""
        rng = np.random.default_rng(123)
        X_small = rng.uniform(-1.0, 1.0, size=(4, 2))
        y_small = rng.normal(0, 1, size=4)

        evaluator = MultiUQEvaluator(seed=123, n_trees=10, k=4, device="cpu")

        with pytest.warns(UserWarning, match=r"smaller than 5 CV folds|cv_folds"):
            evaluator.fit(X_small, y_small)

        assert hasattr(evaluator, "cv_res_")
        assert hasattr(evaluator, "oob_res_")
        np.testing.assert_allclose(evaluator.cv_res_, evaluator.oob_res_)

    def test_shaker_epistemic_uses_variance_units(self, synthetic_data):
        """Assert that shaker_epistemic uses variance units (1.96 * sigma_epistemic), NOT raw entropy bits."""
        X_train, y_train, X_test = synthetic_data
        evaluator = MultiUQEvaluator(seed=42, n_trees=10, k=20, device="cpu").fit(X_train, y_train)
        res = evaluator.evaluate(X_test)

        # Retrieve epistemic mutual information and aleatoric variance
        shaker = evaluator.shaker
        assert shaker is not None

        mi_bits = np.asarray(shaker.shaker_get_epistemic_entropy(X_test, method="huber", backend="cpu"), dtype=np.float64)
        aleatoric_var = np.asarray(shaker.base_get_aleatoric_variance(X_test), dtype=np.float64)
        sigma_aleatoric = np.sqrt(np.maximum(aleatoric_var, 1e-6))

        safe_mi = np.clip(2.0 * mi_bits, 0.0, 50.0)
        sigma_epistemic = sigma_aleatoric * np.sqrt(np.maximum(2.0 ** safe_mi - 1.0, 0.0))
        expected_u = 1.96 * sigma_epistemic

        np.testing.assert_allclose(
            res.u_shaker_epistemic,
            expected_u,
            rtol=1e-5,
            atol=1e-6,
            err_msg="u_shaker_epistemic does not match expected 1.96 * sigma_epistemic in variance units",
        )

        # Assert that it is NOT raw MI or raw entropy bits
        assert not np.allclose(res.u_shaker_epistemic, mi_bits)
        assert not np.allclose(res.u_shaker_epistemic, 1.96 * mi_bits)

    def test_backward_compatibility_aliases(self, synthetic_data):
        """Verify backward compatibility aliases on MultiUQResult."""
        X_train, y_train, X_test = synthetic_data
        evaluator = MultiUQEvaluator(seed=42, n_trees=10, k=20, device="cpu").fit(X_train, y_train)
        res = evaluator.evaluate(X_test)

        # u_hutter_total alias: u_slcb
        assert hasattr(res, "u_slcb")
        np.testing.assert_array_equal(res.u_slcb, res.u_hutter_total)

        # u_prox_a_lower alias for u_prox_a_oob
        assert hasattr(res, "u_prox_a_lower")
        np.testing.assert_array_equal(res.u_prox_a_lower, res.u_prox_a_oob)

        # u_prox_b_lower alias for u_prox_b_oob
        assert hasattr(res, "u_prox_b_lower")
        np.testing.assert_array_equal(res.u_prox_b_lower, res.u_prox_b_oob)

        # u_prox_ac_lower alias for u_prox_ac_oob
        assert hasattr(res, "u_prox_ac_lower")
        np.testing.assert_array_equal(res.u_prox_ac_lower, res.u_prox_ac_oob)

        # u_prox_bc_lower alias for u_prox_bc_oob
        assert hasattr(res, "u_prox_bc_lower")
        np.testing.assert_array_equal(res.u_prox_bc_lower, res.u_prox_bc_oob)

        # u_plcb alias for u_prox_a_oob
        assert hasattr(res, "u_plcb")
        np.testing.assert_array_equal(res.u_plcb, res.u_prox_a_oob)

    def test_fallback_evaluation_interval_unpacking(self):
        """Regression test: verify fallback predict_with_intervals does not collapse upper bound."""
        rng = np.random.default_rng(42)
        X_train = rng.uniform(-1.0, 1.0, size=(10, 2))
        y_train = np.ones(10)

        evaluator = MultiUQEvaluator(seed=42, n_trees=5, k=4, device="cpu")
        evaluator.fit(X_train, y_train)

        # Mock uq_model with predict_with_intervals returning (y_pred_lwr, y_pred, y_pred_upr, local_mae)
        class MockUQModel:
            def predict_with_intervals(self, X, **kwargs):
                n = len(X)
                y_pred = np.full(n, 1.0)
                y_pred_lwr = np.full(n, 0.4)
                y_pred_upr = np.full(n, 1.8)
                local_mae = np.full(n, 0.2)
                return y_pred_lwr, y_pred, y_pred_upr, local_mae

        evaluator.uq_model = MockUQModel()

        X_test = np.zeros((3, 2))
        res = evaluator.evaluate(X_test)

        # Upper bound must not collapse to y_hat (1.0); it must reflect y_pred_upr (1.8)
        assert np.all(res.int_upr_prox_a_oob > res.y_hat), "Upper bound collapsed to y_hat!"
        np.testing.assert_allclose(res.int_upr_prox_a_oob, np.full(3, 1.8), rtol=1e-5)
        np.testing.assert_allclose(res.int_lwr_prox_a_oob, np.full(3, 0.4), rtol=1e-5)
        assert np.all(res.int_lwr_prox_a_oob < res.int_upr_prox_a_oob)

    def test_cv_proximities_unmasked_when_oob_mask_present(self):
        """Regression test: verify valid_oob_mask is applied conditionally only to OOB, leaving CV unmasked."""
        rng = np.random.default_rng(999)
        N_train, D, M = 20, 2, 5
        X_train = rng.uniform(-1.0, 1.0, size=(N_train, D))
        y_train = rng.normal(0, 1, size=N_train)
        X_test = rng.uniform(-1.0, 1.0, size=(M, D))

        evaluator = MultiUQEvaluator(seed=999, n_trees=5, k=15, device="cpu")
        evaluator.fit(X_train, y_train)

        # Artificially set valid_oob_mask on uq_model such that only first 5 samples are valid OOB
        evaluator.uq_model.valid_oob_mask = np.zeros(N_train, dtype=bool)
        evaluator.uq_model.valid_oob_mask[:5] = True

        # Assign distinctive distinct residuals in CV so masked vs unmasked produces different quantiles
        evaluator.cv_res_ = np.linspace(-10.0, 10.0, N_train)

        res = evaluator.evaluate(X_test)

        # If CV had valid_oob_mask applied, positive residuals in samples 5..19 have zero weight,
        # collapsing upper quantile deviation of prox_b_cv to 0.0.
        # But with CV unmasked across all training samples, positive residuals have non-zero weights.
        lwr_b_cv, upr_b_cv = res.get_prediction_interval("prox_b_cv")
        delta_upr_b_cv = upr_b_cv - res.y_hat
        assert np.all(delta_upr_b_cv > 0.0), f"CV proximities were masked; delta_upr collapsed: {delta_upr_b_cv}"

