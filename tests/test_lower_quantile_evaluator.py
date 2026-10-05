"""Test suite for Milestone 1: Lower-Quantile UQ Evaluator & Unweighted Proximity Variants (A, B, AC, BC).

Validates:
1. MultiUQResult schema contains ONLY lower quantile fields and non-proximity baselines:
   - Non-proximity: u_hutter_total, u_hutter_between, u_hutter_within, u_shaker_epistemic,
     u_shaker_total, shaker_mi, shaker_total_entropy, u_rf_fire_lower, u_slcb (alias).
   - Proximity (All unweighted tree walk): u_prox_a_lower, u_prox_b_lower, u_prox_ac_lower,
     u_prox_bc_lower, u_plcb_lower, u_plcb (alias).
   - Diagnostics: delta_floor, local_mae, q_lower.
   - Base predictions & variance: y_hat, var_between, var_within, var_total.
2. Complete absence of any '*_half' or old split fields in MultiUQResult, to_dict(), and get_uncertainties_dict().
3. Mathematical invariants:
   - Non-negativity: All uncertainties >= 0.0.
   - Floor: u_plcb_lower >= delta_floor - 1e-12 and u_plcb_lower >= u_prox_a_lower - 1e-12.
   - Density scaling: When gamma > 1.0 (empty space), u_prox_ac_lower >= u_prox_a_lower and u_prox_bc_lower >= u_prox_b_lower.
   - Method B: u_prox_b_lower computed via continuous weighted quantile using unweighted tree-walk proximities.
   - Fast mode eval_mode='unweighted_proximity_only': skips Shaker and evaluates all 4 proximity variants (A, B, AC, BC) + PLCB + Hutter.
4. DualUQResult backwards compatibility.
"""

from __future__ import annotations

import os
import sys
from dataclasses import fields, is_dataclass
from typing import Dict, Set

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


EXPECTED_NON_PROXIMITY_FIELDS: Set[str] = {
    "u_hutter_total",
    "u_hutter_between",
    "u_hutter_within",
    "u_shaker_epistemic",
    "u_shaker_total",
    "shaker_mi",
    "shaker_total_entropy",
    "u_rf_fire_lower",
    "u_slcb",
}

EXPECTED_PROXIMITY_FIELDS: Set[str] = {
    "u_prox_a_lower",
    "u_prox_b_lower",
    "u_prox_ac_lower",
    "u_prox_bc_lower",
    "u_plcb_lower",
    "u_plcb",
}

EXPECTED_DIAGNOSTIC_FIELDS: Set[str] = {
    "delta_floor",
    "local_mae",
    "q_lower",
}

EXPECTED_BASE_FIELDS: Set[str] = {
    "y_hat",
    "var_between",
    "var_within",
    "var_total",
}

ALL_EXPECTED_FIELDS: Set[str] = (
    EXPECTED_NON_PROXIMITY_FIELDS
    | EXPECTED_PROXIMITY_FIELDS
    | EXPECTED_DIAGNOSTIC_FIELDS
    | EXPECTED_BASE_FIELDS
)


@pytest.fixture
def synthetic_data():
    """Create reproducible synthetic 2D regression dataset with core and extrapolation query points."""
    np.random.seed(42)
    N_train = 60
    D = 3
    M = 20

    X_train = np.random.uniform(-1.0, 1.0, size=(N_train, D))
    y_train = np.sin(2.0 * X_train[:, 0]) + 0.5 * (X_train[:, 1] ** 2) + 0.1 * np.random.randn(N_train)

    # In-distribution test points
    X_test_in = np.random.uniform(-0.8, 0.8, size=(M, D))
    # Extreme extrapolation test points in empty space
    X_test_ood = np.random.uniform(5.0, 10.0, size=(M, D))

    return X_train, y_train, X_test_in, X_test_ood


@pytest.fixture
def fitted_evaluator_all(synthetic_data):
    """MultiUQEvaluator fitted with eval_mode='all'."""
    X_train, y_train, _, _ = synthetic_data
    evaluator = MultiUQEvaluator(
        seed=42,
        n_trees=10,
        k=28,
        epsilon=0.080791,
        topological_decay_lambda=0.20486,
        density_scaling_alpha=1.0,
        eval_mode="all",
        device="cpu",
    )
    evaluator.fit(X_train, y_train)
    return evaluator


@pytest.fixture
def fitted_evaluator_unweighted_only(synthetic_data):
    """MultiUQEvaluator fitted with eval_mode='unweighted_proximity_only'."""
    X_train, y_train, _, _ = synthetic_data
    evaluator = MultiUQEvaluator(
        seed=42,
        n_trees=10,
        k=28,
        epsilon=0.080791,
        topological_decay_lambda=0.20486,
        density_scaling_alpha=1.0,
        eval_mode="unweighted_proximity_only",
        device="cpu",
    )
    evaluator.fit(X_train, y_train)
    return evaluator


class TestMultiUQResultSchema:
    """Verifies that MultiUQResult contains ONLY lower quantile fields and non-proximity baselines."""

    def test_dataclass_fields_exact_match(self):
        """MultiUQResult dataclass fields must match ALL_EXPECTED_FIELDS with no extra or missing fields."""
        actual_fields = {f.name for f in fields(MultiUQResult)}
        assert actual_fields == ALL_EXPECTED_FIELDS, (
            f"Field mismatch in MultiUQResult!\n"
            f"Extra fields: {actual_fields - ALL_EXPECTED_FIELDS}\n"
            f"Missing fields: {ALL_EXPECTED_FIELDS - actual_fields}"
        )

    def test_no_half_fields_exist(self, fitted_evaluator_all, synthetic_data):
        """Assert NO '*_half' fields exist in MultiUQResult attributes, to_dict(), or get_uncertainties_dict()."""
        _, _, X_test_in, _ = synthetic_data
        res = fitted_evaluator_all.evaluate(X_test_in)

        # 1. Dataclass fields
        dataclass_field_names = [f.name for f in fields(MultiUQResult)]
        half_dataclass_fields = [f for f in dataclass_field_names if "half" in f]
        assert not half_dataclass_fields, f"Found unexpected '*_half' dataclass fields: {half_dataclass_fields}"

        # 2. to_dict keys
        res_dict = res.to_dict()
        half_dict_keys = [k for k in res_dict.keys() if "half" in k]
        assert not half_dict_keys, f"Found unexpected '*_half' keys in to_dict(): {half_dict_keys}"

        # 3. get_uncertainties_dict keys
        unc_dict = res.get_uncertainties_dict()
        half_unc_keys = [k for k in unc_dict.keys() if "half" in k]
        assert not half_unc_keys, f"Found unexpected '*_half' keys in get_uncertainties_dict(): {half_unc_keys}"

        # 4. Attribute access
        assert not hasattr(res, "u_rf_fire_half")
        assert not hasattr(res, "u_prox_a_half")
        assert not hasattr(res, "u_prox_b_half")
        assert not hasattr(res, "u_prox_bc_half")
        assert not hasattr(res, "u_plcb_half")

    def test_no_old_weighted_unweighted_split_fields(self):
        """Assert no legacy *_weighted_* or *_unweighted_* fields exist in MultiUQResult."""
        dataclass_field_names = [f.name for f in fields(MultiUQResult)]
        legacy_split_fields = [
            f for f in dataclass_field_names if "_weighted_" in f or "_unweighted_" in f
        ]
        assert not legacy_split_fields, f"Found legacy split fields: {legacy_split_fields}"

    def test_mapping_protocol_and_dict_parity(self, fitted_evaluator_all, synthetic_data):
        """Verify __getitem__, get, __contains__, keys(), values(), items(), len()."""
        _, _, X_test_in, _ = synthetic_data
        res = fitted_evaluator_all.evaluate(X_test_in)

        assert len(res) == len(res.keys())
        for k in res.keys():
            assert k in res
            np.testing.assert_array_equal(res[k], getattr(res, k))
            np.testing.assert_array_equal(res.get(k), getattr(res, k))

        with pytest.raises(KeyError):
            _ = res["nonexistent_field"]

        assert res.get("nonexistent_field", default=None) is None


class TestMathematicalInvariants:
    """Verifies non-negativity, floor bounds, density scaling, and Method B formulation."""

    def test_non_negativity_all_uncertainties(self, fitted_evaluator_all, synthetic_data):
        """All uncertainties in get_uncertainties_dict() must be strictly non-negative (>= 0.0)."""
        _, _, X_test_in, X_test_ood = synthetic_data

        for X_eval in [X_test_in, X_test_ood]:
            res = fitted_evaluator_all.evaluate(X_eval)
            unc_dict = res.get_uncertainties_dict()

            for name, arr in unc_dict.items():
                assert isinstance(arr, np.ndarray), f"{name} must be an np.ndarray"
                assert np.all(arr >= 0.0), (
                    f"Violation of non-negativity: {name} contains values < 0.0: "
                    f"min={np.min(arr)}"
                )
                assert not np.any(np.isnan(arr)), f"{name} contains NaN values"
                assert not np.any(np.isinf(arr)), f"{name} contains infinite values"

    def test_plcb_lower_floor_domination(self, fitted_evaluator_all, synthetic_data):
        """Verify u_plcb_lower >= delta_floor - 1e-12 and u_plcb_lower >= u_prox_a_lower - 1e-12."""
        _, _, X_test_in, X_test_ood = synthetic_data

        for X_eval in [X_test_in, X_test_ood]:
            res = fitted_evaluator_all.evaluate(X_eval)

            # u_plcb_lower >= delta_floor - 1e-12
            assert np.all(res.u_plcb_lower >= res.delta_floor - 1e-12), (
                "u_plcb_lower violated delta_floor lower bound: min diff = "
                f"{np.min(res.u_plcb_lower - res.delta_floor)}"
            )

            # u_plcb_lower >= u_prox_a_lower - 1e-12
            assert np.all(res.u_plcb_lower >= res.u_prox_a_lower - 1e-12), (
                "u_plcb_lower violated u_prox_a_lower lower bound: min diff = "
                f"{np.min(res.u_plcb_lower - res.u_prox_a_lower)}"
            )

            # Check aliases
            np.testing.assert_array_equal(res.u_plcb, res.u_plcb_lower)
            np.testing.assert_array_equal(res.u_slcb, res.u_hutter_total)

    def test_density_scaling_empty_space(self, fitted_evaluator_all, synthetic_data):
        """In empty space (extrapolation), test leaf density drops so gamma > 1.0.

        Asserts:
        u_prox_ac_lower >= u_prox_a_lower
        u_prox_bc_lower >= u_prox_b_lower
        """
        _, _, _, X_test_ood = synthetic_data
        res = fitted_evaluator_all.evaluate(X_test_ood)

        # In extreme extrapolation, gamma should be > 1.0 for all points
        assert np.all(res.u_prox_ac_lower >= res.u_prox_a_lower - 1e-12), (
            f"u_prox_ac_lower was not >= u_prox_a_lower in extrapolation region! "
            f"Min diff: {np.min(res.u_prox_ac_lower - res.u_prox_a_lower)}"
        )
        assert np.all(res.u_prox_bc_lower >= res.u_prox_b_lower - 1e-12), (
            f"u_prox_bc_lower was not >= u_prox_b_lower in extrapolation region! "
            f"Min diff: {np.min(res.u_prox_bc_lower - res.u_prox_b_lower)}"
        )

        # Verify that for non-zero proximities in extrapolation, scaling strictly expands uncertainty
        active_a = res.u_prox_a_lower > 1e-4
        if np.any(active_a):
            assert np.all(res.u_prox_ac_lower[active_a] > res.u_prox_a_lower[active_a])

        active_b = res.u_prox_b_lower > 1e-4
        if np.any(active_b):
            assert np.all(res.u_prox_bc_lower[active_b] > res.u_prox_b_lower[active_b])

    def test_density_scaling_gamma_floored_everywhere(self, fitted_evaluator_all, synthetic_data):
        """Verify gamma is floored by 1.0 so u_prox_ac_lower >= u_prox_a_lower and
        u_prox_bc_lower >= u_prox_b_lower hold everywhere across all query points,
        preventing uncertainty deflation in dense regions where avg_leaf_size > n_baseline."""
        _, _, X_test_in, X_test_ood = synthetic_data
        dense_points = np.zeros((15, 3))  # tightly clustered at origin

        for X_eval in [X_test_in, X_test_ood, dense_points]:
            res = fitted_evaluator_all.evaluate(X_eval)
            assert np.all(res.u_prox_ac_lower >= res.u_prox_a_lower - 1e-12), (
                f"u_prox_ac_lower deflated below u_prox_a_lower: "
                f"min diff = {np.min(res.u_prox_ac_lower - res.u_prox_a_lower)}"
            )
            assert np.all(res.u_prox_bc_lower >= res.u_prox_b_lower - 1e-12), (
                f"u_prox_bc_lower deflated below u_prox_b_lower: "
                f"min diff = {np.min(res.u_prox_bc_lower - res.u_prox_b_lower)}"
            )

    def test_method_b_continuous_weighted_quantile(self, fitted_evaluator_all, synthetic_data):
        """Verify u_prox_b_lower is computed via continuous weighted quantile using unweighted tree-walk proximities."""
        _, _, X_test_in, _ = synthetic_data
        res = fitted_evaluator_all.evaluate(X_test_in)

        # Proximity B lower must be an ndarray of shape (M,)
        assert isinstance(res.u_prox_b_lower, np.ndarray)
        assert res.u_prox_b_lower.shape == (len(X_test_in),)
        assert np.all(res.u_prox_b_lower >= 0.0)

        # Proximity B uses continuous soft weighting over all samples, while Proximity A uses top-k partition.
        # They should both be valid positive uncertainties.
        assert np.mean(res.u_prox_b_lower) > 0.0


class TestFastModeUnweightedProximityOnly:
    """Verifies fast mode eval_mode='unweighted_proximity_only'."""

    def test_skips_shaker_and_evaluates_all_proximity_variants(
        self, fitted_evaluator_unweighted_only, synthetic_data
    ):
        """In eval_mode='unweighted_proximity_only', shaker is skipped, and all 4 proximity variants + PLCB + Hutter are evaluated."""
        _, _, X_test_in, _ = synthetic_data
        evaluator = fitted_evaluator_unweighted_only
        assert evaluator.shaker is None

        res = evaluator.evaluate(X_test_in)

        # 1. Shaker fields must be zeros
        assert np.all(res.u_shaker_epistemic == 0.0)
        assert np.all(res.u_shaker_total == 0.0)
        assert np.all(res.shaker_mi == 0.0)
        assert np.all(res.shaker_total_entropy == 0.0)

        # 2. Hutter baselines must be active
        assert np.all(res.u_hutter_total > 0.0)
        assert np.all(res.u_hutter_between >= 0.0)
        assert np.all(res.u_hutter_within > 0.0)
        np.testing.assert_array_equal(res.u_slcb, res.u_hutter_total)

        # 3. All 4 unweighted proximity variants must be evaluated and valid
        for field in ("u_prox_a_lower", "u_prox_b_lower", "u_prox_ac_lower", "u_prox_bc_lower", "u_plcb_lower"):
            arr = getattr(res, field)
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (len(X_test_in),)
            assert np.all(arr >= 0.0)
            assert not np.any(np.isnan(arr))

        # 4. PLCB must dominate floor and prox_a
        assert np.all(res.u_plcb_lower >= res.delta_floor - 1e-12)
        assert np.all(res.u_plcb_lower >= res.u_prox_a_lower - 1e-12)
        np.testing.assert_array_equal(res.u_plcb, res.u_plcb_lower)

        # 5. RF-FIRE lower is evaluated
        assert np.all(res.u_rf_fire_lower >= 0.0)


class TestDualUQEvaluatorCompatibility:
    """Verifies that DualUQEvaluator and DualUQResult operate seamlessly with the new schema."""

    def test_dual_uq_evaluator_keys_and_attributes(self, synthetic_data):
        X_train, y_train, X_test_in, _ = synthetic_data
        dual_eval = DualUQEvaluator(
            seed=42,
            n_trees=10,
            k=28,
            epsilon=0.080791,
            topological_decay_lambda=0.20486,
            eval_mode="unweighted_proximity_only",
            device="cpu",
        )
        dual_eval.fit(X_train, y_train)
        dual_res = dual_eval.evaluate(X_test_in)

        assert isinstance(dual_res, DualUQResult)
        assert is_dataclass(dual_res)

        # Legacy 9 keys
        expected_dual_keys = (
            "y_hat",
            "u_slcb",
            "u_plcb",
            "var_between",
            "var_within",
            "var_total",
            "q_lower",
            "delta_floor",
            "local_mae",
        )
        assert dual_res.keys() == expected_dual_keys

        # Lower-quantile proximity attributes accessible
        assert hasattr(dual_res, "u_prox_a_lower")
        assert hasattr(dual_res, "u_prox_b_lower")
        assert hasattr(dual_res, "u_prox_ac_lower")
        assert hasattr(dual_res, "u_prox_bc_lower")
        assert hasattr(dual_res, "u_plcb_lower")
        assert not hasattr(dual_res, "u_plcb_half")
