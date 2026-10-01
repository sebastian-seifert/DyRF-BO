"""Comprehensive unit and integration tests for MultiUQEvaluator and MultiUQResult.

Validates all 10 candidate uncertainty estimators (15 uncertainty signals + 2 aliases),
mathematical invariants, shared ensemble constraints, kernel sharing, backwards compatibility,
and container mapping interfaces under Milestone 1 of the Multi-UQ extrapolation masterplan.
"""

import os
import sys
from dataclasses import is_dataclass
from typing import Dict

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq.uq_evaluator import (
    DualUQEvaluator,
    DualUQResult,
    MultiUQEvaluator,
    MultiUQResult,
    create_smac_default_rf,
)


EXPECTED_UQ_KEYS = (
    "u_hutter_total",
    "u_hutter_between",
    "u_hutter_within",
    "u_shaker_epistemic",
    "u_shaker_total",
    "u_rf_fire_half",
    "u_rf_fire_lower",
    "u_prox_a_half",
    "u_prox_a_lower",
    "u_prox_b_half",
    "u_prox_b_lower",
    "u_prox_bc_half",
    "u_prox_bc_lower",
    "u_plcb_half",
    "u_plcb_lower",
)

EXPECTED_ALIASES = (
    "u_slcb",
    "u_plcb",
)


class TestEvaluatorInitializationAndAliases:
    """Verifies class hierarchy, aliases, and initialization contracts."""

    def test_multi_uq_evaluator_alias_and_subclass(self):
        """DualUQEvaluator must be an alias or compatible subclass of MultiUQEvaluator."""
        assert issubclass(DualUQEvaluator, MultiUQEvaluator) or DualUQEvaluator is MultiUQEvaluator

    def test_multi_uq_result_alias_and_subclass(self):
        """DualUQResult must be an alias or compatible subclass of MultiUQResult."""
        assert issubclass(DualUQResult, MultiUQResult) or DualUQResult is MultiUQResult

    def test_initialization_defaults(self):
        evaluator = MultiUQEvaluator(seed=42)
        assert evaluator.seed == 42
        assert evaluator.n_trees == 10
        assert evaluator.k == 28
        assert np.isclose(evaluator.epsilon, 0.080791)
        assert np.isclose(evaluator.topological_decay_lambda, 0.20486)
        assert np.isclose(evaluator.level, 0.95)
        assert np.isclose(evaluator.kappa, 1.96)
        assert evaluator.device == "cpu"
        assert isinstance(evaluator.model, ExtraTreesRegressor)


class TestFitAndOutputShapes:
    """Verifies that all 15 uncertainty signals and 2 aliases are produced with valid shapes and dtypes."""

    @pytest.fixture
    def fitted_bundle(self):
        np.random.seed(42)
        N, D, M = 50, 3, 30
        X_train = np.random.uniform(-0.5, 0.5, size=(N, D))
        y_train = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1] ** 2 + 0.1 * np.random.randn(N)
        evaluator = MultiUQEvaluator(seed=42, n_trees=10, k=28, device="cpu")
        evaluator.fit(X_train, y_train)
        X_test = np.random.uniform(-1.0, 1.0, size=(M, D))
        res = evaluator.evaluate(X_test)
        return evaluator, X_test, res, M

    def test_all_15_uncertainty_columns_present(self, fitted_bundle):
        _, _, res, M = fitted_bundle
        for key in EXPECTED_UQ_KEYS:
            assert hasattr(res, key), f"Missing attribute {key} on MultiUQResult"
            arr = getattr(res, key)
            assert isinstance(arr, np.ndarray), f"{key} must be an np.ndarray"
            assert arr.shape == (M,), f"{key} shape {arr.shape} != ({M},)"
            assert np.issubdtype(arr.dtype, np.floating), f"{key} dtype {arr.dtype} not floating"

    def test_backwards_compatibility_aliases_present(self, fitted_bundle):
        _, _, res, M = fitted_bundle
        for alias in EXPECTED_ALIASES:
            assert hasattr(res, alias), f"Missing alias attribute {alias} on MultiUQResult"
            arr = getattr(res, alias)
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (M,)
            assert np.issubdtype(arr.dtype, np.floating)

        # Exact equivalence of aliases
        assert np.array_equal(res.u_slcb, res.u_hutter_total)
        assert np.array_equal(res.u_plcb, res.u_plcb_lower)

    def test_uncertainties_non_negative_and_finite(self, fitted_bundle):
        _, _, res, _ = fitted_bundle
        for key in EXPECTED_UQ_KEYS:
            arr = getattr(res, key)
            assert np.all(np.isfinite(arr)), f"{key} contains non-finite values"
            assert np.all(arr >= 0.0), f"{key} has negative values (min: {np.min(arr)})"

        for alias in EXPECTED_ALIASES:
            arr = getattr(res, alias)
            assert np.all(np.isfinite(arr))
            assert np.all(arr >= 0.0)


class TestHutterVarianceDecomposition:
    """Verifies SMAC3 Hutter Law of Total Variance decomposition identities."""

    @pytest.fixture
    def evaluator(self):
        np.random.seed(123)
        X_train = np.random.uniform(-0.5, 0.5, size=(45, 3))
        y_train = np.sum(X_train**2, axis=1) + 0.05 * np.random.randn(45)
        evaluator = MultiUQEvaluator(seed=123, n_trees=10, k=28, device="cpu")
        evaluator.fit(X_train, y_train)
        return evaluator

    def test_variance_decomposition_exact_identity(self, evaluator):
        X_test = np.random.uniform(-1.0, 1.0, size=(25, 3))
        res = evaluator.evaluate(X_test)

        # var_total == var_between + var_within
        assert np.allclose(res.var_total, res.var_between + res.var_within, atol=1e-12)
        # u_hutter_total == 1.96 * sqrt(var_total)
        expected_u_total = 1.96 * np.sqrt(res.var_total)
        assert np.allclose(res.u_hutter_total, expected_u_total, atol=1e-7)

        # u_hutter_between == 1.96 * sqrt(var_between)
        expected_u_between = 1.96 * np.sqrt(res.var_between)
        assert np.allclose(res.u_hutter_between, expected_u_between, atol=1e-7)

        # u_hutter_within == 1.96 * sqrt(var_within)
        expected_u_within = 1.96 * np.sqrt(res.var_within)
        assert np.allclose(res.u_hutter_within, expected_u_within, atol=1e-7)

        # Total variance must be >= between variance and within variance
        assert np.all(res.u_hutter_total >= res.u_hutter_between - 1e-12)
        assert np.all(res.u_hutter_total >= res.u_hutter_within - 1e-12)


class TestShakerBoundsAndFormulations:
    """Verifies Shaker mutual information bounds and Gaussian equivalent inversion formulas."""

    @pytest.fixture
    def evaluator(self):
        np.random.seed(456)
        X_train = np.random.uniform(-0.5, 0.5, size=(50, 4))
        y_train = np.exp(X_train[:, 0]) - 2.0 * X_train[:, 1] + 0.1 * np.random.randn(50)
        evaluator = MultiUQEvaluator(seed=456, n_trees=10, k=28, device="cpu")
        evaluator.fit(X_train, y_train)
        return evaluator

    def test_shaker_information_bounds(self, evaluator):
        X_test = np.random.uniform(-1.0, 1.0, size=(20, 4))
        res = evaluator.evaluate(X_test)

        B = evaluator.n_trees
        max_mi_bound = np.log2(B)

        assert hasattr(res, "shaker_mi")
        assert hasattr(res, "shaker_total_entropy")
        # 0 <= MI <= log2(B)
        assert np.all(res.shaker_mi >= -1e-12)
        assert np.all(res.shaker_mi <= max_mi_bound + 1e-4)

        # Positivity of Shaker uncertainties
        assert np.all(res.u_shaker_epistemic >= 0.0)
        assert np.all(res.u_shaker_total >= 0.0)

    def test_shaker_total_entropy_power_relation(self, evaluator):
        X_test = np.random.uniform(-1.0, 1.0, size=(15, 4))
        res = evaluator.evaluate(X_test)

        # Verify u_shaker_total = 1.96 * sqrt( 2^(2*H_total) / (2*pi*e) )
        safe_exp = np.clip(2.0 * res.shaker_total_entropy, -50.0, 50.0)
        entropy_var = (2.0 ** safe_exp) / (2.0 * np.pi * np.e)
        expected_u_total = 1.96 * np.sqrt(np.maximum(entropy_var, 0.0))
        assert np.allclose(res.u_shaker_total, expected_u_total, atol=1e-6)


class TestProximityVariantsAndFloors:
    """Verifies all 5 proximity variants (10 metrics) and shared-kernel invariants."""

    @pytest.fixture
    def evaluator(self):
        np.random.seed(789)
        X_train = np.random.uniform(-0.5, 0.5, size=(60, 3))
        y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])
        evaluator = MultiUQEvaluator(
            seed=789,
            n_trees=10,
            k=28,
            epsilon=0.080791,
            topological_decay_lambda=0.20486,
            device="cpu",
        )
        evaluator.fit(X_train, y_train)
        return evaluator

    def test_plcb_vs_prox_a_floor_domination(self, evaluator):
        """PLCB enforces delta_floor while Proximity A has epsilon=0.

        Therefore, u_plcb_lower >= u_prox_a_lower pointwise everywhere.
        Meanwhile, half-widths u_plcb_half == u_prox_a_half identically.
        """
        X_test = np.random.uniform(-1.0, 1.0, size=(30, 3))
        res = evaluator.evaluate(X_test)

        assert np.all(res.u_plcb_lower >= res.u_prox_a_lower - 1e-12)
        assert np.allclose(res.u_plcb_half, res.u_prox_a_half, atol=1e-7)
        assert np.all(res.u_plcb_lower >= res.delta_floor - 1e-12)

    def test_rf_fire_metrics(self, evaluator):
        """Standard RF-FIRE uses co-occurrence proximity (lambda=0)."""
        X_test = np.random.uniform(-1.0, 1.0, size=(25, 3))
        res = evaluator.evaluate(X_test)

        assert np.all(res.u_rf_fire_half > 0.0)
        assert np.all(res.u_rf_fire_lower >= 0.0)

    def test_proximity_b_continuous_quantiles(self, evaluator):
        """Proximity B uses auto-k continuous weighted quantiles."""
        X_test = np.random.uniform(-1.0, 1.0, size=(20, 3))
        res = evaluator.evaluate(X_test)

        assert np.all(res.u_prox_b_half > 0.0)
        assert np.all(res.u_prox_b_lower >= 0.0)

    def test_proximity_bc_density_scaling_expansion(self, evaluator):
        """Proximity B+C scales inversely with density in empty extrapolation space."""
        # Query near origin (dense training region) vs far extrapolation point
        x_dense = np.zeros((1, 3))
        x_far = np.full((1, 3), 5.0)

        res_dense = evaluator.evaluate(x_dense)
        res_far = evaluator.evaluate(x_far)

        # In far empty space, density is low, so gamma > 1 and u_prox_bc > u_prox_b
        ratio_dense = res_dense.u_prox_bc_half / res_dense.u_prox_b_half
        ratio_far = res_far.u_prox_bc_half / res_far.u_prox_b_half

        assert ratio_far[0] > ratio_dense[0]
        assert res_far.u_prox_bc_half[0] > res_far.u_prox_b_half[0]


class TestContainerAndMappingCompatibility:
    """Verifies MultiUQResult mapping interface and backwards compatibility."""

    @pytest.fixture
    def res(self):
        np.random.seed(999)
        X_train = np.random.randn(40, 3)
        y_train = np.random.randn(40)
        evaluator = MultiUQEvaluator(seed=999, n_trees=10, k=28, device="cpu").fit(X_train, y_train)
        return evaluator.evaluate(np.random.randn(10, 3))

    def test_dataclass_contract(self, res):
        assert is_dataclass(res)

    def test_dict_and_attribute_parity(self, res):
        # Attribute access matches dictionary key access
        for key in EXPECTED_UQ_KEYS:
            assert np.array_equal(getattr(res, key), res[key])

        for alias in EXPECTED_ALIASES:
            assert np.array_equal(getattr(res, alias), res[alias])

    def test_get_uncertainties_dict(self, res):
        assert hasattr(res, "get_uncertainties_dict")
        u_dict = res.get_uncertainties_dict()
        assert isinstance(u_dict, dict)
        for key in EXPECTED_UQ_KEYS:
            assert key in u_dict
            assert np.array_equal(u_dict[key], getattr(res, key))

    def test_to_dict_method(self, res):
        d = res.to_dict()
        assert isinstance(d, dict)
        assert "u_hutter_total" in d
        assert "u_slcb" in d
        assert "u_plcb" in d

    def test_dual_uq_evaluator_legacy_compatibility(self):
        """DualUQEvaluator returns a result compatible with legacy 9-key container contract."""
        X_train = np.random.randn(40, 3)
        y_train = np.random.randn(40)
        legacy_eval = DualUQEvaluator(seed=42, n_trees=10, k=28, device="cpu").fit(X_train, y_train)
        legacy_res = legacy_eval.evaluate(np.random.randn(5, 3))

        # Check legacy attributes exist
        for leg_field in [
            "y_hat", "u_slcb", "u_plcb", "var_between", "var_within",
            "var_total", "q_lower", "delta_floor", "local_mae"
        ]:
            assert hasattr(legacy_res, leg_field)

        # Check new multi-UQ attributes ALSO accessible on legacy result
        assert hasattr(legacy_res, "u_hutter_total")
        assert hasattr(legacy_res, "u_shaker_epistemic")
        assert hasattr(legacy_res, "u_prox_a_half")


class TestDeterministicReproducibility:
    """Verifies that identical random states produce bit-exact identical predictions and UQ."""

    def test_deterministic_output(self):
        X_train = np.random.randn(50, 3)
        y_train = np.random.randn(50)
        X_test = np.random.randn(15, 3)

        ev1 = MultiUQEvaluator(seed=42, n_trees=10, k=28, device="cpu").fit(X_train, y_train)
        ev2 = MultiUQEvaluator(seed=42, n_trees=10, k=28, device="cpu").fit(X_train, y_train)

        res1 = ev1.evaluate(X_test)
        res2 = ev2.evaluate(X_test)

        assert np.array_equal(res1.y_hat, res2.y_hat)
        for key in EXPECTED_UQ_KEYS:
            assert np.array_equal(getattr(res1, key), getattr(res2, key)), f"Mismatch in {key}"
