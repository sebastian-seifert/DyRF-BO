import os
import sys
from dataclasses import is_dataclass

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq.uq_evaluator import (
    DualUQEvaluator,
    DualUQResult,
    create_smac_default_rf,
)


class TestSMACDefaultRF:
    """Unit tests for create_smac_default_rf factory and parameter validation."""

    def test_default_smac_parameters_extra_trees(self):
        rf = create_smac_default_rf(seed=42, n_trees=10)
        assert isinstance(rf, ExtraTreesRegressor)
        assert rf.n_estimators == 10
        assert np.isclose(rf.max_features, 5.0 / 6.0)
        assert rf.min_samples_split == 3
        assert rf.min_samples_leaf == 3
        assert rf.criterion == "squared_error"
        assert rf.bootstrap is True
        assert rf.oob_score is True
        assert rf.random_state == 42

    def test_default_smac_parameters_random_forest(self):
        rf = create_smac_default_rf(seed=123, n_trees=20, use_extra_trees=False)
        assert isinstance(rf, RandomForestRegressor)
        assert rf.n_estimators == 20
        assert np.isclose(rf.max_features, 5.0 / 6.0)
        assert rf.min_samples_split == 3
        assert rf.min_samples_leaf == 3
        assert rf.criterion == "squared_error"
        assert rf.bootstrap is True
        assert rf.oob_score is True
        assert rf.random_state == 123

    def test_reuse_existing_unfitted_model(self):
        custom_rf = RandomForestRegressor(n_estimators=15, max_depth=4, random_state=7)
        model = create_smac_default_rf(model=custom_rf)
        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 15
        assert model.max_depth == 4
        assert model.random_state == 7

    def test_reuse_existing_fitted_model(self):
        X = np.random.randn(35, 3)
        y = np.random.randn(35)
        fitted_rf = create_smac_default_rf(seed=42, n_trees=10).fit(X, y)
        reused = create_smac_default_rf(model=fitted_rf)
        assert reused is fitted_rf
        assert hasattr(reused, "estimators_")

    def test_custom_kwargs_override(self):
        rf = create_smac_default_rf(seed=99, n_trees=5, min_impurity_decrease=1e-8)
        assert rf.min_impurity_decrease == 1e-8
        assert rf.n_estimators == 5


class TestHutterLTV:
    """Unit tests for standard SMAC3 LCB Quantification via Hutter Law of Total Variance."""

    @pytest.fixture
    def fitted_evaluator(self):
        np.random.seed(42)
        X_train = np.random.uniform(-0.5, 0.5, size=(60, 4))
        y_train = np.sum(X_train**2, axis=1) + 0.1 * np.random.randn(60)
        evaluator = DualUQEvaluator(seed=42, n_trees=12, k=28, device="cpu")
        evaluator.fit(X_train, y_train)
        return evaluator

    def test_variance_decomposition_identity(self, fitted_evaluator):
        np.random.seed(101)
        X_test = np.random.uniform(-1.0, 1.0, size=(25, 4))
        res = fitted_evaluator.evaluate(X_test)

        # var_total must exactly equal var_between + var_within
        expected_total = res.var_between + res.var_within
        assert np.allclose(res.var_total, expected_total, atol=1e-12)

    def test_variances_non_negativity(self, fitted_evaluator):
        np.random.seed(102)
        X_test = np.random.uniform(-1.0, 1.0, size=(30, 4))
        res = fitted_evaluator.evaluate(X_test)

        assert np.all(res.var_between >= 0.0)
        assert np.all(res.var_within >= 0.0)
        assert np.all(res.var_total >= 0.0)

    def test_u_slcb_formula_and_scaling(self, fitted_evaluator):
        np.random.seed(103)
        X_test = np.random.uniform(-1.0, 1.0, size=(20, 4))
        res = fitted_evaluator.evaluate(X_test)

        expected_u_slcb = 1.96 * np.sqrt(res.var_total)
        assert np.allclose(res.u_slcb, expected_u_slcb, atol=1e-7)
        assert np.all(res.u_slcb >= 0.0)

    def test_manual_hutter_calculation_matches_exact(self, fitted_evaluator):
        np.random.seed(104)
        X_test = np.random.uniform(-1.0, 1.0, size=(10, 4))
        res = fitted_evaluator.evaluate(X_test)

        # Compute manually from individual trees
        trees = fitted_evaluator.model.estimators_
        B = len(trees)
        manual_tree_preds = np.column_stack([t.predict(X_test) for t in trees])
        manual_y_hat = np.mean(manual_tree_preds, axis=1)
        manual_var_between = np.mean((manual_tree_preds - manual_y_hat[:, None]) ** 2, axis=1)

        manual_tree_impurities = np.column_stack([
            np.maximum(0.0, t.tree_.impurity[t.apply(X_test)]) for t in trees
        ])
        manual_var_within = np.mean(manual_tree_impurities, axis=1)
        manual_var_total = manual_var_between + manual_var_within

        assert np.allclose(res.y_hat, manual_y_hat, atol=1e-7)
        assert np.allclose(res.var_between, manual_var_between, atol=1e-7)
        assert np.allclose(res.var_within, manual_var_within, atol=1e-7)
        assert np.allclose(res.var_total, manual_var_total, atol=1e-7)


class TestPLCBExploration:
    """Unit tests for Proximity LCB Quantification (Pure Extracted Exploration Term)."""

    @pytest.fixture
    def fitted_evaluator(self):
        np.random.seed(42)
        X_train = np.random.uniform(-0.5, 0.5, size=(60, 4))
        y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])
        evaluator = DualUQEvaluator(
            seed=42,
            n_trees=10,
            k=28,
            epsilon=0.080791,
            topological_decay_lambda=0.20486,
            level=0.95,
            device="cpu",
        )
        evaluator.fit(X_train, y_train)
        return evaluator

    def test_plcb_exploration_formula(self, fitted_evaluator):
        np.random.seed(201)
        X_test = np.random.uniform(-1.0, 1.0, size=(25, 4))
        res = fitted_evaluator.evaluate(X_test)

        expected_plcb = np.maximum(res.delta_floor, np.abs(res.q_lower))
        assert np.allclose(res.u_plcb, expected_plcb, atol=1e-7)

    def test_delta_floor_calculation(self, fitted_evaluator):
        np.random.seed(202)
        X_test = np.random.uniform(-1.0, 1.0, size=(20, 4))
        res = fitted_evaluator.evaluate(X_test)

        expected_floor = 0.080791 * 1.96 * res.local_mae
        assert np.allclose(res.delta_floor, expected_floor, atol=1e-7)

    def test_plcb_strictly_ge_delta_floor(self, fitted_evaluator):
        np.random.seed(203)
        X_test = np.random.uniform(-1.0, 1.0, size=(30, 4))
        res = fitted_evaluator.evaluate(X_test)

        assert np.all(res.u_plcb >= res.delta_floor - 1e-12)
        assert np.all(res.u_plcb >= 0.0)

    def test_q_lower_is_non_positive_and_magnitude(self, fitted_evaluator):
        np.random.seed(204)
        X_test = np.random.uniform(-1.0, 1.0, size=(25, 4))
        res = fitted_evaluator.evaluate(X_test)

        # q_0.025 is lower residual quantile <= 0
        assert np.all(res.q_lower <= 1e-5)
        # For non-positive values, |q| == -q
        assert np.allclose(np.abs(res.q_lower), -res.q_lower, atol=1e-5)


class TestSharedEnsembleAndConsistency:
    """Unit tests verifying both UQ heads evaluate the identical tree ensemble."""

    def test_shared_model_instance_and_predictions(self):
        np.random.seed(301)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randn(50)

        evaluator = DualUQEvaluator(seed=7, n_trees=10, k=28, device="cpu")
        evaluator.fit(X_train, y_train)

        # Underlying model reference in proximity engine must be identical
        assert evaluator.uq_model.model is evaluator.model

        X_test = np.random.randn(15, 3)
        res = evaluator.evaluate(X_test)

        # Both heads share the exact same mean prediction \hat{\mu}(x)
        expected_y_hat = evaluator.model.predict(X_test)
        assert np.allclose(res.y_hat, expected_y_hat, atol=1e-6)

    def test_deterministic_reproducibility(self):
        X_train = np.random.randn(50, 3)
        y_train = np.random.randn(50)
        X_test = np.random.randn(10, 3)

        ev1 = DualUQEvaluator(seed=42, n_trees=10, k=28, device="cpu").fit(X_train, y_train)
        ev2 = DualUQEvaluator(seed=42, n_trees=10, k=28, device="cpu").fit(X_train, y_train)

        res1 = ev1.evaluate(X_test)
        res2 = ev2.evaluate(X_test)

        assert np.allclose(res1.y_hat, res2.y_hat)
        assert np.allclose(res1.u_slcb, res2.u_slcb)
        assert np.allclose(res1.u_plcb, res2.u_plcb)


class TestShapesAndGuards:
    """Unit tests for shapes, 1D vs 2D inputs, unfitted guards, and container mapping."""

    @pytest.fixture
    def fitted_evaluator(self):
        np.random.seed(42)
        X_train = np.random.randn(40, 5)
        y_train = np.random.randn(40)
        return DualUQEvaluator(seed=42, n_trees=10, k=28, device="cpu").fit(X_train, y_train)

    def test_unfitted_guard(self):
        evaluator = DualUQEvaluator(seed=42, n_trees=10)
        with pytest.raises(RuntimeError, match="must be fitted"):
            evaluator.evaluate(np.zeros((5, 3)))

    def test_1d_vs_2d_query_consistency(self, fitted_evaluator):
        x_1d = np.array([0.1, -0.2, 0.3, 0.4, -0.5])
        x_2d = x_1d.reshape(1, 5)

        res_1d = fitted_evaluator.evaluate(x_1d)
        res_2d = fitted_evaluator.evaluate(x_2d)

        # Output shapes must be (1,)
        assert res_1d.y_hat.shape == (1,)
        assert res_2d.y_hat.shape == (1,)
        assert res_1d.u_slcb.shape == (1,)
        assert res_1d.u_plcb.shape == (1,)

        # Metric values must be identical
        assert np.allclose(res_1d.y_hat, res_2d.y_hat)
        assert np.allclose(res_1d.u_slcb, res_2d.u_slcb)
        assert np.allclose(res_1d.u_plcb, res_2d.u_plcb)
        assert np.allclose(res_1d.var_total, res_2d.var_total)
        assert np.allclose(res_1d.delta_floor, res_2d.delta_floor)
        assert np.allclose(res_1d.q_lower, res_2d.q_lower)

    def test_batch_query_shapes(self, fitted_evaluator):
        M = 35
        X_test = np.random.randn(M, 5)
        res = fitted_evaluator.evaluate(X_test)

        expected_fields = [
            "y_hat",
            "u_slcb",
            "u_plcb",
            "var_between",
            "var_within",
            "var_total",
            "q_lower",
            "delta_floor",
            "local_mae",
        ]
        for field in expected_fields:
            arr = getattr(res, field)
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (M,)

    def test_dimension_mismatch_guards(self, fitted_evaluator):
        # 3 features instead of 5
        with pytest.raises(ValueError, match="dimension mismatch|features"):
            fitted_evaluator.evaluate(np.zeros((10, 3)))

        # 3D test input
        with pytest.raises(ValueError, match="1D or 2D"):
            fitted_evaluator.evaluate(np.zeros((2, 5, 2)))

    def test_fit_input_guards(self):
        evaluator = DualUQEvaluator()
        # 1D X_train
        with pytest.raises(ValueError, match="2D array"):
            evaluator.fit(np.zeros(10), np.zeros(10))

        # Empty X_train
        with pytest.raises(ValueError, match="empty|samples"):
            evaluator.fit(np.zeros((0, 3)), np.zeros(0))

        # Sample length mismatch
        with pytest.raises(ValueError, match="samples"):
            evaluator.fit(np.zeros((10, 3)), np.zeros(5))

    def test_dual_uq_result_container_mapping(self, fitted_evaluator):
        X_test = np.random.randn(4, 5)
        res = fitted_evaluator.evaluate(X_test)

        # Dataclass check
        assert is_dataclass(res)

        # Attribute access
        assert isinstance(res.y_hat, np.ndarray)

        # Dict / Mapping access
        assert isinstance(res["y_hat"], np.ndarray)
        assert np.array_equal(res["y_hat"], res.y_hat)
        assert "u_slcb" in res
        assert "nonexistent_key" not in res
        assert res.get("u_plcb") is not None
        assert res.get("missing", 42) == 42

        # Keys, values, items
        keys = list(res.keys())
        assert len(keys) == 9
        assert "u_slcb" in keys
        assert len(list(res.values())) == 9
        assert len(list(res.items())) == 9

        # dict conversion
        d = dict(res)
        assert isinstance(d, dict)
        assert len(d) == 9
        assert "delta_floor" in d

        # to_dict method
        d2 = res.to_dict()
        assert isinstance(d2, dict)
        assert "q_lower" in d2

        # KeyError on invalid key
        with pytest.raises(KeyError):
            _ = res["invalid_metric"]
