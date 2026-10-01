"""Unit and integration tests for Multi-Surrogate Random Forest Factory and Evaluator Integration.

Covers Milestone A of the 5 Random Forest Surrogate integration into DyRF-BO:
- SURROGATE_CONFIGS definition for smac_default, mature, shallow, coarse, breiman.
- create_surrogate_rf factory function, validation, and parameter overrides.
- MultiUQEvaluator integration with surrogate_type parameter.
- Package exports in dyrf_bo.extrapolation_uq.
- End-to-end fit and evaluate across all 5 surrogate types.
"""

import os
import sys
from typing import Any, Dict

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq import (
    SURROGATE_CONFIGS,
    MultiUQEvaluator,
    MultiUQResult,
    create_smac_default_rf,
    create_surrogate_rf,
)


class TestSurrogateConfigs:
    """Tests for SURROGATE_CONFIGS mapping and exact hyperparameter definitions."""

    def test_surrogate_configs_keys(self):
        """SURROGATE_CONFIGS must contain exactly the 5 target surrogates."""
        expected_keys = {"smac_default", "mature", "shallow", "coarse", "breiman"}
        assert set(SURROGATE_CONFIGS.keys()) == expected_keys

    def test_smac_default_config(self):
        """smac_default: ExtraTreesRegressor, n_estimators=10, max_depth=None,
        min_samples_split=3, min_samples_leaf=3, max_features=5/6, bootstrap=True, oob_score=True."""
        cfg = SURROGATE_CONFIGS["smac_default"]
        estimator_cls = cfg.get("estimator_class", cfg.get("model_cls", cfg.get("model_class")))
        assert estimator_cls is ExtraTreesRegressor
        assert cfg["n_estimators"] == 10
        assert cfg["max_depth"] is None
        assert cfg["min_samples_split"] == 3
        assert cfg["min_samples_leaf"] == 3
        assert np.isclose(cfg["max_features"], 5.0 / 6.0)
        assert cfg["bootstrap"] is True
        assert cfg["oob_score"] is True

    def test_mature_config(self):
        """mature: ExtraTreesRegressor, n_estimators=100, max_depth=None,
        min_samples_split=3, min_samples_leaf=3, max_features=5/6, bootstrap=True, oob_score=True."""
        cfg = SURROGATE_CONFIGS["mature"]
        estimator_cls = cfg.get("estimator_class", cfg.get("model_cls", cfg.get("model_class")))
        assert estimator_cls is ExtraTreesRegressor
        assert cfg["n_estimators"] == 100
        assert cfg["max_depth"] is None
        assert cfg["min_samples_split"] == 3
        assert cfg["min_samples_leaf"] == 3
        assert np.isclose(cfg["max_features"], 5.0 / 6.0)
        assert cfg["bootstrap"] is True
        assert cfg["oob_score"] is True

    def test_shallow_config(self):
        """shallow: ExtraTreesRegressor, n_estimators=25, max_depth=4,
        min_samples_split=3, min_samples_leaf=5, max_features=5/6, bootstrap=True, oob_score=True."""
        cfg = SURROGATE_CONFIGS["shallow"]
        estimator_cls = cfg.get("estimator_class", cfg.get("model_cls", cfg.get("model_class")))
        assert estimator_cls is ExtraTreesRegressor
        assert cfg["n_estimators"] == 25
        assert cfg["max_depth"] == 4
        assert cfg["min_samples_split"] == 3
        assert cfg["min_samples_leaf"] == 5
        assert np.isclose(cfg["max_features"], 5.0 / 6.0)
        assert cfg["bootstrap"] is True
        assert cfg["oob_score"] is True

    def test_coarse_config(self):
        """coarse: ExtraTreesRegressor, n_estimators=25, max_depth=None,
        min_samples_split=20, min_samples_leaf=10, max_features=5/6, bootstrap=True, oob_score=True."""
        cfg = SURROGATE_CONFIGS["coarse"]
        estimator_cls = cfg.get("estimator_class", cfg.get("model_cls", cfg.get("model_class")))
        assert estimator_cls is ExtraTreesRegressor
        assert cfg["n_estimators"] == 25
        assert cfg["max_depth"] is None
        assert cfg["min_samples_split"] == 20
        assert cfg["min_samples_leaf"] == 10
        assert np.isclose(cfg["max_features"], 5.0 / 6.0)
        assert cfg["bootstrap"] is True
        assert cfg["oob_score"] is True

    def test_breiman_config(self):
        """breiman: RandomForestRegressor, n_estimators=25, max_depth=None,
        min_samples_split=3, min_samples_leaf=3, max_features=5/6, bootstrap=True, oob_score=True."""
        cfg = SURROGATE_CONFIGS["breiman"]
        estimator_cls = cfg.get("estimator_class", cfg.get("model_cls", cfg.get("model_class")))
        assert estimator_cls is RandomForestRegressor
        assert cfg["n_estimators"] == 25
        assert cfg["max_depth"] is None
        assert cfg["min_samples_split"] == 3
        assert cfg["min_samples_leaf"] == 3
        assert np.isclose(cfg["max_features"], 5.0 / 6.0)
        assert cfg["bootstrap"] is True
        assert cfg["oob_score"] is True


class TestCreateSurrogateRF:
    """Unit tests for create_surrogate_rf factory function."""

    @pytest.mark.parametrize(
        "surrogate_type,expected_cls,expected_n_est,expected_depth,expected_split,expected_leaf",
        [
            ("smac_default", ExtraTreesRegressor, 10, None, 3, 3),
            ("mature", ExtraTreesRegressor, 100, None, 3, 3),
            ("shallow", ExtraTreesRegressor, 25, 4, 3, 5),
            ("coarse", ExtraTreesRegressor, 25, None, 20, 10),
            ("breiman", RandomForestRegressor, 25, None, 3, 3),
        ],
    )
    def test_create_surrogate_rf_instances(
        self,
        surrogate_type: str,
        expected_cls: Any,
        expected_n_est: int,
        expected_depth: Any,
        expected_split: int,
        expected_leaf: int,
    ):
        model = create_surrogate_rf(surrogate_type=surrogate_type, seed=42)
        assert isinstance(model, expected_cls)
        if expected_cls is RandomForestRegressor:
            assert type(model) is RandomForestRegressor
        assert model.n_estimators == expected_n_est
        assert model.max_depth == expected_depth
        assert model.min_samples_split == expected_split
        assert model.min_samples_leaf == expected_leaf
        assert np.isclose(model.max_features, 5.0 / 6.0)
        assert model.bootstrap is True
        assert model.oob_score is True
        assert model.random_state == 42

    def test_unknown_surrogate_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown surrogate_type 'nonexistent'"):
            create_surrogate_rf(surrogate_type="nonexistent")

        with pytest.raises(ValueError):
            create_surrogate_rf(surrogate_type="")

    def test_kwargs_override(self):
        model = create_surrogate_rf(
            surrogate_type="shallow",
            seed=123,
            max_depth=7,
            n_estimators=50,
            min_samples_leaf=2,
        )
        assert isinstance(model, ExtraTreesRegressor)
        assert model.max_depth == 7
        assert model.n_estimators == 50
        assert model.min_samples_leaf == 2
        assert model.random_state == 123

    def test_reuse_existing_unfitted_model(self):
        custom_rf = RandomForestRegressor(n_estimators=17, max_depth=5, random_state=99)
        model = create_surrogate_rf(surrogate_type="smac_default", model=custom_rf, seed=7)
        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 17
        assert model.max_depth == 5
        assert model.random_state == 7

    def test_reuse_existing_fitted_model(self):
        X = np.random.randn(30, 2)
        y = np.random.randn(30)
        fitted = ExtraTreesRegressor(n_estimators=10, random_state=42).fit(X, y)
        reused = create_surrogate_rf(surrogate_type="mature", model=fitted)
        assert reused is fitted
        assert hasattr(reused, "estimators_")

    def test_backwards_compatibility_create_smac_default_rf(self):
        rf1 = create_smac_default_rf(seed=42)
        rf2 = create_surrogate_rf("smac_default", seed=42)
        assert type(rf1) is type(rf2)
        assert rf1.n_estimators == rf2.n_estimators == 10
        assert rf1.min_samples_split == rf2.min_samples_split == 3
        assert rf1.min_samples_leaf == rf2.min_samples_leaf == 3
        assert rf1.random_state == rf2.random_state == 42


class TestMultiUQEvaluatorSurrogates:
    """Integration tests for MultiUQEvaluator with 5 surrogate configurations."""

    def test_default_surrogate_is_smac_default(self):
        evaluator = MultiUQEvaluator(seed=42)
        assert evaluator.surrogate_type == "smac_default"
        assert isinstance(evaluator.model, ExtraTreesRegressor)
        assert evaluator.n_trees == 10
        assert evaluator.n_estimators == 10

    @pytest.mark.parametrize(
        "surrogate_type,expected_cls,expected_n_trees",
        [
            ("smac_default", ExtraTreesRegressor, 10),
            ("mature", ExtraTreesRegressor, 100),
            ("shallow", ExtraTreesRegressor, 25),
            ("coarse", ExtraTreesRegressor, 25),
            ("breiman", RandomForestRegressor, 25),
        ],
    )
    def test_evaluator_initialization_surrogate_types(
        self,
        surrogate_type: str,
        expected_cls: Any,
        expected_n_trees: int,
    ):
        evaluator = MultiUQEvaluator(surrogate_type=surrogate_type, seed=42)
        assert evaluator.surrogate_type == surrogate_type
        assert isinstance(evaluator.model, expected_cls)
        if expected_cls is RandomForestRegressor:
            assert type(evaluator.model) is RandomForestRegressor
        assert evaluator.n_trees == expected_n_trees
        assert evaluator.n_estimators == expected_n_trees
        assert evaluator.model.n_estimators == expected_n_trees

    def test_evaluator_unknown_surrogate_raises(self):
        with pytest.raises(ValueError, match="Unknown surrogate_type 'invalid'"):
            MultiUQEvaluator(surrogate_type="invalid")

    def test_explicit_n_trees_override(self):
        # Explicit n_trees differing from default 10 should override surrogate default
        evaluator = MultiUQEvaluator(surrogate_type="mature", n_trees=50, seed=42)
        assert evaluator.n_trees == 50
        assert evaluator.model.n_estimators == 50

        # When n_trees is default 10 and surrogate is mature, mature default 100 is respected
        evaluator_default = MultiUQEvaluator(surrogate_type="mature", n_trees=10, seed=42)
        assert evaluator_default.n_trees == 100
        assert evaluator_default.model.n_estimators == 100

        # Kwarg n_estimators overrides everything
        evaluator_kwarg = MultiUQEvaluator(surrogate_type="shallow", n_estimators=40, seed=42)
        assert evaluator_kwarg.n_trees == 40
        assert evaluator_kwarg.model.n_estimators == 40

    def test_evaluator_with_custom_model(self):
        custom = RandomForestRegressor(n_estimators=15, max_depth=3, random_state=42)
        evaluator = MultiUQEvaluator(model=custom, surrogate_type="breiman")
        assert evaluator.model.n_estimators == 15
        assert evaluator.model.max_depth == 3
        assert evaluator.n_trees == 15

    @pytest.mark.parametrize(
        "surrogate_type",
        ["smac_default", "mature", "shallow", "coarse", "breiman"],
    )
    def test_end_to_end_fit_and_evaluate_all_surrogates(self, surrogate_type: str):
        """Fit and evaluate all 5 surrogates on a synthetic dataset to guarantee full pipeline execution."""
        np.random.seed(42)
        N, D, M = 25, 2, 5
        X_train = np.random.uniform(-0.5, 0.5, size=(N, D))
        y_train = np.sin(X_train[:, 0]) + X_train[:, 1] ** 2 + 0.05 * np.random.randn(N)
        X_test = np.random.uniform(-1.0, 1.0, size=(M, D))

        evaluator = MultiUQEvaluator(
            surrogate_type=surrogate_type,
            seed=42,
            k=8,
            device="cpu",
        )
        evaluator.fit(X_train, y_train)
        res = evaluator.evaluate(X_test)

        assert isinstance(res, MultiUQResult)
        assert res.y_hat.shape == (M,)
        assert np.all(np.isfinite(res.y_hat))

        # Check Hutter LTV components
        assert res.u_hutter_total.shape == (M,)
        assert np.all(res.u_hutter_total > 0.0)
        assert np.all(np.isfinite(res.u_hutter_between))
        assert np.all(np.isfinite(res.u_hutter_within))

        # Check Shaker entropy components
        assert res.u_shaker_epistemic.shape == (M,)
        assert np.all(np.isfinite(res.u_shaker_epistemic))
        assert res.u_shaker_total.shape == (M,)
        assert np.all(np.isfinite(res.u_shaker_total))

        # Check Proximity components
        assert res.u_rf_fire_half.shape == (M,)
        assert np.all(np.isfinite(res.u_rf_fire_half))
        assert res.u_prox_a_half.shape == (M,)
        assert np.all(np.isfinite(res.u_prox_a_half))
        assert res.u_prox_b_half.shape == (M,)
        assert np.all(np.isfinite(res.u_prox_b_half))
        assert res.u_prox_bc_half.shape == (M,)
        assert np.all(np.isfinite(res.u_prox_bc_half))
        assert res.u_plcb_half.shape == (M,)
        assert np.all(np.isfinite(res.u_plcb_half))
        assert res.u_plcb_lower.shape == (M,)
        assert np.all(np.isfinite(res.u_plcb_lower))
