import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ConfigSpace import ConfigurationSpace, Float
from smac.scenario import Scenario
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade
from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
from rf_dynamic.dynamic_rf_surrogate import DynamicRFSurrogate
from carps_integration.optimizer import CARPSDynamicRFOptimizer


class TestFinding006ConfoundedBaselines(unittest.TestCase):
    """
    Tests for Audit Finding 6 (HIGH-006):
    Nominal uncertainty/acquisition comparisons must use matched model controls.
    CustomUncertaintyRandomForest must default to the exact hyperparameter geometry
    and target aggregation (log_y) of SMAC3's HyperparameterOptimizationFacade.
    """

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.cs.add(Float("x1", (0.0, 1.0)))
        self.cs.add(Float("x2", (0.0, 1.0)))
        self.scenario = Scenario(configspace=self.cs, n_trials=10, seed=42)

    def test_default_hyperparameters_match_native_facade(self):
        """
        Verify CustomUncertaintyRandomForest defaults match HyperparameterOptimizationFacade.get_model():
        min_samples_leaf=1, min_samples_split=2, ratio_features=1.0, log_y=True.
        """
        native_model = HyperparameterOptimizationFacade.get_model(self.scenario)
        custom_model = CustomUncertaintyRandomForest(configspace=self.cs, seed=42)

        self.assertEqual(custom_model._rf_opts["min_samples_leaf"], native_model._rf_opts["min_samples_leaf"],
                         "min_samples_leaf must match native facade default (1)")
        self.assertEqual(custom_model._rf_opts["min_samples_split"], native_model._rf_opts["min_samples_split"],
                         "min_samples_split must match native facade default (2)")
        self.assertEqual(custom_model._rf_opts["max_features"], native_model._rf_opts["max_features"],
                         "max_features must match native facade default")
        self.assertEqual(custom_model._log_y, native_model._log_y,
                         "log_y must match native facade default (True)")
        self.assertEqual(custom_model._rf_opts["n_estimators"], native_model._rf_opts["n_estimators"],
                         "n_estimators must match native facade default (10)")

    def test_custom_vs_native_mean_and_variance_exact_match(self):
        """
        When configured with 'standard_disagreement', the custom model must produce
        identical mean predictions and variances to SMAC's native RF on identical data and seed.
        """
        native_model = HyperparameterOptimizationFacade.get_model(self.scenario)
        custom_model = CustomUncertaintyRandomForest(
            configspace=self.cs,
            seed=42,
            uncertainty_func="standard_disagreement"
        )

        np.random.seed(42)
        X_train = np.random.uniform(0.0, 1.0, size=(25, 2))
        # Non-negative targets suitable for log_y transformation
        y_train = np.exp(X_train[:, 0] * 2.0 + X_train[:, 1])

        native_model.train(X_train, y_train)
        custom_model.train(X_train, y_train)

        X_test = np.random.uniform(0.0, 1.0, size=(15, 2))
        mu_native, var_native = native_model.predict(X_test)
        mu_custom, var_custom = custom_model.predict(X_test)

        np.testing.assert_allclose(
            mu_custom, mu_native, rtol=1e-5, atol=1e-5,
            err_msg="Custom model mean predictions must match native SMAC model under matched hyperparameters"
        )
        np.testing.assert_allclose(
            var_custom, var_native, rtol=1e-5, atol=1e-5,
            err_msg="Custom standard_disagreement variance must match native SMAC model variance"
        )

    def test_explicit_overrides_respected(self):
        """
        Verify that explicit user or YAML overrides (e.g. min_samples_leaf=5, log_y=False)
        are properly respected and not overwritten by the defaults.
        """
        custom_model = CustomUncertaintyRandomForest(
            configspace=self.cs,
            seed=42,
            min_samples_leaf=5,
            min_samples_split=6,
            ratio_features=0.5,
            log_y=False
        )
        self.assertEqual(custom_model._rf_opts["min_samples_leaf"], 5)
        self.assertEqual(custom_model._rf_opts["min_samples_split"], 6)
        self.assertEqual(custom_model._rf_opts["max_features"], 1)
        self.assertFalse(custom_model._log_y)

    def test_dynamic_rf_surrogate_allows_matched_controls(self):
        """
        Verify DynamicRFSurrogate and CARPSDynamicRFOptimizer can accept matched rf_kwargs
        (e.g., n_estimators, min_samples_leaf, max_features) to allow controlled ablation.
        """
        surrogate = DynamicRFSurrogate(
            extractor_name="standard_disagreement",
            rf_kwargs={"n_estimators": 50, "random_state": 42},
            enable_adaptation=False,
            min_samples_leaf_base=1,
            max_features_base=1.0
        )
        X = np.random.uniform(0, 1, size=(20, 2))
        y = np.sum(X, axis=1)
        surrogate.fit(X, y)

        self.assertEqual(surrogate.model.n_estimators, 50)
        self.assertEqual(surrogate.model.min_samples_leaf, 1)
        self.assertEqual(surrogate.model.max_features, 1.0)


if __name__ == "__main__":
    unittest.main()
