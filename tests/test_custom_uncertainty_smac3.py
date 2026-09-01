import os
import sys
import unittest
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ConfigSpace import ConfigurationSpace, Float
from smac.scenario import Scenario
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade

class TestCustomUncertaintySMAC3(unittest.TestCase):
    def setUp(self):
        self.cs = ConfigurationSpace()
        self.cs.add([
            Float("x1", (-5.0, 5.0), default=0.0),
            Float("x2", (-5.0, 5.0), default=0.0)
        ])
        
        np.random.seed(42)
        self.X_train = np.random.uniform(-5.0, 5.0, (20, 2))
        self.y_train = (self.X_train[:, 0]**2 + self.X_train[:, 1]**2).reshape(-1, 1)

    def test_custom_uncertainty_rf_registry_string(self):
        """Verify CustomUncertaintyRandomForest uses registry key to override variance."""
        from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
        
        rf = CustomUncertaintyRandomForest(
            uncertainty_func="proximity_bc",
            configspace=self.cs,
            n_trees=10,
            seed=42
        )
        rf.train(self.X_train, self.y_train)
        
        X_test = np.random.uniform(-5.0, 5.0, (5, 2))
        mean, var = rf._predict(X_test)
        
        self.assertEqual(mean.shape, (5, 1))
        self.assertEqual(var.shape, (5, 1))
        self.assertTrue(np.all(var >= 0.0))

    def test_custom_uncertainty_rf_custom_callable(self):
        """Verify CustomUncertaintyRandomForest accepts a custom callable function for uncertainty."""
        from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
        
        def constant_uncertainty_func(model, X, y):
            return np.full(len(X), 7.0)

        rf = CustomUncertaintyRandomForest(
            uncertainty_func=constant_uncertainty_func,
            configspace=self.cs,
            n_trees=10,
            seed=42
        )
        rf.train(self.X_train, self.y_train)
        
        X_test = np.random.uniform(-5.0, 5.0, (5, 2))
        mean, var = rf._predict(X_test)
        
        # Output variance should be (7.0)^2 = 49.0
        expected_var = np.full((5, 1), 49.0)
        np.testing.assert_allclose(var, expected_var)

    def test_custom_uncertainty_smac3_hpo_facade_run(self):
        """Verify native SMAC3 HPOFacade runs using CustomUncertaintyRandomForest."""
        from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
        
        scenario = Scenario(configspace=self.cs, n_trials=5, seed=42)
        
        def target_func(config, seed=0):
            return config["x1"]**2 + config["x2"]**2

        smac = HPOFacade(
            scenario=scenario,
            target_function=target_func,
            model=CustomUncertaintyRandomForest(uncertainty_func="standard_proximity", configspace=self.cs, n_trees=10),
            overwrite=True
        )
        
        incumbent = smac.optimize()
        self.assertIsNotNone(incumbent)

    def test_all_registered_extractors_connect_to_smac3_rf(self):
        """Verify that EVERY single registered UQ extractor in ep_extractors connects cleanly to SMAC3's self._rf."""
        from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
        from ep_extractors import UQExtractorRegistry

        all_keys = list(UQExtractorRegistry._registry.keys())
        self.assertGreater(len(all_keys), 0)

        for key in all_keys:
            with self.subTest(extractor_key=key):
                rf = CustomUncertaintyRandomForest(
                    uncertainty_func=key,
                    configspace=self.cs,
                    n_trees=10,
                    seed=42
                )
                rf.train(self.X_train, self.y_train)
                
                X_test = np.random.uniform(-5.0, 5.0, (4, 2))
                mean, var = rf._predict(X_test)
                
                self.assertEqual(mean.shape, (4, 1))
                self.assertEqual(var.shape, (4, 1))
                self.assertTrue(np.all(var >= 0.0), f"Variance for key '{key}' should be non-negative.")

    def test_smac3_flexibility_acqf_and_initial_design(self):
        """Verify changing acquisition function (LCB, PI) and initial design in SMAC3 works natively with CustomUncertaintyRandomForest."""
        from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
        from smac.acquisition.function import LCB, PI
        from smac.initial_design import RandomInitialDesign

        scenario = Scenario(configspace=self.cs, n_trials=5, seed=42)
        
        def target_func(config, seed=0):
            return config["x1"]**2 + config["x2"]**2

        # 1. Test with LowerConfidenceBound (LCB) & RandomInitialDesign
        smac_lcb = HPOFacade(
            scenario=scenario,
            target_function=target_func,
            model=CustomUncertaintyRandomForest(uncertainty_func="proximity_bc", configspace=self.cs, n_trees=10),
            acquisition_function=LCB(beta=2.0),
            initial_design=RandomInitialDesign(scenario=scenario, n_configs=2),
            overwrite=True
        )
        inc_lcb = smac_lcb.optimize()
        self.assertIsNotNone(inc_lcb)

        # 2. Test with ProbabilityOfImprovement (PI)
        smac_pi = HPOFacade(
            scenario=scenario,
            target_function=target_func,
            model=CustomUncertaintyRandomForest(uncertainty_func="shaker_entropy", configspace=self.cs, n_trees=10),
            acquisition_function=PI(),
            overwrite=True
        )
        inc_pi = smac_pi.optimize()
    def test_impute_inactive_hierarchical_space(self):
        """Verify CustomUncertaintyRandomForest imputes inactive hyperparameters (NaNs) in hierarchical spaces."""
        from ConfigSpace import Categorical, EqualsCondition
        from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest

        cs_hier = ConfigurationSpace(seed=42)
        model_type = Categorical("model_type", ["rf", "svm"], default="rf")
        rf_trees = Float("rf_trees", (10.0, 100.0), default=50.0)
        svm_c = Float("svm_c", (0.1, 10.0), default=1.0)
        cs_hier.add([model_type, rf_trees, svm_c])
        cs_hier.add(EqualsCondition(rf_trees, model_type, "rf"))
        cs_hier.add(EqualsCondition(svm_c, model_type, "svm"))

        # Generate data with inactive hyperparameters containing NaNs
        # Column 0: model_type (0 for rf, 1 for svm)
        # Column 1: rf_trees (NaN when svm)
        # Column 2: svm_c (NaN when rf)
        X_raw = np.array([
            [0.0, 50.0, np.nan],
            [0.0, 75.0, np.nan],
            [1.0, np.nan, 2.5],
            [1.0, np.nan, 5.0],
            [0.0, 20.0, np.nan],
            [1.0, np.nan, 0.5],
        ])
        y = np.array([[1.0], [2.0], [1.5], [3.0], [0.8], [2.2]])

        rf = CustomUncertaintyRandomForest(
            uncertainty_func="proximity_bc",
            configspace=cs_hier,
            n_trees=10,
            seed=42
        )
        # Train should impute inactive NaNs cleanly without error in UQ fit
        rf.train(X_raw, y)

        X_test = np.array([
            [0.0, 60.0, np.nan],
            [1.0, np.nan, 1.2],
        ])
        mean, var = rf._predict(X_test)
        self.assertEqual(mean.shape, (2, 1))
        self.assertEqual(var.shape, (2, 1))
        self.assertFalse(np.isnan(mean).any(), "Mean predictions should not contain NaNs.")
        self.assertFalse(np.isnan(var).any(), "Variance predictions should not contain NaNs.")
        self.assertTrue(np.all(var >= 0.0), "Variance predictions should be non-negative.")

if __name__ == "__main__":
    unittest.main()



