import os
import sys
import unittest
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ConfigSpace import ConfigurationSpace, Float
from smac.scenario import Scenario
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest

class TestSMAC3RFTreeExtraction(unittest.TestCase):
    def setUp(self):
        self.cs = ConfigurationSpace()
        self.cs.add([
            Float("x1", (-5.0, 5.0), default=0.0),
            Float("x2", (-5.0, 5.0), default=0.0)
        ])
        np.random.seed(42)
        self.X_train = np.random.uniform(-5.0, 5.0, (30, 2))
        self.y_train = (self.X_train[:, 0]**2 + np.sin(self.X_train[:, 1])).reshape(-1, 1)

    def test_smac3_rf_direct_leaf_and_tree_extraction(self):
        """Verify leaf assignments, decision paths, and estimators_ can be directly extracted from SMAC3 self._rf."""
        rf = CustomUncertaintyRandomForest(
            uncertainty_func="standard_disagreement",
            configspace=self.cs,
            n_trees=10,
            seed=42
        )
        rf.train(self.X_train, self.y_train)
        
        # Verify underlying fitted model is EPMRandomForest
        self.assertIsNotNone(rf._rf)
        self.assertEqual(len(rf._rf.estimators_), 10)
        
        X_test = np.random.uniform(-5.0, 5.0, (5, 2))
        
        # 1. Leaf Assignments via apply()
        leaf_ids = rf._rf.apply(X_test)
        self.assertEqual(leaf_ids.shape, (5, 10))
        
        # 2. Decision Paths via decision_path()
        indicator, recipe = rf._rf.decision_path(X_test)
        self.assertEqual(indicator.shape[0], 5)
        
        # 3. Individual Tree Predictions via all_trees_pred()
        tree_preds = rf._rf.all_trees_pred(X_test)
        self.assertEqual(tree_preds.shape, (5, 10))

    def test_custom_walking_distance_and_leaf_uncertainty_function(self):
        """
        Verify that a custom user-defined function extracting leaf walking distances 
        and leaf sample counts directly from SMAC3's self._rf can be injected.
        """
        def custom_topological_lca_distance_uq(smac_rf_model, X, y_train):
            """
            Custom UQ function computing mean tree depth LCA walking distance 
            from test candidate leaves to training sample leaves directly on SMAC3's self._rf.
            """
            estimators = smac_rf_model.estimators_
            n_trees = len(estimators)
            n_samples = X.shape[0]
            
            cand_leaves = smac_rf_model.apply(X)  # (n_cand, n_trees)
            train_leaves = smac_rf_model.apply(smac_rf_model.trainX)  # (n_train, n_trees)
            
            # Simple topological distance metric: proportion of trees where test point shares leaf with training sample
            shared_leaf_counts = np.zeros(n_samples)
            for i in range(n_samples):
                # Count max shared leaves across all training samples
                matches = (cand_leaves[i, :] == train_leaves).sum(axis=1)  # (n_train,)
                max_shared = np.max(matches)
                # Unvisited/isolated candidate points share fewer leaves -> higher topological distance (uncertainty)
                shared_leaf_counts[i] = n_trees - max_shared
                
            return shared_leaf_counts

        rf = CustomUncertaintyRandomForest(
            uncertainty_func=custom_topological_lca_distance_uq,
            configspace=self.cs,
            n_trees=15,
            seed=42
        )
        rf.train(self.X_train, self.y_train)
        
        X_test = np.random.uniform(-5.0, 5.0, (10, 2))
        mean, var = rf._predict(X_test)
        
        self.assertEqual(mean.shape, (10, 1))
        self.assertEqual(var.shape, (10, 1))
        self.assertTrue(np.all(var >= 0.0))

    def test_smac3_hpo_facade_with_leaf_walking_distance_uq(self):
        """Verify native SMAC3 HPOFacade executes cleanly using custom leaf walking distance UQ."""
        def leaf_walking_distance_uq(smac_rf_model, X, y_train):
            cand_leaves = smac_rf_model.apply(X)
            train_leaves = smac_rf_model.apply(smac_rf_model.trainX)
            
            # Compute topological distance to nearest training point in tree leaf space
            distances = []
            for i in range(len(X)):
                matches = (cand_leaves[i, :] == train_leaves).sum(axis=1)
                distances.append(float(15 - np.max(matches)))
            return np.array(distances)

        scenario = Scenario(configspace=self.cs, n_trials=5, seed=42)
        
        def target_func(config, seed=0):
            return config["x1"]**2 + config["x2"]**2

        smac = HPOFacade(
            scenario=scenario,
            target_function=target_func,
            model=CustomUncertaintyRandomForest(
                uncertainty_func=leaf_walking_distance_uq,
                configspace=self.cs,
                n_trees=15
            ),
            overwrite=True
        )
        
        incumbent = smac.optimize()
        self.assertIsNotNone(incumbent)

if __name__ == "__main__":
    unittest.main()
