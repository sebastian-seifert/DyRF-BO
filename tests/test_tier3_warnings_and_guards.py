import warnings
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from Epistemic_Quantifier import EpistemicQuantifier
from Credal_Regression_UQ import CredalRegressionUQ
from rf_dynamic.dynamic_rf_surrogate import DynamicRFSurrogate


class TestTier3WarningsAndGuards(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Small dataset designed to produce single-sample leaves
        self.X_small = np.random.uniform(-2, 2, (6, 2))
        self.y_small = (self.X_small[:, 0]**2 + self.X_small[:, 1]**2).ravel()
        
        self.rf = RandomForestRegressor(n_estimators=5, min_samples_leaf=1, random_state=42)
        self.rf.fit(self.X_small, self.y_small)

    def test_zero_runtime_warning_leaf_sample_scaling_epistemic_quantifier(self):
        """Verify that leaf sample scaling in EpistemicQuantifier produces ZERO RuntimeWarnings when N=1."""
        eq = EpistemicQuantifier(self.rf, self.X_small, self.y_small)
        
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            variances = eq._base_calc_per_tree_variance(self.X_small)
            
            # Filter for RuntimeWarnings specifically
            runtime_warnings = [w for w in recorded_warnings if issubclass(w.category, RuntimeWarning)]
            self.assertEqual(
                len(runtime_warnings), 0,
                f"Expected 0 RuntimeWarnings but caught: {[str(w.message) for w in runtime_warnings]}"
            )
            
        self.assertEqual(variances.shape, (5, 6))
        self.assertTrue(np.all(variances >= 0.0))
        self.assertFalse(np.any(np.isnan(variances)))

    def test_zero_runtime_warning_leaf_sample_scaling_credal_regression(self):
        """Verify that leaf sample scaling in CredalRegressionUQ produces ZERO RuntimeWarnings when N=1."""
        credal = CredalRegressionUQ(self.rf, self.X_small, self.y_small)
        
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            means, variances, counts = credal._calc_leaf_stats(self.X_small)
            
            runtime_warnings = [w for w in recorded_warnings if issubclass(w.category, RuntimeWarning)]
            self.assertEqual(
                len(runtime_warnings), 0,
                f"Expected 0 RuntimeWarnings but caught: {[str(w.message) for w in runtime_warnings]}"
            )
            
        self.assertEqual(means.shape, (5, 6))
        self.assertEqual(variances.shape, (5, 6))
        self.assertEqual(counts.shape, (5, 6))
        self.assertTrue(np.all(variances >= 0.0))
        self.assertFalse(np.any(np.isnan(variances)))

    def test_early_bo_warmup_parameter_guards_dynamic_rf(self):
        """Verify that DynamicRFSurrogate safely clamps min_samples_leaf on tiny warmup datasets (N=2, 3)."""
        surrogate = DynamicRFSurrogate(
            extractor_name="standard_disagreement",
            enable_adaptation=True,
            min_samples_leaf_base=10,
            max_features_base=0.5
        )
        
        # N = 2
        X_2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        y_2 = np.array([1.0, 2.0])
        
        surrogate.fit(X_2, y_2)
        self.assertIsNotNone(surrogate.model)
        self.assertLessEqual(surrogate.model.min_samples_leaf, max(1, len(X_2) // 2))
        
        preds, unc = surrogate.predict(X_2)
        self.assertEqual(len(preds), 2)
        self.assertEqual(len(unc), 2)

    def test_dynamic_rf_extreme_features_clamping(self):
        """Verify max_features boundary clamping to [0.1, 1.0]."""
        surrogate = DynamicRFSurrogate(
            extractor_name="standard_disagreement",
            enable_adaptation=False,
            min_samples_leaf_base=2,
            max_features_base=2.5  # Clamped to 1.0
        )
        surrogate.fit(self.X_small, self.y_small)
        self.assertLessEqual(surrogate.model.max_features, 1.0)


if __name__ == "__main__":
    unittest.main()
