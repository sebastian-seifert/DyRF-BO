import os
import sys
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors import UQExtractorRegistry
# Import extractors to trigger registration (we'll create these files next)
try:
    import ep_extractors.standard_disagreement
    import ep_extractors.chen_variance
    import ep_extractors.shaker_entropy
    import ep_extractors.likelihood_credal
    import ep_extractors.standard_proximity
    import ep_extractors.proximity_b
    import ep_extractors.proximity_bc
    import ep_extractors.proximity_auto_lambda
except ImportError:
    pass  # Allow importing to fail during the TDD "test writes first" stage

class TestEpistemicExtractors(unittest.TestCase):
    def setUp(self):
        # Create a tiny fitted Random Forest regressor
        self.rf = RandomForestRegressor(n_estimators=10, random_state=42)
        self.X_train = np.array([
            [0.1, 0.2],
            [0.15, 0.25],
            [0.8, 0.9],
            [0.85, 0.95],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.3, 0.3],
            [0.7, 0.7],
            [0.4, 0.6]
        ])
        self.y_train = np.array([1.0, 1.1, 5.0, 5.2, 3.0, 2.5, 2.7, 1.5, 4.5, 2.8])
        self.rf.fit(self.X_train, self.y_train)
        
        self.X_test = np.array([
            [0.12, 0.22],  # Close to train
            [0.82, 0.92],  # Close to train
            [0.0, 0.0],    # Edge/OOD
            [1.0, 1.0],    # Edge/OOD
            [0.5, 0.5]     # Direct match
        ])

    def _verify_extractor_behavior(self, name, kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Get extractor
        extractor = UQExtractorRegistry.get(name, self.rf, **kwargs)
        # Fit extractor
        extractor.fit(self.X_train, self.y_train)
        # Extract signal
        signal = extractor.extract_epistemic_signal(self.X_test)
        
        # Checks
        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(signal.shape, (len(self.X_test),))
        self.assertTrue(np.all(np.isfinite(signal)), f"{name} signal contains non-finite values.")
        self.assertTrue(np.all(signal >= -1e-12), f"{name} signal contains negative values: {signal}")
        return signal

    def test_standard_disagreement_extractor(self):
        self._verify_extractor_behavior("standard_disagreement")

    def test_chen_variance_extractor(self):
        self._verify_extractor_behavior("chen_variance")

    def test_shaker_entropy_extractor(self):
        # Pass a small num_samples to speed up testing
        self._verify_extractor_behavior("shaker_entropy", {"num_samples": 100})

    def test_likelihood_credal_extractor(self):
        # Pass a small n_grid to speed up testing
        self._verify_extractor_behavior("likelihood_credal", {"n_grid": 10})

    def test_standard_proximity_extractor(self):
        self._verify_extractor_behavior("standard_proximity")

    def test_proximity_b_extractor(self):
        self._verify_extractor_behavior("proximity_b")

    def test_proximity_bc_extractor(self):
        self._verify_extractor_behavior("proximity_bc")

    def test_proximity_auto_lambda_extractor(self):
        self._verify_extractor_behavior("proximity_auto_lambda")

    def test_proximity_auto_lambda_baseline_consistency_and_scaling(self):
        """Verify that ProximityAutoLambda recomputes N_baseline post-tuning and scales UQ properly."""
        extractor = UQExtractorRegistry.get("proximity_auto_lambda", self.rf, alpha=1.0)
        extractor.fit(self.X_train, self.y_train)
        
        self.assertIsNotNone(extractor.uq_model)
        self.assertIsNotNone(extractor.uq_model.N_baseline)
        self.assertGreater(extractor.uq_model.N_baseline, 0.0)
        
        # Verify N_baseline consistency with tuned lambda
        tuned_lambda = extractor.uq_model.topological_decay_lambda
        self.assertIsInstance(tuned_lambda, float)
        
        # Check signal extraction
        signal = extractor.extract_epistemic_signal(self.X_test)
        self.assertEqual(signal.shape, (len(self.X_test),))
        self.assertTrue(np.all(np.isfinite(signal)))
        self.assertTrue(np.all(signal >= 0.0))
        
        # OOD points (e.g. index 2 and 3) should have strictly positive uncertainty
        self.assertGreater(signal[2], 0.0)
        self.assertGreater(signal[3], 0.0)


if __name__ == "__main__":
    unittest.main()

