import os
import sys
import unittest
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rf_dynamic.sliding_window_adaptor import SlidingWindowRFAdaptor

class TestSlidingWindowAdaptor(unittest.TestCase):
    def setUp(self):
        self.adaptor = SlidingWindowRFAdaptor(
            window_size=3,
            min_samples_leaf_base=2,
            min_samples_leaf_min=1,
            min_samples_leaf_max=10,
            alpha=5.0,
            max_features_base=0.5,
            max_features_min=0.1,
            max_features_max=0.8,
            eta=0.5
        )

    def test_initial_values(self):
        # Without any updates, parameters should equal base values
        leaf, max_feat = self.adaptor.get_next_parameters()
        self.assertEqual(leaf, 2)
        self.assertEqual(max_feat, 0.5)

    def test_incremental_updates(self):
        # Update with low epistemic uncertainty
        raw_signals_1 = np.array([0.05, 0.08, 0.1, 0.02, 0.05])
        scaled_signals_1 = self.adaptor.update_and_normalize(raw_signals_1, n_samples=100)
        
        # Verify scaled signals are clipped to 1.0
        np.testing.assert_array_less(scaled_signals_1, 1.0001)
        
        # Get parameters after 1 update
        leaf_1, max_feat_1 = self.adaptor.get_next_parameters()
        # Verify they stay within limits
        self.assertTrue(1 <= leaf_1 <= 10)
        self.assertTrue(0.1 <= max_feat_1 <= 0.8)
        
        # Perform more updates to fill window size
        self.adaptor.update_and_normalize(np.array([0.2, 0.25, 0.3, 0.15]), n_samples=100)
        self.adaptor.update_and_normalize(np.array([0.01, 0.02, 0.03, 0.02]), n_samples=100)
        
        self.assertEqual(len(self.adaptor.q95_history), 3)

    def test_clamping_bounds_and_dataset_size_capping(self):
        # Re-initialize to maximize leaf size scaling
        self.adaptor = SlidingWindowRFAdaptor(
            window_size=3,
            min_samples_leaf_base=2,
            min_samples_leaf_min=1,
            min_samples_leaf_max=10,
            alpha=20.0,  # high alpha
            max_features_base=0.5,
            max_features_min=0.1,
            max_features_max=0.8,
            eta=20.0     # high eta
        )
        
        # Max uncertainty: force mean_scaled = 1.0
        self.adaptor.update_and_normalize(np.array([20.0, 20.0, 20.0]), n_samples=100)
        self.adaptor.update_and_normalize(np.array([20.0, 20.0, 20.0]), n_samples=100)
        self.adaptor.update_and_normalize(np.array([20.0, 20.0, 20.0]), n_samples=100)

        leaf, max_feat = self.adaptor.get_next_parameters()
        # With n_samples=100, the cap is 100 // 4 = 25. min_samples_leaf_max is 10.
        # So it should be capped at min_samples_leaf_max (10).
        self.assertEqual(leaf, 10)
        # max_features should be clamped at max_features_min (0.1) under high uncertainty
        self.assertAlmostEqual(max_feat, 0.1)

        # Now test with very small dataset size (e.g. n_samples=8) where the cap is 8 // 4 = 2.
        # Even with high uncertainty, min_samples_leaf must not exceed 2.
        self.adaptor.update_and_normalize(np.array([20.0, 20.0, 20.0]), n_samples=8)
        self.adaptor.update_and_normalize(np.array([20.0, 20.0, 20.0]), n_samples=8)
        self.adaptor.update_and_normalize(np.array([20.0, 20.0, 20.0]), n_samples=8)
        
        leaf, max_feat = self.adaptor.get_next_parameters()
        self.assertEqual(leaf, 2)

if __name__ == "__main__":
    unittest.main()
