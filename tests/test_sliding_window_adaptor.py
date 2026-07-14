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
            n_base=50,
            n_min=10,
            n_max=100,
            gamma=1.0,
            depth_base=10,
            depth_min=5,
            depth_max=15,
            beta=5.0
        )

    def test_initial_values(self):
        # Without any updates, parameters should equal base values
        n_trees, depth = self.adaptor.get_next_parameters()
        self.assertEqual(n_trees, 50)
        self.assertEqual(depth, 10)

    def test_incremental_updates(self):
        # Update with low epistemic uncertainty
        # Let's say candidate pool has low epistemic values (e.g. around 0.1)
        raw_signals_1 = np.array([0.05, 0.08, 0.1, 0.02, 0.05])
        scaled_signals_1 = self.adaptor.update_and_normalize(raw_signals_1)
        
        # In first step, max_q95 in window is q95(raw_signals_1)
        q95_1 = np.percentile(raw_signals_1, 95)
        # Verify scaled signals are clipped to 1.0, and mostly divided by q95_1
        np.testing.assert_array_less(scaled_signals_1, 1.0001)
        
        # Get parameters after 1 update
        n_trees_1, depth_1 = self.adaptor.get_next_parameters()
        # Since mean uncertainty is > 0, n_trees should adjust, but stay within bounds
        self.assertTrue(10 <= n_trees_1 <= 100)
        self.assertTrue(5 <= depth_1 <= 15)
        
        # Perform more updates to fill window size (3)
        self.adaptor.update_and_normalize(np.array([0.2, 0.25, 0.3, 0.15]))
        self.adaptor.update_and_normalize(np.array([0.01, 0.02, 0.03, 0.02]))
        
        # Verify queue size is capped at 3
        self.assertEqual(len(self.adaptor.q95_history), 3)
        
        # Perform 4th update (slides window, removing 1st update)
        self.adaptor.update_and_normalize(np.array([0.4, 0.5, 0.45, 0.35]))
        self.assertEqual(len(self.adaptor.q95_history), 3)

    def test_clamping_bounds(self):
        # Re-initialize with high beta to test depth upper clamping
        self.adaptor = SlidingWindowRFAdaptor(
            window_size=3,
            n_base=50,
            n_min=10,
            n_max=100,
            gamma=2.0,
            depth_base=10,
            depth_min=5,
            depth_max=15,
            beta=20.0
        )
        # Alternate updates to maximize variance: [0.0, 1.0, 0.0]
        # To get mean_scaled = 0.0:
        self.adaptor.update_and_normalize(np.array([1e-9, 1e-9, 1e-9]))
        # To get mean_scaled = 1.0:
        self.adaptor.update_and_normalize(np.array([20.0, 20.0, 20.0]))
        # To get mean_scaled = 0.0:
        self.adaptor.update_and_normalize(np.array([1e-9, 1e-9, 1e-9]))

        n_trees, depth = self.adaptor.get_next_parameters()
        self.assertEqual(depth, 15)    # Capped at depth_max

        # Force very low values to test lower bounds
        for _ in range(5):
            self.adaptor.update_and_normalize(np.array([1e-9, 1e-9, 1e-9]))
        n_trees, depth = self.adaptor.get_next_parameters()
        self.assertTrue(n_trees >= 10)
        self.assertTrue(depth >= 5)

if __name__ == "__main__":
    unittest.main()
