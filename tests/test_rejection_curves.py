import os
import sys
import unittest
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (
    calculate_rejection_curve,
    calculate_aurc,
    calculate_oracle_rejection_curve,
    calculate_random_rejection_curve
)

class TestRejectionCurves(unittest.TestCase):
    def setUp(self):
        # Setup synthetic inputs
        # We have 100 test samples
        self.y_true = np.linspace(0.0, 10.0, 100)
        # Predictions have varying errors: first 50 points have small errors, last 50 points have large errors
        self.predictions = self.y_true.copy()
        self.predictions[50:] += 5.0 # Add a large bias to the second half
        
        # Uncertainty is correlated with the error: high uncertainty for the second half
        # (This is a good UQ model)
        self.uncertainty_good = np.zeros(100)
        self.uncertainty_good[50:] = 10.0
        
        # Bad UQ model: high uncertainty for the first half (inversely correlated with error)
        self.uncertainty_bad = np.zeros(100)
        self.uncertainty_bad[:50] = 10.0
        
        # Rejection rates from 0% to 90%
        self.rejection_rates = np.linspace(0.0, 0.90, 10)

    def test_rejection_curve_sorting(self):
        # For the good UQ model, at 0% rejection, the MSE is:
        # (50 * 0^2 + 50 * 5^2) / 100 = 1250 / 100 = 12.5
        curve_good = calculate_rejection_curve(
            self.uncertainty_good, self.predictions, self.y_true, self.rejection_rates, loss_type="MSE"
        )
        self.assertAlmostEqual(curve_good[0], 12.5)
        
        # At 50% rejection, we should reject the 50 points with high uncertainty (the ones with error 5.0).
        # The remaining 50 points should have error 0.0, so MSE should drop to 0.0.
        self.assertAlmostEqual(curve_good[5], 0.0)

    def test_aurc_order(self):
        rejection_rates = np.linspace(0.0, 0.95, 96)
        
        curve_oracle = calculate_oracle_rejection_curve(
            self.predictions, self.y_true, rejection_rates, loss_type="MSE"
        )
        curve_good = calculate_rejection_curve(
            self.uncertainty_good, self.predictions, self.y_true, rejection_rates, loss_type="MSE"
        )
        curve_bad = calculate_rejection_curve(
            self.uncertainty_bad, self.predictions, self.y_true, rejection_rates, loss_type="MSE"
        )
        curve_random = calculate_random_rejection_curve(
            self.predictions, self.y_true, rejection_rates, loss_type="MSE", n_shuffles=20, random_state=42
        )
        
        aurc_oracle = calculate_aurc(rejection_rates, curve_oracle)
        aurc_good = calculate_aurc(rejection_rates, curve_good)
        aurc_bad = calculate_aurc(rejection_rates, curve_bad)
        aurc_random = calculate_aurc(rejection_rates, curve_random)
        
        # We expect: Oracle <= Good < Random < Bad
        self.assertLessEqual(aurc_oracle, aurc_good)
        self.assertLess(aurc_good, aurc_random)
        self.assertLess(aurc_random, aurc_bad)
        
        print(f"\n[Test Result] AURC Oracle: {aurc_oracle:.4f}")
        print(f"[Test Result] AURC Good UQ: {aurc_good:.4f}")
        print(f"[Test Result] AURC Random:  {aurc_random:.4f}")
        print(f"[Test Result] AURC Bad UQ:  {aurc_bad:.4f}")

    def test_invalid_loss_type(self):
        with self.assertRaises(ValueError):
            calculate_rejection_curve(
                self.uncertainty_good, self.predictions, self.y_true, self.rejection_rates, loss_type="INVALID"
            )

if __name__ == "__main__":
    unittest.main()
