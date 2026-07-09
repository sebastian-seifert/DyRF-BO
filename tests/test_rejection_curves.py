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
    calculate_random_rejection_curve,
    calculate_naurc,
    calculate_aurc_exact,
    calculate_roc_metrics,
    calculate_jensen_shannon_divergence,
    calculate_mutual_information,
    calculate_aupr
)


class TestRejectionCurves(unittest.TestCase):
    def test_aupr(self):
        from sklearn.metrics import average_precision_score
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        u_good = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.9, 0.8, 0.9, 0.7, 0.9])
        
        aupr = calculate_aupr(y_true, u_good)
        expected_aupr = average_precision_score(y_true, u_good)
        self.assertAlmostEqual(aupr, expected_aupr)

    def test_mi_freedman_diaconis(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        u_good = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.9, 0.8, 0.9, 0.7, 0.9])
        
        mi_fixed = calculate_mutual_information(u_good, y_true, n_bins=50)
        self.assertTrue(0.0 <= mi_fixed <= 1.0)
        
        mi_fd = calculate_mutual_information(u_good, y_true, n_bins="fd")
        self.assertTrue(0.0 <= mi_fd <= 1.0)

    def test_jsd_freedman_diaconis(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        u_good = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.9, 0.8, 0.9, 0.7, 0.9])
        
        jsd_fixed = calculate_jensen_shannon_divergence(u_good, y_true, n_bins=50)
        self.assertTrue(0.0 <= jsd_fixed <= 1.0)
        
        jsd_fd = calculate_jensen_shannon_divergence(u_good, y_true, n_bins="fd")
        self.assertTrue(0.0 <= jsd_fd <= 1.0)

    def test_roc_metrics(self):
        from sklearn.metrics import roc_auc_score
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        u_good = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.9, 0.8, 0.9, 0.7, 0.9])
        
        auroc, fpr95 = calculate_roc_metrics(y_true, u_good)
        expected_auroc = roc_auc_score(y_true, u_good)
        
        self.assertAlmostEqual(auroc, expected_auroc)
        self.assertAlmostEqual(fpr95, 0.0)

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

    def test_naurc_calculation(self):
        rejection_rates = np.linspace(0.0, 0.95, 96)
        
        curve_oracle = calculate_oracle_rejection_curve(self.predictions, self.y_true, rejection_rates)
        curve_good = calculate_rejection_curve(self.uncertainty_good, self.predictions, self.y_true, rejection_rates)
        curve_bad = calculate_rejection_curve(self.uncertainty_bad, self.predictions, self.y_true, rejection_rates)
        curve_random = calculate_random_rejection_curve(self.predictions, self.y_true, rejection_rates, n_shuffles=20, random_state=42)
        
        naurc_good = calculate_naurc(rejection_rates, curve_good, curve_oracle, curve_random)
        naurc_bad = calculate_naurc(rejection_rates, curve_bad, curve_oracle, curve_random)
        naurc_oracle_self = calculate_naurc(rejection_rates, curve_oracle, curve_oracle, curve_random)
        naurc_random_self = calculate_naurc(rejection_rates, curve_random, curve_oracle, curve_random)
        
        # Oracle NAURC must be exactly 0.0
        self.assertAlmostEqual(naurc_oracle_self, 0.0)
        # Random NAURC must be exactly 1.0
        self.assertAlmostEqual(naurc_random_self, 1.0)
        # Good UQ should have NAURC close to 0.0 (and <= 0.1)
        self.assertLessEqual(naurc_good, 0.1)
        # Bad UQ should have NAURC > 1.0
        self.assertGreater(naurc_bad, 1.0)
        
        print(f"\n[Test Result] NAURC Oracle: {naurc_oracle_self:.4f}")
        print(f"[Test Result] NAURC Good:   {naurc_good:.4f}")
        print(f"[Test Result] NAURC Random: {naurc_random_self:.4f}")
        print(f"[Test Result] NAURC Bad:    {naurc_bad:.4f}")

    def test_aurc_exact(self):
        # Exact AURC up to p_max=0.95
        p_max = 0.95
        rejection_rates_dense = np.linspace(0.0, p_max, 1000)
        
        # 1. Oracle
        curve_oracle = calculate_oracle_rejection_curve(self.predictions, self.y_true, rejection_rates_dense)
        aurc_oracle_trapz = calculate_aurc(rejection_rates_dense, curve_oracle)
        aurc_oracle_exact = calculate_aurc_exact(self.predictions - self.y_true, self.predictions, self.y_true, p_max=p_max)
        # Should be very close (trapz with 1000 points is a good approximation of the step function)
        self.assertAlmostEqual(aurc_oracle_trapz, aurc_oracle_exact, places=3)
        
        # 2. Good UQ
        curve_good = calculate_rejection_curve(self.uncertainty_good, self.predictions, self.y_true, rejection_rates_dense)
        aurc_good_trapz = calculate_aurc(rejection_rates_dense, curve_good)
        aurc_good_exact = calculate_aurc_exact(self.uncertainty_good, self.predictions, self.y_true, p_max=p_max)
        self.assertAlmostEqual(aurc_good_trapz, aurc_good_exact, places=3)
        
        # 3. Bad UQ
        curve_bad = calculate_rejection_curve(self.uncertainty_bad, self.predictions, self.y_true, rejection_rates_dense)
        aurc_bad_trapz = calculate_aurc(rejection_rates_dense, curve_bad)
        aurc_bad_exact = calculate_aurc_exact(self.uncertainty_bad, self.predictions, self.y_true, p_max=p_max)
        self.assertAlmostEqual(aurc_bad_trapz, aurc_bad_exact, places=3)

        # 4. Expected relation: Oracle <= Good < Random < Bad
        aurc_random_exact = p_max * np.mean((self.predictions - self.y_true)**2)
        self.assertLessEqual(aurc_oracle_exact, aurc_good_exact)
        self.assertLess(aurc_good_exact, aurc_random_exact)
        self.assertLess(aurc_random_exact, aurc_bad_exact)
        
        print(f"\n[Test Result] Exact AURC Oracle: {aurc_oracle_exact:.4f}")
        print(f"[Test Result] Exact AURC Good:   {aurc_good_exact:.4f}")
        print(f"[Test Result] Exact AURC Random: {aurc_random_exact:.4f}")
        print(f"[Test Result] Exact AURC Bad:    {aurc_bad_exact:.4f}")

if __name__ == "__main__":
    unittest.main()
