import unittest
import numpy as np
from metrics import calculate_nlpd

class TestMetrics(unittest.TestCase):
    def test_calculate_nlpd_standard(self):
        # Normal inputs
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        y_var = np.array([0.04, 0.09, 0.16])
        
        # Calculate expected NLPD analytically
        # term1 = 0.5 * ln(2 * pi * var)
        # term2 = (y_true - y_pred)^2 / (2 * var)
        term1 = 0.5 * np.log(2.0 * np.pi * y_var)
        term2 = ((y_true - y_pred) ** 2) / (2.0 * y_var)
        expected = np.mean(term1 + term2)
        
        actual = calculate_nlpd(y_true, y_pred, y_var)
        self.assertAlmostEqual(actual, expected, places=5)

    def test_calculate_nlpd_zero_variance(self):
        # Test numerical stability with zero or negative variance
        y_true = np.array([1.0])
        y_pred = np.array([1.0])
        y_var = np.array([0.0])
        
        # Should not crash or produce NaN/inf, should clip variance
        actual = calculate_nlpd(y_true, y_pred, y_var)
        self.assertTrue(np.isfinite(actual))

if __name__ == "__main__":
    unittest.main()
