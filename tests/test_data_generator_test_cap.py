import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_functions import get_1d_functions, get_10d_functions
from data_generator import generate_data

class TestDataGeneratorTestCap(unittest.TestCase):
    def test_hypercube_test_set_capping_1d(self):
        funcs = get_1d_functions()
        # 1D: n_samples = 1200 -> n_test = min(int(1200 * 0.3), 1000) = 360
        X_train, y_train, X_test, y_test, y_true_binary = generate_data(
            funcs, "sin", seed=42, ood_type="hypercube"
        )
        expected_n_test = min(int(1200 * 0.3), 1000)
        self.assertEqual(len(X_test), expected_n_test)
        self.assertEqual(len(y_test), expected_n_test)
        self.assertEqual(len(y_true_binary), expected_n_test)
        
        # 70% ID (0), 30% OOD (1)
        expected_n_id = int(expected_n_test * 0.7)
        expected_n_ood = expected_n_test - expected_n_id
        self.assertEqual(np.sum(y_true_binary == 0), expected_n_id)
        self.assertEqual(np.sum(y_true_binary == 1), expected_n_ood)

    def test_hypercube_test_set_capping_10d(self):
        funcs = get_10d_functions()
        # 10D: n_samples = 10000 -> n_test = min(int(10000 * 0.3), 1000) = 1000
        X_train, y_train, X_test, y_test, y_true_binary = generate_data(
            funcs, "sin_cos_10d", seed=42, ood_type="hypercube"
        )
        expected_n_test = 1000
        self.assertEqual(len(X_test), expected_n_test)
        self.assertEqual(len(y_test), expected_n_test)
        self.assertEqual(len(y_true_binary), expected_n_test)

        expected_n_id = int(1000 * 0.7)
        expected_n_ood = 1000 - expected_n_id
        self.assertEqual(np.sum(y_true_binary == 0), expected_n_id)
        self.assertEqual(np.sum(y_true_binary == 1), expected_n_ood)

if __name__ == "__main__":
    unittest.main()
