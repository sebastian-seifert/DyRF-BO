import os
import sys
import unittest
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_functions import get_branin_hartmann_functions
from data_generator import generate_data


class TestBraninHartmannFunctions(unittest.TestCase):
    """
    Test suite for Branin (2D), Hartmann-3D, and Hartmann-6D synthetic functions.
    """

    def setUp(self):
        self.funcs = get_branin_hartmann_functions()

    def test_keys_exist(self):
        """Verify that get_branin_hartmann_functions returns branin, hartmann3, and hartmann6."""
        self.assertIn("branin", self.funcs)
        self.assertIn("hartmann3", self.funcs)
        self.assertIn("hartmann6", self.funcs)

    def test_branin_global_minima_value(self):
        """Verify Branin output at known global minima (mapped coordinates)."""
        branin_fn = self.funcs["branin"]["func"]
        # Global minimum at x1_real = pi, x2_real = 2.275
        # Mapped to u1 = pi + 5, u2 = 2.275 / 1.5
        u1 = np.pi + 5.0
        u2 = 2.275 / 1.5
        val = branin_fn(u1, u2)
        self.assertAlmostEqual(val, 0.397887, places=4)

    def test_hartmann3_global_minimum_value(self):
        """Verify Hartmann-3D output at its global minimum."""
        h3_fn = self.funcs["hartmann3"]["func"]
        # Global minimum at x = (0.132078, 0.792743, 0.375733), min value = -3.86278
        val = h3_fn(0.132078, 0.792743, 0.375733)
        self.assertAlmostEqual(val, -3.86278, places=1)

    def test_hartmann6_global_minimum_value(self):
        """Verify Hartmann-6D output at its global minimum."""
        h6_fn = self.funcs["hartmann6"]["func"]
        # Global minimum at x = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573), min value = -3.32237
        val = h6_fn(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
        self.assertAlmostEqual(val, -3.32237, places=4)

    def test_data_generator_compatibility(self):
        """Verify data_generator works seamlessly with Branin and Hartmann functions."""
        for name in ["branin", "hartmann3", "hartmann6"]:
            X_tr, y_tr, X_te, y_te, y_bin = generate_data(
                self.funcs, name, seed=42, gap_type="empty"
            )
            self.assertGreater(len(X_tr), 0)
            self.assertEqual(len(X_tr), len(y_tr))
            self.assertEqual(len(X_te), len(y_te))
            self.assertEqual(len(X_te), len(y_bin))


if __name__ == "__main__":
    unittest.main()
