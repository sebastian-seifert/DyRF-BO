import unittest
import numpy as np
from synthetic_functions import (
    get_2d_functions,
    get_4d_functions,
    get_6d_functions,
)

class TestSyntheticOODFunctions(unittest.TestCase):
    def test_ackley_2d_registration_and_evaluation(self):
        funcs_2d = get_2d_functions()
        self.assertIn("ackley_2d", funcs_2d, "ackley_2d must be registered in get_2d_functions()")
        
        cfg = funcs_2d["ackley_2d"]
        self.assertIn("func", cfg)
        self.assertIn("gap", cfg)
        self.assertIn("range", cfg)
        self.assertEqual(cfg["gap"], (3.5, 6.5))
        
        # Test evaluation at global minimum (0, 0) -> Ackley(0, 0) == 0.0
        val = cfg["func"](np.array([0.0]), np.array([0.0]))
        self.assertTrue(np.isclose(val[0], 0.0, atol=1e-5))

    def test_rosenbrock_2d_registration_and_evaluation(self):
        funcs_2d = get_2d_functions()
        self.assertIn("rosenbrock_2d", funcs_2d, "rosenbrock_2d must be registered in get_2d_functions()")
        
        cfg = funcs_2d["rosenbrock_2d"]
        self.assertIn("func", cfg)
        self.assertIn("gap", cfg)
        self.assertIn("range", cfg)
        self.assertEqual(cfg["range"], (-2.0, 2.0))
        
        # Test evaluation at global minimum (1, 1) -> Rosenbrock(1, 1) == 0.0
        val = cfg["func"](np.array([1.0]), np.array([1.0]))
        self.assertTrue(np.isclose(val[0], 0.0, atol=1e-5))

    def test_ackley_4d_registration_and_evaluation(self):
        funcs_4d = get_4d_functions()
        self.assertIn("ackley_4d", funcs_4d, "ackley_4d must be registered in get_4d_functions()")
        
        cfg = funcs_4d["ackley_4d"]
        val = cfg["func"](np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
        self.assertTrue(np.isclose(val[0], 0.0, atol=1e-5))

    def test_rosenbrock_4d_registration_and_evaluation(self):
        funcs_4d = get_4d_functions()
        self.assertIn("rosenbrock_4d", funcs_4d, "rosenbrock_4d must be registered in get_4d_functions()")
        
        cfg = funcs_4d["rosenbrock_4d"]
        val = cfg["func"](np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]))
        self.assertTrue(np.isclose(val[0], 0.0, atol=1e-5))

    def test_hartmann_6d_registration_and_evaluation(self):
        funcs_6d = get_6d_functions()
        self.assertIn("hartmann_6d", funcs_6d, "hartmann_6d must be registered in get_6d_functions()")
        
        cfg = funcs_6d["hartmann_6d"]
        self.assertEqual(cfg["range"], (0.0, 1.0))
        
        # Test evaluation at known evaluation point
        x_test = np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])
        val = cfg["func"](x_test[:, 0], x_test[:, 1], x_test[:, 2], x_test[:, 3], x_test[:, 4], x_test[:, 5])
        # Hartmann 6D minimum is approx -3.32237
        self.assertLess(val[0], -3.0)

if __name__ == "__main__":
    unittest.main()
