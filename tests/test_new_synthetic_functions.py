import unittest
import numpy as np
import sys
import os

# Adjust path to import synthetic_functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestNewSyntheticFunctions(unittest.TestCase):
    def test_dimensions_11_to_15(self):
        # We try to import the new functions
        try:
            from synthetic_functions import (
                get_11d_functions,
                get_12d_functions,
                get_13d_functions,
                get_14d_functions,
                get_15d_functions
            )
        except ImportError as e:
            self.fail(f"Failed to import new dimension functions: {e}")

        getters = [
            (11, get_11d_functions),
            (12, get_12d_functions),
            (13, get_13d_functions),
            (14, get_14d_functions),
            (15, get_15d_functions)
        ]

        for dim, getter in getters:
            funcs = getter()
            self.assertEqual(len(funcs), 1)
            name = list(funcs.keys())[0]
            func_entry = funcs[name]
            
            self.assertIn("func", func_entry)
            self.assertIn("gap", func_entry)
            self.assertIn("range", func_entry)
            
            func_obj = func_entry["func"]
            self.assertEqual(func_obj.__code__.co_argcount, dim)
            
            # Test calling the function
            dummy_inputs = [np.ones(5) for _ in range(dim)]
            out = func_obj(*dummy_inputs)
            self.assertEqual(out.shape, (5,))

    def test_aleatoric_approaches(self):
        from Uncertainty_Quantification import run_single_test
        from synthetic_functions import get_1d_functions
        
        funcs = get_1d_functions()
        res, timings = run_single_test(
            func_dict=funcs,
            func_name="sin",
            seed=42,
            approaches=["Standard_Aleatoric", "Shaker_Aleatoric"],
            rf_config=1,
            gap_type="empty",
            ood_type="manifold"
        )
        
        self.assertIn("Standard_Aleatoric", res)
        self.assertIn("Shaker_Aleatoric", res)
        
        # Check that we have auroc and other keys
        self.assertIsNotNone(res["Standard_Aleatoric"]["auroc"])
        self.assertIsNotNone(res["Shaker_Aleatoric"]["auroc"])

if __name__ == "__main__":
    unittest.main()
