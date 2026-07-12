import unittest
import numpy as np
from synthetic_functions import get_1d_functions, get_15d_functions
from data_generator import generate_data

class TestDataGenerator(unittest.TestCase):
    def test_unit_variance_normalization(self):
        funcs_1d = get_1d_functions()
        funcs_15d = get_15d_functions()
        
        # Test 1D function
        X_train, y_train, X_test, y_test, y_true_binary = generate_data(
            funcs_1d, "sin", seed=42, ood_type="manifold"
        )
        # The training targets y_train should not be identical to 0 or contain NaNs
        self.assertFalse(np.isnan(y_train).any())
        self.assertFalse(np.isnan(y_test).any())
        
        # Test 15D function
        # Previously, in 15D, the signal variance of the product of sines and cosines decayed to almost zero
        # causing the training targets to be purely noise.
        # Now, with normalization, the signal component should be scaled back to unit variance.
        name_15d = list(funcs_15d.keys())[0]
        X_train_15, y_train_15, X_test_15, y_test_15, y_true_binary_15 = generate_data(
            funcs_15d, name_15d, seed=42, ood_type="manifold"
        )
        
        self.assertFalse(np.isnan(y_train_15).any())
        self.assertFalse(np.isnan(y_test_15).any())
        
        # Check standard deviation is reasonable (close to sqrt(1.0 + 0.1^2) ~ 1.005)
        # Note: training targets have noise added, so variance is Var(y_clean) + Var(noise) = 1.0 + 0.01 = 1.01
        self.assertTrue(0.8 < np.std(y_train_15) < 1.2)
        self.assertTrue(0.8 < np.std(y_train) < 1.2)

if __name__ == "__main__":
    unittest.main()
