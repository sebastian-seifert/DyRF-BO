import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sys
import os

# Adjust path to import evaluate_aleatoric
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestAleatoricEvaluation(unittest.TestCase):
    def test_compute_nll(self):
        from evaluate_aleatoric import compute_nll
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 1.8, 3.2])
        # Test perfect or close variances
        variance = np.array([0.1, 0.1, 0.1])
        nll = compute_nll(y_true, y_pred, variance)
        self.assertTrue(isinstance(nll, float))
        self.assertTrue(np.isfinite(nll))

    def test_generate_heteroscedastic_data(self):
        from evaluate_aleatoric import generate_heteroscedastic_data
        func = lambda x, y: np.sin(x) * np.cos(y)
        X, y, sigma_true = generate_heteroscedastic_data(
            func=func,
            x_range=[0.0, 10.0],
            ndim=2,
            n_samples=100,
            seed=42
        )
        self.assertEqual(X.shape, (100, 2))
        self.assertEqual(y.shape, (100,))
        self.assertEqual(sigma_true.shape, (100,))
        self.assertTrue(np.all(sigma_true >= 0.05))

    def test_evaluate_aleatoric_quality(self):
        from evaluate_aleatoric import evaluate_aleatoric_quality
        func = lambda x, y: np.sin(x) * np.cos(y)
        
        # Small training and testing sets
        X_train, y_train, _ = generate_heteroscedastic_data(func, [0.0, 10.0], 2, 80, 42)
        X_test, y_test, sigma_test_true = generate_heteroscedastic_data(func, [0.0, 10.0], 2, 50, 43)
        
        results = evaluate_aleatoric_quality(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sigma_test_true=sigma_test_true,
            seed=42
        )
        
        # Verify keys in the results dictionary
        self.assertIn("Standard", results)
        self.assertIn("Shaker", results)
        
        for approach in ["Standard", "Shaker"]:
            self.assertIn("pearson_true_var", results[approach])
            self.assertIn("spearman_true_var", results[approach])
            self.assertIn("pearson_sq_res", results[approach])
            self.assertIn("spearman_sq_res", results[approach])
            self.assertIn("mse_true_var", results[approach])
            self.assertIn("mae_true_var", results[approach])
            self.assertIn("nll", results[approach])
            
            # Check metrics are valid floats (or can be None if constant variance occurs, but should be float here)
            self.assertTrue(isinstance(results[approach]["nll"], float))

    def test_bash_script_execution(self):
        import subprocess
        # Run the bash script to make sure it executes without syntax/runtime error
        res = subprocess.run(["bash", "run_aleatoric_evaluation.sh", "--quick"], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.assertEqual(res.returncode, 0, f"Bash script failed with output: {res.stderr}\nStdout: {res.stdout}")

def generate_heteroscedastic_data(func, x_range, ndim, n_samples, seed):
    # Temporary copy of the function logic to allow setUp to run in this test file
    rng = np.random.default_rng(seed)
    X = rng.uniform(x_range[0], x_range[1], size=(n_samples, ndim))
    sigma_true = 0.05 + 0.25 * (np.sin(X[:, 0]) ** 2)
    y_true = func(*[X[:, d] for d in range(ndim)])
    noise = rng.normal(0, sigma_true)
    y = y_true + noise
    return X, y, sigma_true

if __name__ == "__main__":
    unittest.main()
