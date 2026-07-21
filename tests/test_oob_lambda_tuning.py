import os
import sys
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
from data_generator import generate_data


class TestOOBLambdaTuning(unittest.TestCase):
    """
    Test suite for Out-of-Bag (OOB) Negative Log-Likelihood (NLL) calculation
    and continuous lambda tuning using Brent's fminbound method.
    """

    @classmethod
    def setUpClass(cls):
        # Generate 1D synthetic sine wave data
        func_dict = {
            "sin": {
                "func": lambda x: np.sin(x),
                "gap": [4.0, 6.0],
                "range": [0.0, 10.0]
            }
        }
        cls.X_train, cls.y_train, cls.X_test, cls.y_test, _ = generate_data(
            func_dict, "sin", seed=42, points_per_dim=80, gap_type="empty"
        )
        
        # Fit Random Forest with oob_score=True
        cls.rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=3, oob_score=True, random_state=42)
        cls.rf.fit(cls.X_train, cls.y_train)

    def test_compute_oob_nll_returns_valid_float(self):
        """Verify that compute_oob_nll returns a finite positive float for candidate lambdas."""
        uq_engine = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train, device="cpu", topological_decay_lambda=1.0
        )
        
        for lmbda in [0.1, 1.0, 5.0]:
            nll = uq_engine.compute_oob_nll(lmbda)
            self.assertIsInstance(nll, float)
            self.assertFalse(np.isnan(nll))
            self.assertFalse(np.isinf(nll))

    def test_tune_lambda_oob_converges_within_bounds(self):
        """Verify that tune_lambda_oob finds a continuous lambda* within specified bounds."""
        uq_engine = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train, device="cpu", topological_decay_lambda=1.0
        )
        
        bounds = (0.001, 20.0)
        best_lambda = uq_engine.tune_lambda_oob(bounds=bounds, xtol=1e-4)
        
        self.assertIsInstance(best_lambda, float)
        self.assertTrue(bounds[0] <= best_lambda <= bounds[1])
        self.assertAlmostEqual(uq_engine.topological_decay_lambda, best_lambda, places=5)

    def test_small_sample_fallback(self):
        """Verify fallback to lambda=1.0 for small sample sizes N < 3."""
        X_small = self.X_train[:2]
        y_small = self.y_train[:2]
        rf_small = RandomForestRegressor(n_estimators=5, random_state=42)
        rf_small.fit(X_small, y_small)
        
        uq_engine = GPUProximityRegressionUQ(
            rf_small, X_small, y_small, device="cpu", topological_decay_lambda=1.0
        )
        best_lambda = uq_engine.tune_lambda_oob(bounds=(0.001, 20.0))
        self.assertEqual(best_lambda, 1.0)

    def test_uq_pipeline_integration(self):
        """Verify that Proximity_Auto_Lambda approach runs cleanly in Uncertainty_Quantification.py."""
        from Uncertainty_Quantification import run_single_test
        func_dict = {
            "sin": {
                "func": lambda x: np.sin(x),
                "gap": [4.0, 6.0],
                "range": [0.0, 10.0]
            }
        }
        res, timings = run_single_test(
            func_dict, "sin", seed=42, approaches=["Proximity_Auto_Lambda"],
            rf_config=1, k_neighbors="auto", gap_type="empty"
        )
        self.assertIn("Proximity_Auto_Lambda", res)
        self.assertIsInstance(res["Proximity_Auto_Lambda"]["auroc"], (float, np.floating))


if __name__ == "__main__":
    unittest.main()

