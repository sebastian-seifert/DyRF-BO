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

    def test_compute_oob_nll_alias_and_default_batch_size(self):
        """Verify that _compute_oob_nll exists as an alias and matches compute_oob_nll with B=256 default."""
        uq_engine = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train, device="cpu", topological_decay_lambda=1.0
        )
        self.assertTrue(hasattr(uq_engine, "_compute_oob_nll"))
        nll_public = uq_engine.compute_oob_nll(1.2)
        nll_private = uq_engine._compute_oob_nll(1.2)
        nll_b256 = uq_engine.compute_oob_nll(1.2, batch_size=256)
        self.assertEqual(nll_public, nll_private)
        self.assertEqual(nll_public, nll_b256)

    def test_compute_oob_nll_chunking_parity(self):
        """Verify that chunked OOB NLL calculation yields exactly the same values across multiple batch sizes."""
        uq_engine = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train, device="cpu", topological_decay_lambda=1.0
        )
        # Check parity across varied chunk batch sizes: 1, 5, 16, 64, 256, 1000
        nll_ref = uq_engine.compute_oob_nll(1.5, batch_size=1000)
        for b_size in [1, 5, 16, 64, 256]:
            nll_chunked = uq_engine.compute_oob_nll(1.5, batch_size=b_size)
            self.assertAlmostEqual(nll_chunked, nll_ref, places=6)

    def test_oob_chunking_memory_stability_large_sample(self):
        """Verify that OOB NLL calculation with B=256 runs stably without OOM on larger sample size (N=600)."""
        np.random.seed(42)
        X_large = np.random.uniform(0.0, 10.0, size=(600, 3))
        y_large = np.sin(X_large[:, 0]) + 0.5 * np.cos(X_large[:, 1]) + np.random.normal(0, 0.1, size=600)
        rf_large = RandomForestRegressor(n_estimators=15, min_samples_leaf=3, oob_score=True, random_state=42)
        rf_large.fit(X_large, y_large)

        uq_large = GPUProximityRegressionUQ(
            rf_large, X_large, y_large, device="cpu", topological_decay_lambda=1.0
        )
        # Compute with default B=256 and small B=64 chunking
        nll_256 = uq_large._compute_oob_nll(1.0, batch_size=256)
        nll_64 = uq_large._compute_oob_nll(1.0, batch_size=64)
        self.assertIsInstance(nll_256, float)
        self.assertFalse(np.isnan(nll_256))
        self.assertFalse(np.isinf(nll_256))
        self.assertAlmostEqual(nll_256, nll_64, places=6)


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

    def test_tune_lambda_oob_recomputes_baseline_density(self):
        """Verify that tune_lambda_oob recomputes N_baseline so it is consistent with the optimized lambda."""
        uq_engine = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train, device="cpu", topological_decay_lambda=0.01
        )
        uq_engine.fit()
        initial_baseline = uq_engine.N_baseline
        initial_lambda = uq_engine.topological_decay_lambda

        # Tune lambda to a distinct value
        best_lambda = uq_engine.tune_lambda_oob(bounds=(0.5, 5.0), xtol=1e-4)
        self.assertNotEqual(best_lambda, initial_lambda)
        self.assertEqual(uq_engine.topological_decay_lambda, best_lambda)

        # Baseline density must have been updated to reflect the new decay rate
        self.assertNotEqual(uq_engine.N_baseline, initial_baseline)
        
        # Verify that _compute_baseline_density helper exists and is idempotent
        self.assertTrue(hasattr(uq_engine, "_compute_baseline_density"))
        current_baseline = uq_engine.N_baseline
        uq_engine._compute_baseline_density()
        self.assertAlmostEqual(uq_engine.N_baseline, current_baseline, places=6)


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

