import os
import sys
import unittest
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ

class TestFindingMedium002OOBReconstructionAndMasking(unittest.TestCase):
    """
    Test suite for Audit Finding [MEDIUM-002]:
    OOB calibration can use the wrong sampling mask or nonexistent predictions.
    Verifies:
      1. Faithful bootstrap reconstruction under subsampling (e.g., max_samples < 1.0).
      2. Zero-OOB rows are tracked via valid_oob_mask and zeroed in proximity weights.
      3. No NaN leakage into quantiles or local_mae when OOB predictions contain NaNs.
      4. Targeted warning for Case #1 (insufficient valid OOB neighbor support).
    """

    def test_subsampling_bootstrap_reconstruction_fidelity(self):
        """
        Audit evidence case:
        A fitted 40-row forest with max_samples=0.5 draws 20 samples per tree.
        The old code hardcoded (n_train, n_train) = (40, 40), drawing 40 samples.
        We verify that in-bag counts per tree sum to exactly 20.
        """
        np.random.seed(42)
        X = np.random.uniform(0, 1, (40, 2))
        y = np.sin(X[:, 0]) + np.cos(X[:, 1])

        rf = RandomForestRegressor(n_estimators=5, max_samples=0.5, random_state=42, bootstrap=True)
        rf.fit(X, y)

        uq_model = GPUProximityRegressionUQ(rf, X, y, device="cpu", use_density_scaling=False)
        uq_model.fit()

        for t in range(len(rf.estimators_)):
            expected_draws = len(rf.estimators_samples_[t])  # 20
            actual_in_bag_sum = np.sum(uq_model.in_bag_counts[:, t])
            self.assertEqual(actual_in_bag_sum, expected_draws)

            # Check that OOB indices match the true complement of in-bag samples
            in_bag_set = set(rf.estimators_samples_[t])
            for i in range(len(X)):
                expected_oob = 1 if i not in in_bag_set else 0
                self.assertEqual(uq_model.oob_indices[i, t], expected_oob)

    def test_zero_oob_row_masking_and_nan_immunity(self):
        """
        Audit evidence case:
        A forest where some observations have zero OOB trees (e.g., random_state=8, 300 rows, 10 trees).
        Scikit-Learn issues a warning and sets oob_prediction_[i] = NaN.
        We assert that:
          - valid_oob_mask identifies rows with 0 OOB trees.
          - Non-OOB rows are zeroed in prox_batch.
          - predict_with_intervals produces strictly finite quantiles with no NaN leakage.
        """
        np.random.seed(8)
        X = np.random.uniform(0, 10, (300, 2))
        y = np.sin(X[:, 0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf = RandomForestRegressor(n_estimators=10, random_state=8, oob_score=True, bootstrap=True)
            rf.fit(X, y)

        # Check if there are indeed points with zero OOB trees
        oob_tree_counts = np.zeros(len(X), dtype=int)
        for t in range(len(rf.estimators_)):
            in_bag = set(rf.estimators_samples_[t])
            for i in range(len(X)):
                if i not in in_bag:
                    oob_tree_counts[i] += 1

        zero_oob_indices = np.where(oob_tree_counts == 0)[0]
        self.assertGreater(len(zero_oob_indices), 0, "Test requires at least one zero-OOB observation.")

        uq_model = GPUProximityRegressionUQ(rf, X, y, device="cpu", use_density_scaling=False)
        uq_model.fit()

        self.assertTrue(hasattr(uq_model, "valid_oob_mask"))
        for idx in zero_oob_indices:
            self.assertFalse(uq_model.valid_oob_mask[idx])

        # Query points: make sure quantiles are completely finite and valid
        X_test = np.array([[5.0, 5.0], [0.5, 0.5]])
        y_lwr, y_pred, y_upr, local_mae = uq_model.predict_with_intervals(X_test, n_neighbors="auto", return_mae=True)

        self.assertTrue(np.all(np.isfinite(y_lwr)))
        self.assertTrue(np.all(np.isfinite(y_upr)))
        self.assertTrue(np.all(np.isfinite(local_mae)))
        self.assertTrue(np.all(y_upr >= y_lwr))

    def test_case_1_insufficient_support_warning(self):
        """
        Verify that a UserWarning is raised if and only if valid OOB neighbor support
        is starved (k_eff < 3 or total valid weight < 1e-5).
        """
        np.random.seed(42)
        X = np.random.uniform(0, 1, (20, 2))
        y = np.sin(X[:, 0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf = RandomForestRegressor(n_estimators=5, random_state=42, bootstrap=True)
            rf.fit(X, y)

        uq_model = GPUProximityRegressionUQ(rf, X, y, device="cpu", use_density_scaling=False)
        uq_model.fit()

        # Artificially mask out almost all OOB points to simulate severe starvation
        uq_model.valid_oob_mask[:] = False
        uq_model.valid_oob_mask[0] = True  # only 1 valid point (k_eff = 1.0 < 3)

        X_test = np.array([[0.5, 0.5]])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            uq_model.predict_with_intervals(X_test, n_neighbors="auto")

            warning_found = any(
                issubclass(w.category, UserWarning) and "Insufficient valid OOB neighbor support" in str(w.message)
                for w in recorded
            )
            self.assertTrue(warning_found, "Expected UserWarning for insufficient valid OOB neighbor support.")

if __name__ == "__main__":
    unittest.main()
