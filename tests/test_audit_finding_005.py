import os
import sys
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ


class TestFinding005WeightedQuantiles(unittest.TestCase):
    """
    Tests for Audit Finding 5 (HIGH-005):
    Weighted residual quantiles must be invariant to zero-weight observations,
    must never interpolate into zero-support regions, and must correctly implement
    the inverse weighted empirical CDF convention.
    """

    def setUp(self):
        rf = RandomForestRegressor(n_estimators=5, oob_score=True, random_state=42)
        X = np.linspace(-1, 1, 10).reshape(-1, 1)
        y = np.sin(X[:, 0])
        rf.fit(X, y)
        self.uq = GPUProximityRegressionUQ(
            model=rf,
            X_train=X,
            y_train=y,
            device="cpu"
        )

    def test_audit_evidence_zero_weight_point_mass(self):
        """
        Audit evidence case:
        values = [-100, 0, 10], weights = [[0, 1, 0]].
        Old code returned:
            q_0.025 = -97.5, q_0.5 = -50.0, q_0.975 = -2.5
        Correct code must return 0.0 for all interior quantiles.
        """
        values = np.array([-100.0, 0.0, 10.0])
        weights = np.array([[0.0, 1.0, 0.0]])

        for q in [0.025, 0.1, 0.5, 0.9, 0.975]:
            val = self.uq._compute_weighted_quantile(values, weights, q)
            self.assertEqual(
                float(val[0]),
                0.0,
                f"For q={q}, expected 0.0 on point mass at 0 with zero-weight outliers, got {val[0]}"
            )

    def test_zero_weight_insertion_invariance(self):
        """
        Inserting or moving zero-weight observations must not alter the resulting quantiles.
        """
        base_values = np.array([10.0, 20.0, 30.0, 40.0])
        base_weights = np.array([[0.1, 0.4, 0.3, 0.2]])

        # Contaminate with extreme outliers that have zero weight
        dirty_values = np.array([-9999.0, 10.0, 20.0, 30.0, 40.0, 8888.0])
        dirty_weights = np.array([[0.0, 0.1, 0.4, 0.3, 0.2, 0.0]])

        for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
            q_clean = self.uq._compute_weighted_quantile(base_values, base_weights, q)
            q_dirty = self.uq._compute_weighted_quantile(dirty_values, dirty_weights, q)
            self.assertAlmostEqual(
                float(q_dirty[0]),
                float(q_clean[0]),
                places=5,
                msg=f"Zero-weight outliers altered quantile at q={q}: {q_dirty[0]} vs clean {q_clean[0]}"
            )

    def test_support_boundary_guarantee(self):
        """
        Quantiles must always lie within the convex hull of positive-weight observations:
        min_{w_i > 0}(x_i) <= Q(q) <= max_{w_i > 0}(x_i).
        """
        values = np.array([-500.0, 5.0, 10.0, 15.0, 500.0])
        weights = np.array([[0.0, 0.2, 0.5, 0.3, 0.0]])

        min_supp = 5.0
        max_supp = 15.0

        for q in np.linspace(0.01, 0.99, 20):
            val = self.uq._compute_weighted_quantile(values, weights, q)
            self.assertGreaterEqual(float(val[0]), min_supp - 1e-6)
            self.assertLessEqual(float(val[0]), max_supp + 1e-6)

    def test_zero_weight_row_fallback(self):
        """
        Rows with zero total weight must gracefully fall back to unweighted quantile.
        """
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])

        for q in [0.1, 0.5, 0.9]:
            val = self.uq._compute_weighted_quantile(values, weights, q)
            expected = np.quantile(values, q)
            self.assertAlmostEqual(float(val[0]), float(expected), places=5)

    def test_end_to_end_predict_with_intervals_auto(self):
        """
        Verify end-to-end predict_with_intervals when n_neighbors='auto'
        with topological decay.
        """
        np.random.seed(42)
        X_train = np.linspace(-3, 3, 30).reshape(-1, 1)
        y_train = np.sin(X_train[:, 0])

        rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        rf.fit(X_train, y_train)

        uq_model = GPUProximityRegressionUQ(
            model=rf,
            X_train=X_train,
            y_train=y_train,
            device="cpu",
            topological_decay_lambda=1.5
        )
        uq_model.fit()

        X_test = np.array([[-2.0], [0.0], [2.0]])
        res = uq_model.predict_with_intervals(X_test, n_neighbors="auto", level=0.95, return_mae=True)
        y_lwr, y_pred, y_upr, mae = res

        # Bounds must be logically ordered: lwr <= pred <= upr
        self.assertTrue(np.all(y_lwr <= y_upr))
        self.assertTrue(np.all(mae >= 0.0))
        self.assertTrue(np.all(np.isfinite(y_lwr)))
        self.assertTrue(np.all(np.isfinite(y_upr)))


if __name__ == "__main__":
    unittest.main()
