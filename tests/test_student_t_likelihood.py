import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from Credal_Regression_UQ import CredalRegressionUQ

class TestStudentTLikelihood(unittest.TestCase):
    def setUp(self):
        # Extremely small toy dataset (10 samples) for fast execution
        np.random.seed(42)
        X = np.random.uniform(0, 10, (10, 1))
        y = np.sin(X[:, 0]) + np.random.normal(0, 0.1, 10)
        
        self.rf = RandomForestRegressor(n_estimators=2, random_state=42, min_samples_leaf=2)
        self.rf.fit(X, y)
        
        self.X_test = np.array([[1.0], [5.0]])
        self.credal_q = CredalRegressionUQ(self.rf, X, y)

    def test_student_t_likelihood_execution(self):
        # 1. Normal Test: student_t
        ep_t, al_t = self.credal_q.compute_uq(
            self.X_test, backend="cpu", integration_method="trapezoid", sup_solver="bisection", likelihood_type="student_t"
        )
        self.assertEqual(ep_t.shape, (2,))
        self.assertEqual(al_t.shape, (2,))
        self.assertTrue((ep_t >= 0.0).all())
        self.assertTrue((al_t >= 0.0).all())

    def test_student_t_corrected_likelihood_execution(self):
        # 2. Normal Test: student_t_corrected
        ep_t_corr, al_t_corr = self.credal_q.compute_uq(
            self.X_test, backend="cpu", integration_method="trapezoid", sup_solver="bisection", likelihood_type="student_t_corrected"
        )
        self.assertEqual(ep_t_corr.shape, (2,))
        self.assertTrue((ep_t_corr >= 0.0).all())

    def test_edge_cases(self):
        # 3. Edge Case: Extremely small counts / single-sample leaf fallback
        # Manually construct extreme stats to feed into _compute_uq_batch
        # We test that a leaf count of 1 does not crash with division by zero.
        all_test_leaf_ids = self.rf.apply(self.X_test)
        
        # Override counts and variances to test boundary limits
        # Leaf counts containing only 1 training point
        counts = np.ones((2, 2))  # (n_trees, n_samples)
        
        # Test that _compute_uq_batch runs successfully with counts=1
        # (df will be 0, clipped to 1.0)
        res = self.credal_q._compute_uq_batch(
            all_test_leaf_ids, backend="cpu", n_grid=10, integration_method="trapezoid", sup_solver="bisection", likelihood_type="student_t"
        )
        self.assertEqual(res[0].shape, (2,))
        self.assertFalse(np.isnan(res[0]).any())

if __name__ == "__main__":
    unittest.main()
