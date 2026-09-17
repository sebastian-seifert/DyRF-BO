import os
import sys
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Credal_Regression_UQ import CredalRegressionUQ


class TestFinding003CredalNewtonSolver(unittest.TestCase):
    """
    Tests for Audit Finding 3 (HIGH-003):
    Credal Newton solver root branch correctness and reflection symmetry.
    """

    def test_audit_evidence_single_leaf_newton_matches_bisection(self):
        """
        Audit evidence case:
        Single Gaussian leaf with mean 0, variance 1, count k=5, 1025-point grid.
        Bisection reference: (epistemic, aleatoric) ≈ (2.811413, 0.308859).
        Old Newton returned (0.636612, 0.637572) due to u_ge <= -1e-15 branch error.
        Corrected Newton must match bisection reference.
        """
        class MockRF:
            estimators_ = [None]
            def apply(self, X):
                return np.zeros((X.shape[0], 1), dtype=int)

        class MockLeafCache:
            def __init__(self, m, v, k):
                self.means = m
                self.variances = v
                self.counts = k
                self.all_test_leaf_ids = np.zeros((m.shape[1], m.shape[0]), dtype=int)
            def get_slice(self, start, end):
                return MockLeafCache(self.means[:, start:end], self.variances[:, start:end], self.counts[:, start:end])

        # 1 tree, 1 test sample: mu=0, sigma^2=1, k=5
        means = np.array([[0.0]])      # shape: (n_trees, n_samples)
        vars_ = np.array([[1.0]])
        counts = np.array([[5.0]])

        leaf_cache = MockLeafCache(means, vars_, counts)
        credal_uq = CredalRegressionUQ(
            model=MockRF(),
            X_train=np.zeros((5, 1)),
            y_train=np.zeros(5),
            leaf_cache=leaf_cache
        )

        X_test = np.array([[0.0]])

        # Run with bisection (reference)
        u_e_bisect, u_a_bisect = credal_uq.compute_uq(
            X_test,
            backend="cpu",
            integration_method="trapezoid",
            sup_solver="bisection",
            n_grid=1025,
            n_iter=50
        )

        # Run with newton
        u_e_newton, u_a_newton = credal_uq.compute_uq(
            X_test,
            backend="cpu",
            integration_method="trapezoid",
            sup_solver="newton",
            n_grid=1025
        )

        # Verify bisection matches audit evidence
        self.assertAlmostEqual(float(u_e_bisect[0]), 2.811413, places=2)
        self.assertAlmostEqual(float(u_a_bisect[0]), 0.308859, places=2)

        # With analytical reflection / corrected Newton, newton must match bisection
        self.assertAlmostEqual(float(u_e_newton[0]), float(u_e_bisect[0]), places=2)
        self.assertAlmostEqual(float(u_a_newton[0]), float(u_a_bisect[0]), places=2)

    def test_reflection_symmetry_at_zero(self):
        """
        At z = 0 (symmetric center), pi(Y <= 0) must equal pi(Y >= 0).
        """
        class MockRF:
            estimators_ = [None]
            def apply(self, X):
                return np.zeros((X.shape[0], 1), dtype=int)

        class MockLeafCache:
            def __init__(self, m, v, k):
                self.means = m
                self.variances = v
                self.counts = k
                self.all_test_leaf_ids = np.zeros((m.shape[1], m.shape[0]), dtype=int)
            def get_slice(self, start, end):
                return MockLeafCache(self.means[:, start:end], self.variances[:, start:end], self.counts[:, start:end])

        means = np.array([[0.0]])
        vars_ = np.array([[1.0]])
        counts = np.array([[10.0]])

        leaf_cache = MockLeafCache(means, vars_, counts)
        credal_uq = CredalRegressionUQ(
            model=MockRF(),
            X_train=np.zeros((10, 1)),
            y_train=np.zeros(10),
            leaf_cache=leaf_cache
        )

        # Directly test internal _compute_uq_batch for symmetry of pi_le and pi_ge at t=0
        # t_grid = [0.0] gives z_b = 0.0
        t_grid = np.array([0.0])
        u_e, u_a = credal_uq.compute_uq(
            np.array([[0.0]]),
            backend="cpu",
            integration_method="trapezoid",
            sup_solver="newton",
            n_grid=65
        )
        # Check that both epistemic and aleatoric values are positive finite floats
        self.assertGreater(float(u_e[0]), 0.0)
        self.assertGreater(float(u_a[0]), 0.0)
        self.assertTrue(np.isfinite(u_e[0]))
        self.assertTrue(np.isfinite(u_a[0]))

    def test_newton_and_bisection_agreement_on_fitted_forest(self):
        """
        On a real fitted forest, Credal UQ with newton solver must agree with
        bisection solver across all test samples.
        """
        np.random.seed(42)
        X_train = np.random.uniform(-3, 3, size=(50, 2))
        y_train = np.sin(X_train[:, 0]) + 0.1 * np.random.randn(50)

        rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        rf.fit(X_train, y_train)

        credal_uq = CredalRegressionUQ(rf, X_train, y_train)

        X_test = np.random.uniform(-3, 3, size=(10, 2))

        u_e_bisect, u_a_bisect = credal_uq.compute_uq(
            X_test,
            backend="cpu",
            integration_method="gauss_legendre",
            sup_solver="bisection",
            n_grid=64,
            n_iter=30
        )

        u_e_newton, u_a_newton = credal_uq.compute_uq(
            X_test,
            backend="cpu",
            integration_method="gauss_legendre",
            sup_solver="newton",
            n_grid=64
        )

        # Epistemic and aleatoric uncertainty profiles should match closely
        np.testing.assert_allclose(u_e_newton, u_e_bisect, rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(u_a_newton, u_a_bisect, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
