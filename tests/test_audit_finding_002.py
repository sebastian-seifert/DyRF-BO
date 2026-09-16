import os
import sys
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Epistemic_Quantifier import EpistemicQuantifier
from ep_extractors.shaker_entropy import ShakerEntropyExtractor


class TestFinding002ShakerQuadrature(unittest.TestCase):
    """Tests for Audit Finding 2: Shaker numerical integration correctness and bounds."""

    def test_audit_evidence_reproduction_fitted_forest(self):
        """
        Audit evidence case:
        Fitted 100-tree forest on X=0,...,39 with targets -1 for X<20 and +1 otherwise,
        max_depth=1, min_samples_leaf=1, bootstrap=True, random_state=3.
        At X=19.5, trees predict either -1 or +1 with near-zero leaf variance (clamped to 1e-6).
        Old implementation returned MI = 7.9187 bits (violating log2(100) = 6.6439 bits).
        True theoretical MI is H(0.85, 0.15) ≈ 0.6098 bits.
        """
        X_train = np.arange(40).reshape(-1, 1).astype(float)
        y_train = np.where(X_train[:, 0] < 20, -1.0, 1.0)

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=1,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=3
        )
        rf.fit(X_train, y_train)

        eq = EpistemicQuantifier(rf, X_train, y_train)
        X_test = np.array([[19.5]])

        # 1. Epistemic entropy (Mutual Information in bits)
        mi_bits = eq.shaker_get_epistemic_entropy(X_test)[0]
        max_mi_bound = np.log2(len(rf.estimators_))

        # MI must never exceed the theoretical upper bound log2(M)
        self.assertLessEqual(
            mi_bits,
            max_mi_bound + 1e-6,
            f"MI {mi_bits} exceeded mathematical upper bound {max_mi_bound} bits!"
        )

        # MI must be close to the reference two-mode entropy H(0.85, 0.15) ≈ 0.6098 bits
        # Old code returned 7.9187 bits
        self.assertAlmostEqual(mi_bits, 0.60984, places=2)

        # 2. Epistemic variance
        ep_var = eq.shaker_get_epistemic_variance(X_test)[0]
        self.assertGreaterEqual(ep_var, 0.0)
        self.assertFalse(np.isnan(ep_var))
        self.assertFalse(np.isinf(ep_var))

    def test_identical_components_yield_zero_mi(self):
        """When all ensemble components are identical, epistemic uncertainty (MI) must be zero."""
        # 20 identical trees predicting constant 0 with identical data
        X_train = np.linspace(-1, 1, 20).reshape(-1, 1)
        y_train = np.zeros(20)

        rf = RandomForestRegressor(n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)

        eq = EpistemicQuantifier(rf, X_train, y_train)
        X_test = np.array([[0.0]])

        mi_bits = eq.shaker_get_epistemic_entropy(X_test)[0]
        self.assertAlmostEqual(mi_bits, 0.0, places=4)

        ep_var = eq.shaker_get_epistemic_variance(X_test)[0]
        self.assertAlmostEqual(ep_var, 0.0, places=4)

    def test_fully_separated_components_achieve_max_mi_bound(self):
        """
        When M components are fully separated with negligible overlap,
        MI must equal log2(M) within numerical precision and never exceed it.
        """
        # Mocking leaf stats with 10 well-separated means [0, 50, 100, ..., 450] and small sigmas
        M = 10
        means = (np.arange(M) * 50.0).reshape(M, 1) # shape (n_trees, n_samples)
        vars_ = np.full((M, 1), 0.01) # sigma = 0.1

        class MockRF:
            estimators_ = [None] * M
            def apply(self, X):
                return np.zeros((X.shape[0], M), dtype=int)

        class MockLeafCache:
            def __init__(self, m, v):
                self.means = m
                self.variances = v
                self.all_test_leaf_ids = np.zeros((1, m.shape[0]), dtype=int)

        rf = MockRF()
        eq = EpistemicQuantifier(rf, np.zeros((10, 1)), np.zeros(10), leaf_cache=MockLeafCache(means, vars_))
        X_test = np.array([[0.0]])

        mi_bits = eq.shaker_get_epistemic_entropy(X_test)[0]
        expected_bound = np.log2(M) # 3.3219 bits

        self.assertLessEqual(mi_bits, expected_bound + 1e-5)
        self.assertAlmostEqual(mi_bits, expected_bound, places=2)

    def test_unequal_width_components_bounded(self):
        """Unequal component widths with various spreads must satisfy 0 <= MI <= log2(M)."""
        M = 5
        means = np.array([[0.0], [1.0], [5.0], [10.0], [20.0]])
        vars_ = np.array([[1e-6], [0.1], [1.0], [0.01], [4.0]])

        class MockRF:
            estimators_ = [None] * M
            def apply(self, X):
                return np.zeros((X.shape[0], M), dtype=int)

        class MockLeafCache:
            def __init__(self, m, v):
                self.means = m
                self.variances = v
                self.all_test_leaf_ids = np.zeros((1, m.shape[0]), dtype=int)

        rf = MockRF()
        eq = EpistemicQuantifier(rf, np.zeros((10, 1)), np.zeros(10), leaf_cache=MockLeafCache(means, vars_))
        X_test = np.array([[0.0]])

        mi_bits = eq.shaker_get_epistemic_entropy(X_test)[0]
        self.assertGreaterEqual(mi_bits, 0.0)
        self.assertLessEqual(mi_bits, np.log2(M) + 1e-6)

    def test_shaker_extractor_respects_accuracy_controls(self):
        """Verify that ShakerEntropyExtractor propagates quadrature parameters and extracts valid signals."""
        X_train = np.arange(20).reshape(-1, 1).astype(float)
        y_train = np.where(X_train[:, 0] < 10, -1.0, 1.0)

        rf = RandomForestRegressor(n_estimators=10, max_depth=1, random_state=42)
        rf.fit(X_train, y_train)

        extractor = ShakerEntropyExtractor(rf, n_quadrature_points=40)
        extractor.fit(X_train, y_train)

        signal = extractor.extract_epistemic_signal(np.array([[9.5], [10.5]]))
        self.assertEqual(len(signal), 2)
        self.assertTrue(np.all(np.isfinite(signal)))
        self.assertTrue(np.all(signal >= 0.0))


if __name__ == "__main__":
    unittest.main()
