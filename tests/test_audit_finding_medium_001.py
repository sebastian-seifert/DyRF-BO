import os
import sys
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Epistemic_Quantifier import EpistemicQuantifier, LeafCache
from Credal_Regression_UQ import CredalRegressionUQ

class TestFindingMedium001LeafVarianceCorrection(unittest.TestCase):
    """
    Test suite for Audit Finding [MEDIUM-001]:
    Leaf variance correction mixes bootstrap weights and unique counts.
    Verifies that leaf sample variances use the exact unbiased scaling factor
    V1^2 / (V1^2 - V2) for weighted bootstrap observations, and Kish effective sample
    sizes for evidence counts.
    """

    def test_audit_evidence_reproduction_and_correction(self):
        """
        Reproduce the audit probe from RESEARCH_REPOSITORY_AUDIT.txt:
        Targets [0, 1, 2, 4, 7, 12] with random_state=4 on a single no-split tree
        has multiplicities [0, 1, 0, 0, 1, 4], unique count 3, weighted count 6,
        and node impurity 17.22222222.
        
        The faulty unweighted Bessel correction returned 25.83333433.
        The fixed-weight unbiased reference is 34.44444444.
        """
        y = np.array([0, 1, 2, 4, 7, 12], dtype=np.float64)
        X = np.zeros((len(y), 1), dtype=np.float64)

        rf = RandomForestRegressor(n_estimators=1, max_depth=1, random_state=4, bootstrap=True)
        rf.fit(X, y)

        X_test = np.array([[0.0]])

        # 1. Test EpistemicQuantifier uncached
        eq = EpistemicQuantifier(rf, X, y)
        var_eq = eq._base_calc_per_tree_variance(X_test, min_var=0.0)
        np.testing.assert_allclose(var_eq[0, 0], 34.44444444, rtol=1e-5)

        # 2. Test CredalRegressionUQ leaf stats
        credal = CredalRegressionUQ(rf, X, y)
        means, variances, counts = credal._calc_leaf_stats(X_test, min_var=0.0)
        np.testing.assert_allclose(variances[0, 0], 34.44444444, rtol=1e-5)
        # Kish Neff = V1^2 / V2 = 36 / 18 = 2.0
        np.testing.assert_allclose(counts[0, 0], 2.0, rtol=1e-5)

        # 3. Test LeafCache with X_train provided
        cache = LeafCache(rf, X_test, X_train=X, min_var=0.0)
        np.testing.assert_allclose(cache.variances[0, 0], 34.44444444, rtol=1e-5)
        np.testing.assert_allclose(cache.counts[0, 0], 2.0, rtol=1e-5)

    def test_unweighted_bootstrap_false_matches_sample_variance(self):
        """
        When bootstrap=False, all in-bag weights w_i = 1.
        The scaling factor V1^2 / (V1^2 - V2) simplifies identically to N / (N - 1),
        matching the classical unbiased sample variance np.var(y, ddof=1).
        """
        y = np.array([0.5, 1.2, 3.8, 4.1, 7.9, 12.3, 15.0], dtype=np.float64)
        X = np.zeros((len(y), 1), dtype=np.float64)

        rf = RandomForestRegressor(n_estimators=1, max_depth=1, random_state=42, bootstrap=False)
        rf.fit(X, y)

        X_test = np.array([[0.0]])
        expected_var = float(np.var(y, ddof=1))

        eq = EpistemicQuantifier(rf, X, y)
        var_eq = eq._base_calc_per_tree_variance(X_test, min_var=0.0)
        np.testing.assert_allclose(var_eq[0, 0], expected_var, rtol=1e-6)

        credal = CredalRegressionUQ(rf, X, y)
        _, variances, counts = credal._calc_leaf_stats(X_test, min_var=0.0)
        np.testing.assert_allclose(variances[0, 0], expected_var, rtol=1e-6)
        np.testing.assert_allclose(counts[0, 0], len(y), rtol=1e-6)

        cache = LeafCache(rf, X_test, X_train=X, min_var=0.0)
        np.testing.assert_allclose(cache.variances[0, 0], expected_var, rtol=1e-6)
        np.testing.assert_allclose(cache.counts[0, 0], len(y), rtol=1e-6)

    def test_single_sample_leaf_degeneracy_zero_variance(self):
        """
        If a leaf contains samples from only 1 distinct observation (or 1 sample total),
        unbiased sample variance is undefined / 0.0 (plus min_var).
        """
        y = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        X = np.zeros((len(y), 1), dtype=np.float64)

        rf = RandomForestRegressor(n_estimators=1, max_depth=1, random_state=42, bootstrap=True)
        rf.fit(X, y)

        X_test = np.array([[0.0]])
        min_var = 1e-6

        eq = EpistemicQuantifier(rf, X, y)
        var_eq = eq._base_calc_per_tree_variance(X_test, min_var=min_var)
        np.testing.assert_allclose(var_eq[0, 0], min_var, atol=1e-10)

        credal = CredalRegressionUQ(rf, X, y)
        _, variances, _ = credal._calc_leaf_stats(X_test, min_var=min_var)
        np.testing.assert_allclose(variances[0, 0], min_var, atol=1e-10)

    def test_cached_vs_uncached_parity_multi_tree(self):
        """
        Verify complete numerical equivalence between cached and uncached paths
        for multi-tree, multi-depth forests across EpistemicQuantifier and CredalRegressionUQ.
        """
        np.random.seed(123)
        X_train = np.random.uniform(-3, 3, (80, 3))
        y_train = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1]**2 + np.random.normal(0, 0.2, 80)

        rf = RandomForestRegressor(n_estimators=5, max_depth=4, random_state=99, bootstrap=True)
        rf.fit(X_train, y_train)

        X_test = np.random.uniform(-3, 3, (25, 3))

        cache = LeafCache(rf, X_test, X_train=X_train)
        eq_uncached = EpistemicQuantifier(rf, X_train, y_train)
        eq_cached = EpistemicQuantifier(rf, X_train, y_train, leaf_cache=cache)

        var_uncached = eq_uncached._base_calc_per_tree_variance(X_test)
        var_cached = eq_cached._base_calc_per_tree_variance(X_test)
        np.testing.assert_allclose(var_cached, var_uncached, rtol=1e-7)

        credal_uncached = CredalRegressionUQ(rf, X_train, y_train)
        credal_cached = CredalRegressionUQ(rf, X_train, y_train, leaf_cache=cache)

        _, vars_uncached, counts_uncached = credal_uncached._calc_leaf_stats(X_test)
        _, vars_cached, counts_cached = credal_cached._calc_leaf_stats(X_test)
        np.testing.assert_allclose(vars_cached, vars_uncached, rtol=1e-7)
        np.testing.assert_allclose(counts_cached, counts_uncached, rtol=1e-7)

if __name__ == "__main__":
    unittest.main()
