import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from Epistemic_Quantifier import EpistemicQuantifier, LeafCache
from Credal_Regression_UQ import CredalRegressionUQ

class TestLeafCache(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        X = np.random.uniform(0, 10, (100, 2))
        y = np.sin(X[:, 0]) + np.random.normal(0, 0.1, 100)
        
        self.rf = RandomForestRegressor(n_estimators=5, random_state=42, min_samples_leaf=2)
        self.rf.fit(X, y)
        
        self.X_test = np.random.uniform(0, 10, (30, 2))
        self.X_train = X
        self.y_train = y

    def test_leaf_cache_parity(self):
        # 1. Compute without cache
        quantifier_no_cache = EpistemicQuantifier(self.rf, self.X_train, self.y_train)
        credal_no_cache = CredalRegressionUQ(self.rf, self.X_train, self.y_train)
        
        u_std_no = quantifier_no_cache.standard_get_epistemic_variance(self.X_test)
        u_chen_no = quantifier_no_cache.chen_get_epistemic_variance(self.X_test)
        u_al_no = quantifier_no_cache.base_get_aleatoric_variance(self.X_test)
        u_credal_e_no, u_credal_a_no = credal_no_cache.compute_uq(self.X_test, likelihood_type="normal")
        
        # 2. Compute with cache
        cache = LeafCache(self.rf, self.X_test)
        quantifier_with_cache = EpistemicQuantifier(self.rf, self.X_train, self.y_train, leaf_cache=cache)
        credal_with_cache = CredalRegressionUQ(self.rf, self.X_train, self.y_train, leaf_cache=cache)
        
        u_std_yes = quantifier_with_cache.standard_get_epistemic_variance(self.X_test)
        u_chen_yes = quantifier_with_cache.chen_get_epistemic_variance(self.X_test)
        u_al_yes = quantifier_with_cache.base_get_aleatoric_variance(self.X_test)
        u_credal_e_yes, u_credal_a_yes = credal_with_cache.compute_uq(self.X_test, likelihood_type="normal")
        
        # 3. Check exact equality
        np.testing.assert_allclose(u_std_no, u_std_yes, rtol=1e-6)
        np.testing.assert_allclose(u_chen_no, u_chen_yes, rtol=1e-6)
        np.testing.assert_allclose(u_al_no, u_al_yes, rtol=1e-6)
        np.testing.assert_allclose(u_credal_e_no, u_credal_e_yes, rtol=1e-6)
        np.testing.assert_allclose(u_credal_a_no, u_credal_a_yes, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()
