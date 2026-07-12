import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from Epistemic_Quantifier import EpistemicQuantifier

class TestShakerQuadrature(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        X = np.random.uniform(0, 10, (50, 2))
        y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.1, 50)
        
        self.rf = RandomForestRegressor(n_estimators=10, min_samples_leaf=3, random_state=42)
        self.rf.fit(X, y)
        
        self.X_test = np.random.uniform(0, 10, (15, 2))
        self.quantifier = EpistemicQuantifier(self.rf, X, y)

    def test_quadrature_execution_cpu(self):
        # Calculate epistemic variance using the deterministic quadrature (CPU)
        u_e = self.quantifier.shaker_get_epistemic_variance(self.X_test, backend="cpu")
        self.assertEqual(u_e.shape, (15,))
        self.assertFalse(np.isnan(u_e).any())
        self.assertTrue((u_e >= 0.0).all())

    def test_entropy_conversion(self):
        # Check standard properties of differential entropy conversion:
        # H = 0.5 * log2(2 * pi * e * var)
        # Var = 2^(2H) / (2 * pi * e)
        var_in = 2.5
        entropy = self.quantifier._shaker_convert_var_to_entropy(var_in)
        var_out = self.quantifier._shaker_convert_entropy_to_var(entropy)
        self.assertAlmostEqual(var_in, var_out, places=5)

if __name__ == "__main__":
    unittest.main()
