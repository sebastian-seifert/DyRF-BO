from __future__ import annotations
import unittest
import numpy as np
from scipy.stats import norm
from ConfigSpace import ConfigurationSpace, Configuration, Float

from smac.acquisition.function import AbstractAcquisitionFunction
from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
from carps_integration.proximity_lcb import ProximityLowerBoundAcquisition

class MockProximitySurrogate:
    """Mock surrogate mimicking CustomUncertaintyRandomForest for unit testing."""
    def __init__(self, y_pred: np.ndarray, y_lwr: np.ndarray, y_upr: np.ndarray, local_mae: np.ndarray, n_train: int = 50):
        self._y_pred = np.asarray(y_pred, dtype=np.float64)
        self._y_lwr = np.asarray(y_lwr, dtype=np.float64)
        self._y_upr = np.asarray(y_upr, dtype=np.float64)
        self._local_mae = np.asarray(local_mae, dtype=np.float64)
        self.n_train = n_train
        self.last_X = np.zeros((n_train, 2))

    def predict_with_intervals(
        self,
        X: np.ndarray,
        n_neighbors: int | str = "auto",
        level: float = 0.95,
        return_mae: bool = True
    ):
        n = len(X)
        y_pred = self._y_pred[:n]
        y_lwr = self._y_lwr[:n]
        y_upr = self._y_upr[:n]
        local_mae = self._local_mae[:n]
        if return_mae:
            return y_lwr, y_pred, y_upr, local_mae
        return y_lwr, y_pred, y_upr

    def predict_standard_rf(self, X: np.ndarray):
        n = len(X)
        mean_rf = self._y_pred[:n]
        var_rf = np.full((n, 1), 0.25)  # std = 0.5
        return mean_rf.reshape(-1, 1), var_rf

class TestProximityLowerBoundAcquisition(unittest.TestCase):
    def setUp(self):
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameter(Float("x0", (-5.0, 5.0), default=0.0))
        self.cs.add_hyperparameter(Float("x1", (-5.0, 5.0), default=0.0))

    def test_smac_contract_and_shapes(self):
        """Verify compliance with SMAC3 AbstractAcquisitionFunction contract."""
        acq = ProximityLowerBoundAcquisition(eps=0.10, level=0.95, k=10)
        self.assertIsInstance(acq, AbstractAcquisitionFunction)
        self.assertIn("Proximity Lower Bound", acq.name)
        self.assertEqual(acq.meta["eps"], 0.10)
        self.assertEqual(acq.meta["level"], 0.95)
        self.assertEqual(acq.meta["k"], 10)

        # Attach mock surrogate
        mock_surrogate = MockProximitySurrogate(
            y_pred=[10.0, 12.0],
            y_lwr=[8.0, 9.0],
            y_upr=[12.0, 15.0],
            local_mae=[1.5, 2.0]
        )
        acq.update(mock_surrogate)

        # Test 2D input array (N, D)
        X_2d = np.array([[0.0, 1.0], [1.0, -1.0]])
        scores = acq._compute(X_2d)
        self.assertEqual(scores.shape, (2, 1))
        self.assertTrue(np.all(np.isfinite(scores)))

        # Test 1D input array (D,)
        X_1d = np.array([0.0, 1.0])
        scores_1d = acq._compute(X_1d)
        self.assertEqual(scores_1d.shape, (1, 1))

        # Test ConfigSpace Configuration input via __call__
        configs = [Configuration(self.cs, values={"x0": 0.0, "x1": 1.0})]
        call_scores = acq(configs)
        self.assertEqual(call_scores.shape, (1, 1))

    def test_maximization_sign_convention(self):
        """
        Verify that lower predicted bound gives strictly higher SMAC acquisition score.
        For minimization, lower is better, so argmax acq(X) must select the minimum lower bound.
        """
        mock_surrogate = MockProximitySurrogate(
            y_pred=[10.0, 10.0],
            y_lwr=[5.0, 8.0],   # Candidate A has much lower bound than B
            y_upr=[15.0, 12.0],
            local_mae=[2.0, 1.0]
        )
        acq = ProximityLowerBoundAcquisition(eps=0.10, level=0.95)
        acq.update(mock_surrogate)

        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        scores = acq._compute(X).flatten()

        # Candidate A (lwr=5.0) must score strictly higher than Candidate B (lwr=8.0)
        self.assertGreater(scores[0], scores[1])

    def test_floor_enforcement_on_pathological_positive_residuals(self):
        """
        When proximate residuals are all positive, y_lwr >= y_pred (anti-exploration).
        The floor must pull y_lwr down to (y_pred - delta_floor).
        """
        level = 0.95
        eps = 0.10
        kappa = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
        local_mae = 2.0
        expected_floor = eps * kappa * local_mae

        y_pred = 10.0
        # Inverted / pathological: lower quantile is +1.0 above mean!
        y_lwr_pathological = 11.0

        mock_surrogate = MockProximitySurrogate(
            y_pred=[y_pred],
            y_lwr=[y_lwr_pathological],
            y_upr=[13.0],
            local_mae=[local_mae]
        )
        acq = ProximityLowerBoundAcquisition(eps=eps, level=level)
        acq.update(mock_surrogate)

        X = np.array([[0.0, 0.0]])
        score = acq._compute(X).item()

        # Floored lower bound must be y_pred - expected_floor
        expected_y_lwr_floored = y_pred - expected_floor
        expected_score = -expected_y_lwr_floored

        self.assertAlmostEqual(score, expected_score, places=6)
        self.assertAlmostEqual(score, -y_pred + expected_floor, places=6)

    def test_floor_invariance_under_healthy_spread(self):
        """
        When empirical exploration (y_pred - y_lwr) > delta_floor,
        the floor must NOT modify the raw statistical lower bound.
        """
        level = 0.95
        eps = 0.10
        local_mae = 1.0
        # Natural exploration bonus is 10.0 - 5.0 = 5.0
        # Expected floor is 0.10 * 1.96 * 1.0 = 0.196 << 5.0
        y_pred = 10.0
        y_lwr = 5.0

        mock_surrogate = MockProximitySurrogate(
            y_pred=[y_pred],
            y_lwr=[y_lwr],
            y_upr=[15.0],
            local_mae=[local_mae]
        )
        acq = ProximityLowerBoundAcquisition(eps=eps, level=level)
        acq.update(mock_surrogate)

        X = np.array([[0.0, 0.0]])
        score = acq._compute(X).item()

        # Score must strictly equal -y_lwr without any floor modification
        self.assertAlmostEqual(score, -y_lwr, places=6)

    def test_eps_zero_ablation(self):
        """When eps=0.0, floor is disabled; score strictly equals -y_lwr even if inverted."""
        mock_surrogate = MockProximitySurrogate(
            y_pred=[10.0],
            y_lwr=[10.5],  # Inverted
            y_upr=[12.0],
            local_mae=[2.0]
        )
        acq = ProximityLowerBoundAcquisition(eps=0.0, level=0.95)
        acq.update(mock_surrogate)

        X = np.array([[0.0, 0.0]])
        score = acq._compute(X).item()
        self.assertAlmostEqual(score, -10.5, places=6)

    def test_alpha_boundary_pure_exploitation(self):
        """
        When level -> 0.0 (alpha -> 1.0, median evaluation),
        kappa -> 0, so delta_floor -> 0.
        """
        acq = ProximityLowerBoundAcquisition(eps=0.10, level=0.0)
        self.assertAlmostEqual(acq._kappa, 0.0, places=6)

        mock_surrogate = MockProximitySurrogate(
            y_pred=[10.0],
            y_lwr=[10.0],
            y_upr=[10.0],
            local_mae=[2.0]
        )
        acq.update(mock_surrogate)
        score = acq._compute(np.array([[0.0, 0.0]])).item()
        self.assertAlmostEqual(score, -10.0, places=6)

    def test_end_to_end_custom_rf_surrogate_integration(self):
        """
        Full end-to-end contract test with CustomUncertaintyRandomForest and StandardProximity.
        """
        np.random.seed(42)
        X_train = np.random.uniform(-3.0, 3.0, size=(20, 2))
        y_train = np.sin(X_train[:, 0]) + 0.1 * np.random.normal(size=20)

        surrogate = CustomUncertaintyRandomForest(
            uncertainty_func="standard_proximity",
            configspace=self.cs,
            n_trees=10,
            oob_score=True
        )
        surrogate.train(X_train, y_train)

        # Surrogate must provide predict_with_intervals
        self.assertTrue(hasattr(surrogate, "predict_with_intervals"))

        acq = ProximityLowerBoundAcquisition(eps=0.10, level=0.95, k=5)
        acq.update(surrogate)

        X_test = np.random.uniform(-3.0, 3.0, size=(5, 2))
        scores = acq._compute(X_test)

        self.assertEqual(scores.shape, (5, 1))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_local_mae_calculation_exact_match(self):
        """Verify that local_mae matches the mean of absolute proximate residuals in [r_lwr, r_upr]."""
        from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X_train = np.random.uniform(-3.0, 3.0, size=(25, 2))
        y_train = np.sin(X_train[:, 0])

        base_rf = RandomForestRegressor(n_estimators=15, oob_score=True, random_state=42)
        base_rf.fit(X_train, y_train)

        uq = GPUProximityRegressionUQ(base_rf, X_train, y_train, device="cpu")
        uq.fit()

        X_test = np.array([[0.0, 0.0]])
        y_lwr, y_pred, y_upr, local_mae = uq.predict_with_intervals(X_test, n_neighbors=5, level=0.95, return_mae=True)

        self.assertEqual(len(local_mae), 1)
        self.assertGreater(local_mae[0], 0.0)
        self.assertTrue(np.isfinite(local_mae[0]))

    def test_warmup_fallback_trigger_n_le_k(self):
        """When N <= k, verify acquisition routes to standard Random Forest LCB."""
        # Setup mock surrogate where y_pred=10.0, var_rf=0.25 (std=0.5)
        mock_surrogate = MockProximitySurrogate(
            y_pred=[10.0],
            y_lwr=[5.0],   # Proximity would give 5.0
            y_upr=[15.0],
            local_mae=[1.0]
        )
        mock_surrogate.n_train = 5  # N = 5 <= k = 10

        acq = ProximityLowerBoundAcquisition(eps=0.10, level=0.95, k=10)
        acq.update(mock_surrogate, num_data=5)

        X = np.array([[0.0, 0.0]])
        score = acq._compute(X).item()

        # Standard RF LCB for minimization:
        # LCB = mean - kappa * std = 10.0 - 1.95996 * 0.5 = 9.02002
        # Acq = -LCB = -9.02002
        kappa = float(norm.ppf(1.0 - (1.0 - 0.95) / 2.0))
        expected_score = -(10.0 - kappa * 0.5)
        self.assertAlmostEqual(score, expected_score, places=5)

    def test_proximity_activation_n_gt_k(self):
        """When N > k, verify acquisition switches to floored Proximity Lower Bound."""
        mock_surrogate = MockProximitySurrogate(
            y_pred=[10.0],
            y_lwr=[5.0],   # Proximity gives 5.0
            y_upr=[15.0],
            local_mae=[1.0]
        )
        mock_surrogate.n_train = 15  # N = 15 > k = 10

        acq = ProximityLowerBoundAcquisition(eps=0.10, level=0.95, k=10)
        acq.update(mock_surrogate, num_data=15)

        X = np.array([[0.0, 0.0]])
        score = acq._compute(X).item()

        # Proximity score: -y_lwr = -5.0
        self.assertAlmostEqual(score, -5.0, places=5)

    def test_pure_rf_variance_used_in_fallback(self):
        """
        Verify that during warm start (N <= k), the fallback strictly uses
        native Random Forest tree variance from super()._predict (predict_standard_rf),
        completely bypassing custom epistemic/uncertainty overrides.
        """
        np.random.seed(42)
        X_train = np.random.uniform(-3.0, 3.0, size=(6, 2))
        y_train = np.sin(X_train[:, 0])

        surrogate = CustomUncertaintyRandomForest(
            uncertainty_func="standard_proximity",
            configspace=self.cs,
            n_trees=10,
            oob_score=True
        )
        surrogate.train(X_train, y_train)

        # Has predict_standard_rf method
        self.assertTrue(hasattr(surrogate, "predict_standard_rf"))
        mean_rf, var_rf = surrogate.predict_standard_rf(X_train[:2])
        self.assertEqual(mean_rf.shape, (2, 1))
        self.assertEqual(var_rf.shape, (2, 1))

        # With k=10 and N=6, N <= k triggers standard RF LCB
        acq = ProximityLowerBoundAcquisition(eps=0.10, level=0.95, k=10)
        acq.update(surrogate, num_data=6)

        X_test = X_train[:1]
        score = acq._compute(X_test).item()

        # Score must match -(mean_rf - kappa * std_rf) exactly
        mean_test, var_test = surrogate.predict_standard_rf(X_test)
        kappa = float(norm.ppf(1.0 - (1.0 - 0.95) / 2.0))
        expected_score = float(-mean_test.item() + kappa * np.sqrt(max(var_test.item(), 1e-10)))
        self.assertAlmostEqual(score, expected_score, places=5)

    def test_k_auto_warmup_fallback(self):
        """When k='auto' and N <= k_warmup, verify warmup fallback activates."""
        mock_surrogate = MockProximitySurrogate(
            y_pred=[10.0],
            y_lwr=[5.0],
            y_upr=[15.0],
            local_mae=[1.0]
        )
        mock_surrogate.n_train = 5

        acq = ProximityLowerBoundAcquisition(eps=0.10, level=0.95, k="auto", k_warmup=10)
        acq.update(mock_surrogate, num_data=5)

        X = np.array([[0.0, 0.0]])
        score = acq._compute(X).item()

        # N=5 <= k_warmup=10 -> triggers standard RF LCB
        kappa = float(norm.ppf(1.0 - (1.0 - 0.95) / 2.0))
        expected_score = -(10.0 - kappa * 0.5)
        self.assertAlmostEqual(score, expected_score, places=5)

if __name__ == "__main__":
    unittest.main()
