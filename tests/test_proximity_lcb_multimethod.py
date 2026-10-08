from __future__ import annotations
import unittest
import numpy as np
from scipy.stats import norm
from ConfigSpace import ConfigurationSpace, Float

from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
from carps_integration.proximity_lcb import ProximityLowerBoundAcquisition
from ep_extractors.standard_proximity import StandardProximityExtractor
from ep_extractors.proximity_b import ProximityBExtractor
from ep_extractors.proximity_ac import ProximityACExtractor
from ep_extractors.proximity_bc import ProximityBCExtractor


class MockSurrogateWithCallTracking:
    """Mock surrogate tracking calls to predict_standard_rf vs predict_with_intervals."""

    def __init__(
        self,
        n_train: int = 10,
        uncertainty_func: str = "standard_proximity",
        oob_mae: float = 0.5,
        extractor: object | None = None,
    ):
        self.n_train = n_train
        self.last_X = np.zeros((n_train, 2))
        self.uncertainty_func = uncertainty_func
        self.oob_mae = oob_mae
        self.uq_extractor = extractor
        self.standard_rf_calls = 0
        self.predict_with_intervals_calls = 0
        self.last_intervals_kwargs = {}

    def predict_standard_rf(self, X: np.ndarray):
        self.standard_rf_calls += 1
        n = len(X)
        mean = np.full(n, 2.0)
        var = np.full(n, 1.0)
        return mean, var

    def predict_marginalized(self, X: np.ndarray):
        return self.predict_standard_rf(X)

    def predict_with_intervals(self, X: np.ndarray, **kwargs):
        self.predict_with_intervals_calls += 1
        self.last_intervals_kwargs = kwargs
        n = len(X)
        y_pred = np.full(n, 2.0)
        # Default y_pred_lwr is narrower than delta floor to test floor binding if needed
        y_pred_lwr = kwargs.get("mock_y_lwr", np.full(n, 1.5))
        y_pred_upr = np.full(n, 2.5)
        local_mae = np.full(n, self.oob_mae)
        if kwargs.get("return_mae", False):
            return y_pred_lwr, y_pred, y_pred_upr, local_mae
        return y_pred_lwr, y_pred, y_pred_upr


class TestProximityLCBMultiMethod(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.cs = ConfigurationSpace(seed=42)
        self.cs.add([
            Float("x0", (-2.0, 2.0), default=0.0),
            Float("x1", (-2.0, 2.0), default=0.0),
        ])
        self.X_eval = np.array([[0.0, 0.0], [0.5, -0.5]])

    def test_discrete_mode_warmup_a_and_ac(self):
        """
        Verify that for discrete methods A and AC (either via method parameter or model inspection):
        - When N <= k_warmup (Phase 1): RF warmup is active (calls predict_standard_rf).
        - When N > k_warmup (Phase 2): activates predict_with_intervals with n_neighbors=k.
        """
        discrete_configs = [
            ("method_a_explicit", {"method": "a"}, "arbitrary_func"),
            ("method_ac_explicit", {"method": "ac"}, "arbitrary_func"),
            ("method_a_by_model", {}, "standard_proximity"),
            ("method_ac_by_model", {}, "proximity_ac"),
            ("method_a_cv_by_model", {}, "standard_proximity_cv"),
            ("method_ac_cv_by_model", {}, "proximity_ac_cv"),
        ]

        for desc, acq_kwargs, u_func in discrete_configs:
            with self.subTest(case=desc):
                # Phase 1: N = 10 <= k_warmup = 25
                model_warm = MockSurrogateWithCallTracking(n_train=10, uncertainty_func=u_func)
                acq = ProximityLowerBoundAcquisition(k=25, k_warmup=25, eps=0.16, level=0.95, **acq_kwargs)
                acq.update(model=model_warm, num_data=10)

                self.assertFalse(
                    acq._is_continuous_method(),
                    f"Case {desc} must NOT be classified as continuous."
                )

                scores_warm = acq._compute(self.X_eval)
                self.assertEqual(model_warm.standard_rf_calls, 1)
                self.assertEqual(model_warm.predict_with_intervals_calls, 0)
                # Verify formula: -mean + kappa * std = -2.0 + 1.95996 * 1.0
                expected_warm = -2.0 + acq._kappa * 1.0
                np.testing.assert_allclose(scores_warm, expected_warm, rtol=1e-5)

                # Phase 2: N = 30 > k_warmup = 25
                model_post = MockSurrogateWithCallTracking(n_train=30, uncertainty_func=u_func)
                acq.update(model=model_post, num_data=30)
                scores_post = acq._compute(self.X_eval)

                self.assertEqual(model_post.standard_rf_calls, 0)
                self.assertEqual(model_post.predict_with_intervals_calls, 1)
                self.assertEqual(
                    model_post.last_intervals_kwargs.get("n_neighbors"),
                    25,
                    f"Discrete mode must pass n_neighbors=25 (self._k), got {model_post.last_intervals_kwargs.get('n_neighbors')}"
                )

    def test_continuous_mode_b_and_bc_bypasses_k_warmup(self):
        """
        Verify that for continuous methods B and BC (either via method parameter or model inspection):
        - Bypasses k-warmup when N > 0: even when N=5 or N=10 <= 25, Phase 2 is immediately active.
        - Calls predict_with_intervals with n_neighbors='all'.
        - When N=0 or N is None, falls back to standard RF warmup.
        """
        continuous_configs = [
            ("method_b_explicit", {"method": "b"}, "arbitrary_func"),
            ("method_bc_explicit", {"method": "bc"}, "arbitrary_func"),
            ("method_b_cv_explicit", {"method": "b_cv"}, "arbitrary_func"),
            ("method_bc_cv_explicit", {"method": "bc_cv"}, "arbitrary_func"),
            ("method_proximity_b_cv_explicit", {"method": "proximity_b_cv"}, "arbitrary_func"),
            ("method_proximity_bc_cv_explicit", {"method": "proximity_bc_cv"}, "arbitrary_func"),
            ("method_b_by_model", {}, "proximity_b"),
            ("method_bc_by_model", {}, "proximity_bc"),
            ("method_b_cv_by_model", {}, "proximity_b_cv"),
            ("method_bc_cv_by_model", {}, "proximity_bc_cv"),
        ]

        for desc, acq_kwargs, u_func in continuous_configs:
            for n_samples in [5, 10]:
                with self.subTest(case=desc, n_samples=n_samples):
                    model = MockSurrogateWithCallTracking(n_train=n_samples, uncertainty_func=u_func)
                    acq = ProximityLowerBoundAcquisition(k=25, k_warmup=25, eps=0.16, level=0.95, **acq_kwargs)
                    acq.update(model=model, num_data=n_samples)

                    self.assertTrue(
                        acq._is_continuous_method(),
                        f"Case {desc} must be classified as continuous."
                    )

                    scores = acq._compute(self.X_eval)
                    # Must bypass RF warmup!
                    self.assertEqual(
                        model.standard_rf_calls, 0,
                        f"Continuous case {desc} with N={n_samples} should not call standard RF."
                    )
                    self.assertEqual(
                        model.predict_with_intervals_calls, 1,
                        f"Continuous case {desc} with N={n_samples} must call predict_with_intervals."
                    )
                    n_nbrs = model.last_intervals_kwargs.get("n_neighbors")
                    self.assertIn(
                        n_nbrs,
                        ("auto", "all"),
                        f"Continuous case {desc} must call predict_with_intervals with n_neighbors='auto' or 'all', got {n_nbrs}."
                    )

            # Check that N=0 or None falls back to standard RF warmup
            with self.subTest(case=f"{desc}_fallback_n0"):
                model_zero = MockSurrogateWithCallTracking(n_train=0, uncertainty_func=u_func)
                acq_zero = ProximityLowerBoundAcquisition(k=25, k_warmup=25, eps=0.16, level=0.95, **acq_kwargs)
                acq_zero.update(model=model_zero, num_data=0)
                scores_zero = acq_zero._compute(self.X_eval)
                self.assertEqual(model_zero.standard_rf_calls, 1)
                self.assertEqual(model_zero.predict_with_intervals_calls, 0)

    def test_continuous_mode_with_real_custom_rf(self):
        """
        Verify real CustomUncertaintyRandomForest surrogates with proximity_b and proximity_bc
        activate immediately with N=10 (< 25) without raising ValueError, and that query points
        at differing distances from training data produce differing acquisition values.
        """
        X_train = np.random.uniform(-0.5, 0.5, size=(10, 2))
        y_train = (X_train[:, 0] ** 2 + X_train[:, 1] ** 2).flatten()
        X_test_dist = np.array([[0.0, 0.0], [2.0, 2.0]])

        for u_func in ["proximity_b", "proximity_bc"]:
            with self.subTest(u_func=u_func):
                model = CustomUncertaintyRandomForest(
                    uncertainty_func=u_func,
                    configspace=self.cs,
                    n_trees=10,
                    seed=42,
                )
                model.train(X_train, y_train)

                acq = ProximityLowerBoundAcquisition(k=25, k_warmup=25, eps=0.16, level=0.95)
                acq.update(model=model, num_data=10)

                self.assertTrue(acq._is_continuous_method())
                scores = acq._compute(X_test_dist)
                self.assertEqual(scores.shape, (2, 1))
                self.assertTrue(np.all(np.isfinite(scores)))
                # Query points at differing distances must produce differing acquisition values
                self.assertFalse(
                    np.isclose(scores[0, 0], scores[1, 0]),
                    f"Acquisition values for close and far points must differ in continuous mode ({u_func})."
                )

    def test_floor_enforcement_all_methods(self):
        """
        Verify that delta_floor = eps * kappa * local_mae is applied across all 4 methods (A, B, AC, BC).
        Specifically, when raw y_pred_lwr is degenerate (e.g. y_pred_lwr == y_pred),
        the floored lower bound is clamped to y_pred - delta_floor.
        """
        eps = 0.16
        alpha = 1.0 - 0.95
        kappa = float(norm.ppf(1.0 - alpha / 2.0))
        local_mae_val = 0.75
        expected_delta_floor = eps * kappa * local_mae_val

        for method in ["a", "b", "ac", "bc"]:
            with self.subTest(method=method):
                # For discrete methods, use N=30 > k_warmup to test Phase 2
                # For continuous methods, use N=10 to test Phase 2
                n_train = 30 if method in ("a", "ac") else 10

                class DegenerateIntervalModel(MockSurrogateWithCallTracking):
                    def predict_with_intervals(self, X: np.ndarray, **kwargs):
                        n = len(X)
                        y_pred = np.full(n, 3.0)
                        # Degenerate: y_pred_lwr equals y_pred (no uncertainty)
                        y_pred_lwr = np.copy(y_pred)
                        y_pred_upr = np.copy(y_pred)
                        local_mae = np.full(n, local_mae_val)
                        return y_pred_lwr, y_pred, y_pred_upr, local_mae

                model = DegenerateIntervalModel(n_train=n_train, oob_mae=local_mae_val)
                acq = ProximityLowerBoundAcquisition(method=method, eps=eps, level=0.95, k=25, k_warmup=25)
                acq.update(model=model, num_data=n_train)

                scores = acq._compute(self.X_eval)
                # Raw y_pred_lwr = 3.0
                # y_pred - delta_floor = 3.0 - expected_delta_floor
                # min(3.0, 3.0 - delta_floor) = 3.0 - delta_floor
                # acquisition score = -(3.0 - delta_floor) = -3.0 + expected_delta_floor
                expected_score = -(3.0 - expected_delta_floor)
                np.testing.assert_allclose(scores, expected_score, rtol=1e-5)

    def test_update_resets_num_data_when_omitted(self):
        """
        Verify that _update resets self._num_data to None when kwargs omits num_data,
        preventing stale sample count locking in SMAC3 optimization loops.
        """
        acq = ProximityLowerBoundAcquisition(k=25, k_warmup=25)
        model_1 = MockSurrogateWithCallTracking(n_train=5)
        acq.update(model=model_1, num_data=5)
        self.assertEqual(acq._num_data, 5)

        # Subsequent update without num_data
        model_2 = MockSurrogateWithCallTracking(n_train=35)
        acq.update(model=model_2)
        self.assertIsNone(acq._num_data)

        # In _compute, N should be dynamically resolved to model_2.last_X / n_train (35 > 25)
        _ = acq._compute(self.X_eval)
        self.assertEqual(model_2.predict_with_intervals_calls, 1)

    def test_backward_compatibility_default(self):
        """
        Verify default instantiation ProximityLowerBoundAcquisition() maintains exact legacy
        defaults and discrete k-warmup behavior.
        """
        acq = ProximityLowerBoundAcquisition()
        self.assertEqual(acq._k, 25)
        self.assertEqual(acq._k_warmup, 25)
        self.assertAlmostEqual(acq._eps, 0.16, places=4)
        self.assertAlmostEqual(acq._level, 0.95, places=4)
        self.assertEqual(acq._weighting, "unweighted")
        self.assertIs(acq._weighted, False)
        self.assertIsNone(acq._method)

        # Meta dict backwards compatibility
        meta = acq.meta
        self.assertEqual(meta["k"], 25)
        self.assertEqual(meta["k_warmup"], 25)
        self.assertAlmostEqual(meta["eps"], 0.16, places=4)
        self.assertEqual(meta["weighting"], "unweighted")
        self.assertIs(meta["weighted"], False)

        # With default surrogate, should not be continuous
        model = MockSurrogateWithCallTracking(n_train=10, uncertainty_func="standard_proximity")
        acq.update(model=model, num_data=10)
        self.assertFalse(acq._is_continuous_method())

        # Phase 1 warmup active when N <= 25
        _ = acq._compute(self.X_eval)
        self.assertEqual(model.standard_rf_calls, 1)
        self.assertEqual(model.predict_with_intervals_calls, 0)

    def test_gpu_proximity_continuous_weighted_quantiles_trigger(self):
        """
        Verify that GPUProximityRegressionUQ.compute_uq and predict_with_intervals
        trigger continuous tree-walk weighted quantiles when topological_decay_lambda > 0.0
        both for n_neighbors='auto' and n_neighbors='all' (Milestone Criterion 5).
        """
        from sklearn.ensemble import RandomForestRegressor
        from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ

        np.random.seed(42)
        X_train = np.random.uniform(-1.0, 1.0, size=(20, 2))
        y_train = (X_train[:, 0] ** 2 + X_train[:, 1] ** 2).flatten()
        rf = RandomForestRegressor(n_estimators=10, random_state=42, oob_score=True).fit(X_train, y_train)

        uq = GPUProximityRegressionUQ(
            rf, X_train, y_train, device="cpu", topological_decay_lambda=1.0, use_density_scaling=False
        )
        uq.fit()

        X_test = np.array([[0.0, 0.0], [3.0, 3.0]])

        # compute_uq with n_neighbors="all" vs "auto"
        uq_sig_all = uq.compute_uq(X_test, n_neighbors="all")
        uq_sig_auto = uq.compute_uq(X_test, n_neighbors="auto")
        # Both must produce identical continuous weighted quantiles
        np.testing.assert_allclose(uq_sig_all, uq_sig_auto, rtol=1e-5)

        # predict_with_intervals with n_neighbors="all" vs "auto"
        res_all = uq.predict_with_intervals(X_test, n_neighbors="all", return_mae=True)
        res_auto = uq.predict_with_intervals(X_test, n_neighbors="auto", return_mae=True)
        # Interval lower bounds and local MAE must match
        np.testing.assert_allclose(res_all[0], res_auto[0], rtol=1e-5)
        np.testing.assert_allclose(res_all[3], res_auto[3], rtol=1e-5)
        # Local MAE differs across space in continuous mode
        self.assertFalse(np.isclose(res_all[3][0], res_all[3][1]))

    def test_gpu_proximity_discrete_mode_auto_vs_all_separation(self):
        """
        Verify that for discrete Method A/AC (topological_decay_lambda=None),
        n_neighbors='all' evaluates global training residuals (k = n_train)
        while n_neighbors='auto' computes in-leaf nanquantiles.
        """
        from sklearn.ensemble import RandomForestRegressor
        from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ

        np.random.seed(42)
        X_train = np.random.uniform(-1.0, 1.0, size=(25, 2))
        y_train = (X_train[:, 0] + X_train[:, 1]).flatten()
        rf = RandomForestRegressor(n_estimators=10, random_state=42, oob_score=True).fit(X_train, y_train)

        uq = GPUProximityRegressionUQ(
            rf, X_train, y_train, device="cpu", topological_decay_lambda=None, use_density_scaling=False
        )
        uq.fit()

        X_test = np.array([[0.0, 0.0], [0.5, 0.5]])

        # Discrete n_neighbors="all" evaluates all training points (k=25)
        uq_all = uq.compute_uq(X_test, n_neighbors="all")
        # Global residual interval width is constant across query points
        self.assertTrue(np.isclose(uq_all[0], uq_all[1]))

        # Discrete n_neighbors="auto" evaluates only in-leaf samples, producing point-specific interval widths
        uq_auto = uq.compute_uq(X_test, n_neighbors="auto")
        self.assertFalse(np.isclose(uq_auto[0], uq_auto[1]))

        # predict_with_intervals interval width check
        res_all = uq.predict_with_intervals(X_test, n_neighbors="all")
        res_all_width = res_all[2] - res_all[0]
        self.assertTrue(np.isclose(res_all_width[0], res_all_width[1]))

        res_auto = uq.predict_with_intervals(X_test, n_neighbors="auto")
        res_auto_width = res_auto[2] - res_auto[0]
        self.assertFalse(np.isclose(res_auto_width[0], res_auto_width[1]))


if __name__ == "__main__":
    unittest.main()
