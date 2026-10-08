from __future__ import annotations
import unittest
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
from carps_integration.proximity_lcb import ProximityLowerBoundAcquisition


class MockSurrogateTrackingFallback:
    def __init__(self, n_train: int = 10, oob_mae: float = 0.5):
        self.n_train = n_train
        self.last_X = np.zeros((n_train, 2))
        self.oob_mae = oob_mae
        self.uncertainty_func = "proximity_b"
        self.last_intervals_kwargs = {}

    def predict_with_intervals(self, X: np.ndarray, **kwargs):
        self.last_intervals_kwargs = kwargs
        n = len(X)
        y_pred = np.full(n, 2.0)
        y_pred_lwr = np.full(n, 1.0)
        y_pred_upr = np.full(n, 3.0)
        local_mae = np.full(n, self.oob_mae)
        if kwargs.get("return_mae", False):
            return y_pred_lwr, y_pred, y_pred_upr, local_mae
        return y_pred_lwr, y_pred, y_pred_upr


class TestProximityLowSupportFallback(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Synthetic dataset: 40 points in dense cluster [-0.5, 0.5]^2, 2 sparse points at [10.0, 10.0]
        X_cluster = np.random.uniform(-0.5, 0.5, size=(40, 2))
        X_sparse = np.array([[10.0, 10.0], [10.1, 10.1]])
        self.X_train = np.vstack([X_cluster, X_sparse])
        self.y_train = (
            np.sin(self.X_train[:, 0] * np.pi)
            + 0.5 * self.X_train[:, 1]
            + np.random.normal(0, 0.05, size=len(self.X_train))
        )

        self.rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            bootstrap=True,
            oob_score=True,
        )
        self.rf.fit(self.X_train, self.y_train)

        # In-distribution point in dense training cluster (high support: k_eff >= 3.0)
        self.X_indist = np.array([[0.1, 0.1]])
        # Query point near sparse region (low support: k_eff < 3.0)
        self.X_low_support = np.array([[10.0, 10.0]])

    def test_fallback_on_low_support_false_default(self):
        """
        Verify that default is fallback_on_low_support=False and warn_on_low_support=False.
        Continuous Method B computes pure continuous tree-walk quantiles without falling back.
        """
        uq_model = GPUProximityRegressionUQ(
            self.rf,
            self.X_train,
            self.y_train,
            device="cpu",
            topological_decay_lambda=5.0,
        )

        self.assertFalse(uq_model.fallback_on_low_support)
        self.assertFalse(uq_model.warn_on_low_support)

        # Predict on low support query point
        lwr_def, _, upr_def, _ = uq_model.predict_with_intervals(
            self.X_low_support, n_neighbors="auto", return_mae=True
        )

        # Explicit fallback_on_low_support=False must produce identical results
        uq_explicit_false = GPUProximityRegressionUQ(
            self.rf,
            self.X_train,
            self.y_train,
            device="cpu",
            topological_decay_lambda=5.0,
            fallback_on_low_support=False,
        )
        lwr_exp, _, upr_exp, _ = uq_explicit_false.predict_with_intervals(
            self.X_low_support, n_neighbors="auto", return_mae=True
        )

        np.testing.assert_allclose(lwr_def, lwr_exp)
        np.testing.assert_allclose(upr_def, upr_exp)

    def test_fallback_on_low_support_true_activates_proximity_a(self):
        """
        Verify that when fallback_on_low_support=True:
        1. Low-support query points (where k_eff < 3) receive Proximity A discrete leaf co-occurrence intervals.
        2. High-support in-distribution query points continue using pure continuous tree-walk quantiles.
        """
        # Continuous Method B without fallback
        uq_continuous = GPUProximityRegressionUQ(
            self.rf,
            self.X_train,
            self.y_train,
            device="cpu",
            topological_decay_lambda=5.0,
            fallback_on_low_support=False,
        )

        # Continuous Method B WITH fallback
        uq_fallback = GPUProximityRegressionUQ(
            self.rf,
            self.X_train,
            self.y_train,
            device="cpu",
            topological_decay_lambda=5.0,
            fallback_on_low_support=True,
        )

        # Discrete Proximity A reference model (top-25 leaf co-occurrence)
        uq_prox_a = GPUProximityRegressionUQ(
            self.rf,
            self.X_train,
            self.y_train,
            device="cpu",
            topological_decay_lambda=None,
        )

        # Check low support query point
        lwr_cont, _, upr_cont, _ = uq_continuous.predict_with_intervals(
            self.X_low_support, n_neighbors="auto", return_mae=True
        )
        lwr_fb, _, upr_fb, mae_fb = uq_fallback.predict_with_intervals(
            self.X_low_support, n_neighbors="auto", return_mae=True
        )
        lwr_a, _, upr_a, mae_a = uq_prox_a.predict_with_intervals(
            self.X_low_support, n_neighbors=25, return_mae=True
        )

        # Low support point with fallback should match Proximity A discrete leaf co-occurrence
        np.testing.assert_allclose(
            lwr_fb,
            lwr_a,
            atol=1e-5,
            err_msg="Low support lower interval should match Proximity A discrete quantiles",
        )
        np.testing.assert_allclose(
            upr_fb,
            upr_a,
            atol=1e-5,
            err_msg="Low support upper interval should match Proximity A discrete quantiles",
        )
        np.testing.assert_allclose(
            mae_fb,
            mae_a,
            atol=1e-5,
            err_msg="Low support MAE should match Proximity A discrete MAE",
        )

        # Verify it diverges from pure continuous Method B on this low support point
        self.assertFalse(
            np.allclose(lwr_cont, lwr_fb, atol=1e-3),
            "Pure continuous interval should differ from Proximity A fallback on low support",
        )

        # Check in-distribution point (high support): should NOT fall back to Proximity A
        lwr_cont_in, _, upr_cont_in, _ = uq_continuous.predict_with_intervals(
            self.X_indist, n_neighbors="auto", return_mae=True
        )
        lwr_fb_in, _, upr_fb_in, _ = uq_fallback.predict_with_intervals(
            self.X_indist, n_neighbors="auto", return_mae=True
        )
        np.testing.assert_allclose(
            lwr_cont_in,
            lwr_fb_in,
            atol=1e-5,
            err_msg="In-distribution high support points should keep pure continuous quantiles",
        )
        np.testing.assert_allclose(
            upr_cont_in,
            upr_fb_in,
            atol=1e-5,
            err_msg="In-distribution high support points should keep pure continuous quantiles",
        )

    def test_warn_on_low_support_silenced_by_default(self):
        """
        Verify that:
        1. No UserWarning is raised when warn_on_low_support=False (default).
        2. When warn_on_low_support=True, a warning is raised at most once per fit.
        """
        uq_silent = GPUProximityRegressionUQ(
            self.rf,
            self.X_train,
            self.y_train,
            device="cpu",
            topological_decay_lambda=5.0,
            warn_on_low_support=False,
        )

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            for _ in range(5):
                _ = uq_silent.predict_with_intervals(self.X_low_support, n_neighbors="auto")
                _ = uq_silent.compute_uq(self.X_low_support, n_neighbors="auto")

            low_support_warnings = [
                w for w in captured if "Insufficient valid OOB neighbor support" in str(w.message)
            ]
            self.assertEqual(
                len(low_support_warnings),
                0,
                f"Expected 0 warnings when warn_on_low_support=False, got {len(low_support_warnings)}",
            )

        # When warn_on_low_support=True, emit at most once per fit
        uq_warn = GPUProximityRegressionUQ(
            self.rf,
            self.X_train,
            self.y_train,
            device="cpu",
            topological_decay_lambda=5.0,
            warn_on_low_support=True,
        )

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            for _ in range(5):
                _ = uq_warn.predict_with_intervals(self.X_low_support, n_neighbors="auto")
                _ = uq_warn.compute_uq(self.X_low_support, n_neighbors="auto")

            low_support_warnings = [
                w for w in captured if "Insufficient valid OOB neighbor support" in str(w.message)
            ]
            self.assertEqual(
                len(low_support_warnings),
                1,
                f"Expected exactly 1 warning across multiple batches per fit, got {len(low_support_warnings)}",
            )

        # Re-fitting resets the warning flag to allow 1 warning for the new fit
        uq_warn.fit()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            _ = uq_warn.predict_with_intervals(self.X_low_support, n_neighbors="auto")
            low_support_warnings = [
                w for w in captured if "Insufficient valid OOB neighbor support" in str(w.message)
            ]
            self.assertEqual(
                len(low_support_warnings),
                1,
                "Expected 1 warning after re-fit",
            )

    def test_acquisition_forwards_fallback_flag(self):
        """
        Verify that ProximityLowerBoundAcquisition accepts fallback_on_low_support,
        records it in meta, and passes it to predict_with_intervals.
        """
        # Test default
        acq_default = ProximityLowerBoundAcquisition()
        self.assertFalse(acq_default._fallback_on_low_support)
        self.assertFalse(acq_default.meta["fallback_on_low_support"])

        # Test explicit True
        acq_true = ProximityLowerBoundAcquisition(fallback_on_low_support=True)
        self.assertTrue(acq_true._fallback_on_low_support)
        self.assertTrue(acq_true.meta["fallback_on_low_support"])

        # Verify forwarding to model.predict_with_intervals
        mock_model = MockSurrogateTrackingFallback(n_train=20)
        acq_true.update(model=mock_model, num_data=20)
        _ = acq_true._compute(self.X_indist)

        self.assertIn("fallback_on_low_support", mock_model.last_intervals_kwargs)
        self.assertTrue(mock_model.last_intervals_kwargs["fallback_on_low_support"])

        # Verify forwarding with False
        acq_false = ProximityLowerBoundAcquisition(fallback_on_low_support=False)
        acq_false.update(model=mock_model, num_data=20)
        _ = acq_false._compute(self.X_indist)

        self.assertIn("fallback_on_low_support", mock_model.last_intervals_kwargs)
        self.assertFalse(mock_model.last_intervals_kwargs["fallback_on_low_support"])
