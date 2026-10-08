from __future__ import annotations
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ConfigSpace import ConfigurationSpace, Float

from ep_extractors import UQExtractorRegistry
from ep_extractors.standard_proximity import StandardProximityExtractor
from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest


class TestStandardProximityCVExtractor(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 40 samples, 3 features
        self.X_train = np.random.uniform(-2.0, 2.0, size=(40, 3))
        self.y_train = (
            np.sin(self.X_train[:, 0])
            + 0.5 * self.X_train[:, 1] ** 2
            + np.random.normal(0, 0.05, size=40)
        )
        self.X_test = np.random.uniform(-2.0, 2.0, size=(12, 3))

        self.rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=4,
            random_state=42,
            bootstrap=True,
            oob_score=True,
        )
        self.rf.fit(self.X_train, self.y_train)

    def test_registry_resolution(self):
        """
        Verify UQExtractorRegistry.get("standard_proximity_cv", rf_model) correctly resolves,
        instantiates, and sets residual_mode="cv" and cv_folds=5 by default.
        """
        extractor = UQExtractorRegistry.get("standard_proximity_cv", self.rf)
        self.assertIsInstance(extractor, StandardProximityExtractor)
        self.assertEqual(extractor.residual_mode, "cv")
        self.assertEqual(extractor.cv_folds, 5)

    def test_standard_proximity_extractor_explicit_kwargs(self):
        """
        Verify StandardProximityExtractor can be instantiated with residual_mode="cv"
        and custom cv_folds, and passes them to GPUProximityRegressionUQ upon fit().
        """
        extractor = StandardProximityExtractor(
            self.rf,
            device="cpu",
            residual_mode="cv",
            cv_folds=4,
        )
        self.assertEqual(extractor.residual_mode, "cv")
        self.assertEqual(extractor.cv_folds, 4)

        extractor.fit(self.X_train, self.y_train)
        self.assertIsNotNone(extractor.uq_model)
        self.assertEqual(extractor.uq_model.residual_mode, "cv")
        self.assertEqual(extractor.uq_model.cv_folds, 4)

    def test_extractor_fit_and_signal(self):
        """
        Verify fitting the extractor populates uq_model with CV residuals
        and extract_epistemic_signal returns finite non-negative values.
        """
        extractor = UQExtractorRegistry.get("standard_proximity_cv", self.rf, device="cpu")
        extractor.fit(self.X_train, self.y_train)

        self.assertIsNotNone(extractor.uq_model)
        self.assertEqual(len(extractor.uq_model.oob_residuals), len(self.y_train))
        self.assertTrue(np.all(np.isfinite(extractor.uq_model.oob_residuals)))
        self.assertTrue(np.all(extractor.uq_model.valid_oob_mask))

        signal = extractor.extract_epistemic_signal(self.X_test)
        self.assertEqual(signal.shape, (len(self.X_test),))
        self.assertTrue(np.all(np.isfinite(signal)))
        self.assertTrue(np.all(signal >= 0.0))

    def test_extractor_predict_with_intervals(self):
        """
        Verify predict_with_intervals succeeds, returning valid bounds and local MAE.
        """
        extractor = UQExtractorRegistry.get("standard_proximity_cv", self.rf, device="cpu")
        extractor.fit(self.X_train, self.y_train)

        # With MAE
        preds = extractor.predict_with_intervals(self.X_test, level=0.9, return_mae=True)
        self.assertEqual(len(preds), 4)
        y_lwr, y_pred, y_upr, mae = preds

        self.assertEqual(y_lwr.shape, (len(self.X_test),))
        self.assertEqual(y_pred.shape, (len(self.X_test),))
        self.assertEqual(y_upr.shape, (len(self.X_test),))
        self.assertEqual(mae.shape, (len(self.X_test),))

        self.assertTrue(np.all(np.isfinite(y_lwr)))
        self.assertTrue(np.all(np.isfinite(y_pred)))
        self.assertTrue(np.all(np.isfinite(y_upr)))
        self.assertTrue(np.all(np.isfinite(mae)))

        self.assertTrue(np.all(y_lwr <= y_pred + 1e-6))
        self.assertTrue(np.all(y_pred <= y_upr + 1e-6))
        self.assertTrue(np.all(mae >= 0.0))

        # Without MAE
        preds_no_mae = extractor.predict_with_intervals(self.X_test, level=0.9, return_mae=False)
        self.assertEqual(len(preds_no_mae), 3)
        y_lwr_nm, y_pred_nm, y_upr_nm = preds_no_mae
        np.testing.assert_allclose(y_lwr, y_lwr_nm, atol=1e-6)
        np.testing.assert_allclose(y_pred, y_pred_nm, atol=1e-6)
        np.testing.assert_allclose(y_upr, y_upr_nm, atol=1e-6)

    def test_custom_uncertainty_rf_integration(self):
        """
        Verify CustomUncertaintyRandomForest(uncertainty_func="standard_proximity_cv")
        fits and calls predict_with_intervals and predict seamlessly.
        """
        cs = ConfigurationSpace(seed=42)
        cs.add([
            Float("x0", (-2.0, 2.0), default=0.0),
            Float("x1", (-2.0, 2.0), default=0.0),
            Float("x2", (-2.0, 2.0), default=0.0),
        ])

        model = CustomUncertaintyRandomForest(
            uncertainty_func="standard_proximity_cv",
            configspace=cs,
            n_trees=10,
            seed=42,
        )
        model.train(self.X_train, self.y_train)

        self.assertIsNotNone(model.uq_extractor)
        self.assertEqual(getattr(model.uq_extractor, "residual_mode", None), "cv")
        self.assertEqual(getattr(model.uq_extractor, "cv_folds", None), 5)

        # Check predict_with_intervals
        y_lwr, y_mean, y_upr = model.predict_with_intervals(self.X_test, return_mae=False)
        self.assertEqual(y_lwr.shape, (len(self.X_test),))
        self.assertEqual(y_mean.shape, (len(self.X_test),))
        self.assertEqual(y_upr.shape, (len(self.X_test),))
        self.assertTrue(np.all(np.isfinite(y_lwr)))
        self.assertTrue(np.all(np.isfinite(y_mean)))
        self.assertTrue(np.all(np.isfinite(y_upr)))
        self.assertTrue(np.all(y_upr >= y_lwr - 1e-6))

        # Check predict() SMAC interface
        mean, var = model.predict(self.X_test)
        self.assertEqual(mean.shape, (len(self.X_test), 1))
        self.assertEqual(var.shape, (len(self.X_test), 1))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.isfinite(var)))
        self.assertTrue(np.all(var >= 0.0))

    def test_extractor_small_n_fallback(self):
        """
        Verify that when N=4 < 5, extractor.fit() completes cleanly using the OOB fallback
        and emits a UserWarning.
        """
        import pytest
        X_small = self.X_train[:4]
        y_small = self.y_train[:4]
        rf_small = RandomForestRegressor(n_estimators=10, random_state=42, bootstrap=True, oob_score=True)
        rf_small.fit(X_small, y_small)

        extractor = UQExtractorRegistry.get("standard_proximity_cv", rf_small, device="cpu")
        with pytest.warns(UserWarning, match=r"smaller than cv_folds"):
            extractor.fit(X_small, y_small)

        self.assertIsNotNone(extractor.uq_model)
        self.assertEqual(len(extractor.uq_model.oob_residuals), 4)
        self.assertTrue(np.all(np.isfinite(extractor.uq_model.oob_residuals)))


if __name__ == "__main__":
    unittest.main()
