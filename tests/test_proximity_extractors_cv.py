from __future__ import annotations
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ConfigSpace import ConfigurationSpace, Float

from ep_extractors import UQExtractorRegistry
from ep_extractors.proximity_b import ProximityBExtractor, ProximityBCVExtractor
from ep_extractors.proximity_ac import ProximityACExtractor, ProximityACCVExtractor
from ep_extractors.proximity_bc import ProximityBCExtractor, ProximityBCCVExtractor
from ep_extractors.standard_proximity import StandardProximityExtractor
from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest


class TestUQExtractorRegistryResolution(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.uniform(-2.0, 2.0, size=(30, 3))
        self.y_train = (
            np.sin(self.X_train[:, 0])
            + 0.5 * self.X_train[:, 1] ** 2
            + np.random.normal(0, 0.05, size=30)
        )
        self.rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=4,
            random_state=42,
            bootstrap=True,
            oob_score=True,
        )
        self.rf.fit(self.X_train, self.y_train)

    def test_registry_contains_all_variants(self):
        registered = UQExtractorRegistry.list_registered()
        for key in ["proximity_b_cv", "proximity_ac", "proximity_ac_cv", "proximity_bc_cv"]:
            self.assertIn(key, registered, f"Registry key '{key}' must be registered.")

    def test_proximity_b_cv_resolution(self):
        extractor = UQExtractorRegistry.get("proximity_b_cv", self.rf)
        self.assertIsInstance(extractor, ProximityBExtractor)
        self.assertIsInstance(extractor, ProximityBCVExtractor)
        self.assertEqual(extractor.residual_mode, "cv")
        self.assertEqual(extractor.cv_folds, 5)
        self.assertAlmostEqual(extractor.decay_lambda, 1.345, places=4)

    def test_proximity_ac_resolution(self):
        extractor = UQExtractorRegistry.get("proximity_ac", self.rf)
        self.assertIsInstance(extractor, ProximityACExtractor)
        self.assertEqual(extractor.residual_mode, "oob")
        self.assertEqual(extractor.cv_folds, 5)
        self.assertAlmostEqual(extractor.alpha, 1.0, places=4)

    def test_proximity_ac_cv_resolution(self):
        extractor = UQExtractorRegistry.get("proximity_ac_cv", self.rf)
        self.assertIsInstance(extractor, ProximityACExtractor)
        self.assertIsInstance(extractor, ProximityACCVExtractor)
        self.assertEqual(extractor.residual_mode, "cv")
        self.assertEqual(extractor.cv_folds, 5)
        self.assertAlmostEqual(extractor.alpha, 1.0, places=4)

    def test_proximity_bc_cv_resolution(self):
        extractor = UQExtractorRegistry.get("proximity_bc_cv", self.rf)
        self.assertIsInstance(extractor, ProximityBCExtractor)
        self.assertIsInstance(extractor, ProximityBCCVExtractor)
        self.assertEqual(extractor.residual_mode, "cv")
        self.assertEqual(extractor.cv_folds, 5)
        self.assertAlmostEqual(extractor.alpha, 1.0, places=4)


class TestProximityCVResidualForwarding(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 40
        self.X_train = np.random.uniform(-2.0, 2.0, size=(self.n_samples, 3))
        self.y_train = (
            np.sin(self.X_train[:, 0])
            + 0.5 * self.X_train[:, 1] ** 2
            + np.random.normal(0, 0.05, size=self.n_samples)
        )
        self.rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=4,
            random_state=42,
            bootstrap=True,
            oob_score=True,
        )
        self.rf.fit(self.X_train, self.y_train)

    def test_proximity_b_cv_residual_forwarding(self):
        extractor = UQExtractorRegistry.get("proximity_b_cv", self.rf, device="cpu", cv_folds=5)
        extractor.fit(self.X_train, self.y_train)

        self.assertIsNotNone(extractor.uq_model)
        self.assertEqual(extractor.uq_model.residual_mode, "cv")
        self.assertEqual(extractor.uq_model.cv_folds, 5)
        self.assertEqual(len(extractor.uq_model.oob_residuals), self.n_samples)
        self.assertTrue(np.all(np.isfinite(extractor.uq_model.oob_residuals)))
        self.assertTrue(np.all(extractor.uq_model.valid_oob_mask))
        self.assertEqual(np.sum(extractor.uq_model.valid_oob_mask), self.n_samples)

    def test_proximity_ac_cv_residual_forwarding(self):
        extractor = UQExtractorRegistry.get("proximity_ac_cv", self.rf, device="cpu", cv_folds=5)
        extractor.fit(self.X_train, self.y_train)

        self.assertIsNotNone(extractor.uq_model)
        self.assertEqual(extractor.uq_model.residual_mode, "cv")
        self.assertEqual(extractor.uq_model.cv_folds, 5)
        self.assertEqual(len(extractor.uq_model.oob_residuals), self.n_samples)
        self.assertTrue(np.all(np.isfinite(extractor.uq_model.oob_residuals)))
        self.assertTrue(np.all(extractor.uq_model.valid_oob_mask))
        self.assertEqual(np.sum(extractor.uq_model.valid_oob_mask), self.n_samples)

    def test_proximity_bc_cv_residual_forwarding(self):
        extractor = UQExtractorRegistry.get("proximity_bc_cv", self.rf, device="cpu", cv_folds=5)
        extractor.fit(self.X_train, self.y_train)

        self.assertIsNotNone(extractor.uq_model)
        self.assertEqual(extractor.uq_model.residual_mode, "cv")
        self.assertEqual(extractor.uq_model.cv_folds, 5)
        self.assertEqual(len(extractor.uq_model.oob_residuals), self.n_samples)
        self.assertTrue(np.all(np.isfinite(extractor.uq_model.oob_residuals)))
        self.assertTrue(np.all(extractor.uq_model.valid_oob_mask))
        self.assertEqual(np.sum(extractor.uq_model.valid_oob_mask), self.n_samples)

    def test_explicit_residual_mode_and_custom_folds(self):
        # Test ProximityBExtractor explicit residual_mode="cv" and cv_folds=4
        b_ext = ProximityBExtractor(self.rf, device="cpu", residual_mode="cv", cv_folds=4)
        self.assertEqual(b_ext.residual_mode, "cv")
        self.assertEqual(b_ext.cv_folds, 4)
        b_ext.fit(self.X_train, self.y_train)
        self.assertEqual(b_ext.uq_model.residual_mode, "cv")
        self.assertEqual(b_ext.uq_model.cv_folds, 4)
        self.assertEqual(len(b_ext.uq_model.oob_residuals), self.n_samples)

        # Test ProximityBCExtractor explicit residual_mode="cv" and cv_folds=3
        bc_ext = ProximityBCExtractor(self.rf, device="cpu", residual_mode="cv", cv_folds=3)
        self.assertEqual(bc_ext.residual_mode, "cv")
        self.assertEqual(bc_ext.cv_folds, 3)
        bc_ext.fit(self.X_train, self.y_train)
        self.assertEqual(bc_ext.uq_model.residual_mode, "cv")
        self.assertEqual(bc_ext.uq_model.cv_folds, 3)
        self.assertEqual(len(bc_ext.uq_model.oob_residuals), self.n_samples)


class TestProximityACDensityScaling(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Cluster of 40 points in [-1, 1], and a few outer points
        self.X_train = np.random.uniform(-1.0, 1.0, size=(50, 2))
        self.y_train = self.X_train[:, 0] ** 2 + self.X_train[:, 1] ** 2 + np.random.normal(0, 0.05, size=50)

        self.rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=4,
            random_state=42,
            bootstrap=True,
            oob_score=True,
        )
        self.rf.fit(self.X_train, self.y_train)

        # Test points: one dense central point and one distant extrapolation point
        self.X_test = np.array([
            [0.0, 0.0],
            [3.5, 3.5],
        ])

    def test_proximity_ac_configuration(self):
        ext = ProximityACExtractor(self.rf, device="cpu", alpha=1.0)
        ext.fit(self.X_train, self.y_train)

        self.assertIsNotNone(ext.uq_model)
        self.assertTrue(ext.uq_model.use_density_scaling)
        self.assertEqual(ext.uq_model.density_scaling_alpha, 1.0)
        self.assertIsNone(ext.uq_model.topological_decay_lambda)
        self.assertGreater(ext.uq_model.N_baseline, 0.0)

    def test_proximity_ac_density_multiplier_floor(self):
        ext_ac = ProximityACExtractor(self.rf, device="cpu", alpha=1.0)
        ext_ac.fit(self.X_train, self.y_train)

        ext_std = StandardProximityExtractor(self.rf, device="cpu")
        ext_std.fit(self.X_train, self.y_train)

        sig_ac = ext_ac.extract_epistemic_signal(self.X_test)
        sig_std = ext_std.extract_epistemic_signal(self.X_test)

        self.assertEqual(sig_ac.shape, (len(self.X_test),))
        self.assertTrue(np.all(np.isfinite(sig_ac)))
        self.assertTrue(np.all(sig_ac >= 0.0))

        # Because gamma(x) = max(1.0, (N_baseline / N_leaf_bar(x))^alpha) >= 1.0,
        # sig_ac >= sig_std everywhere (with small float tolerance)
        self.assertTrue(np.all(sig_ac >= sig_std - 1e-6), f"AC: {sig_ac}, Std: {sig_std}")

        # For the distant extrapolation point [3.5, 3.5], leaf size is likely small,
        # causing gamma > 1.0 or at minimum gamma >= 1.0
        ratio = sig_ac / np.maximum(sig_std, 1e-10)
        self.assertTrue(np.all(ratio >= 1.0 - 1e-6))

    def test_proximity_ac_cv_predict_with_intervals(self):
        ext_cv = UQExtractorRegistry.get("proximity_ac_cv", self.rf, device="cpu")
        ext_cv.fit(self.X_train, self.y_train)

        y_lwr, y_pred, y_upr, mae = ext_cv.predict_with_intervals(self.X_test, level=0.9, return_mae=True)
        self.assertTrue(np.all(np.isfinite(y_lwr)))
        self.assertTrue(np.all(np.isfinite(y_pred)))
        self.assertTrue(np.all(np.isfinite(y_upr)))
        self.assertTrue(np.all(np.isfinite(mae)))
        self.assertTrue(np.all(y_lwr <= y_pred + 1e-6))
        self.assertTrue(np.all(y_pred <= y_upr + 1e-6))
        self.assertTrue(np.all(mae >= 0.0))


class TestProximityContinuousQuantiles(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.uniform(-2.0, 2.0, size=(45, 3))
        self.y_train = (
            np.sin(self.X_train[:, 0])
            + 0.5 * self.X_train[:, 1] ** 2
            + np.random.normal(0, 0.05, size=45)
        )
        self.X_test = np.random.uniform(-2.5, 2.5, size=(15, 3))

        self.rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=4,
            random_state=42,
            bootstrap=True,
            oob_score=True,
        )
        self.rf.fit(self.X_train, self.y_train)

    def test_proximity_b_cv_continuous_quantiles(self):
        """
        Verify Proximity B continuous quantiles evaluate cleanly over out-of-fold residuals with lambda=1.345.
        """
        extractor = UQExtractorRegistry.get(
            "proximity_b_cv", self.rf, device="cpu", decay_lambda=1.345, cv_folds=5
        )
        extractor.fit(self.X_train, self.y_train)

        self.assertEqual(extractor.uq_model.residual_mode, "cv")
        self.assertEqual(extractor.uq_model.topological_decay_lambda, 1.345)
        self.assertEqual(len(extractor.uq_model.oob_residuals), len(self.y_train))
        self.assertTrue(np.all(np.isfinite(extractor.uq_model.oob_residuals)))

        sig = extractor.extract_epistemic_signal(self.X_test)
        self.assertEqual(sig.shape, (len(self.X_test),))
        self.assertTrue(np.all(np.isfinite(sig)))
        self.assertTrue(np.all(sig >= 0.0))

        y_lwr, y_pred, y_upr, mae = extractor.predict_with_intervals(
            self.X_test, level=0.95, return_mae=True
        )
        self.assertTrue(np.all(np.isfinite(y_lwr)))
        self.assertTrue(np.all(np.isfinite(y_pred)))
        self.assertTrue(np.all(np.isfinite(y_upr)))
        self.assertTrue(np.all(np.isfinite(mae)))
        self.assertTrue(np.all(y_lwr <= y_upr + 1e-5))
        self.assertTrue(np.all(mae >= 0.0))

    def test_proximity_bc_cv_continuous_quantiles(self):
        """
        Verify Proximity BC continuous quantiles evaluate cleanly over out-of-fold residuals with lambda=1.345 and alpha=1.0.
        """
        extractor = UQExtractorRegistry.get(
            "proximity_bc_cv", self.rf, device="cpu", decay_lambda=1.345, alpha=1.0, cv_folds=5
        )
        extractor.fit(self.X_train, self.y_train)

        self.assertEqual(extractor.uq_model.residual_mode, "cv")
        self.assertEqual(extractor.uq_model.topological_decay_lambda, 1.345)
        self.assertEqual(extractor.uq_model.density_scaling_alpha, 1.0)
        self.assertTrue(extractor.uq_model.use_density_scaling)
        self.assertEqual(len(extractor.uq_model.oob_residuals), len(self.y_train))
        self.assertTrue(np.all(np.isfinite(extractor.uq_model.oob_residuals)))

        sig = extractor.extract_epistemic_signal(self.X_test)
        self.assertEqual(sig.shape, (len(self.X_test),))
        self.assertTrue(np.all(np.isfinite(sig)))
        self.assertTrue(np.all(sig >= 0.0))

        y_lwr, y_pred, y_upr, mae = extractor.predict_with_intervals(
            self.X_test, level=0.95, return_mae=True
        )
        self.assertTrue(np.all(np.isfinite(y_lwr)))
        self.assertTrue(np.all(np.isfinite(y_pred)))
        self.assertTrue(np.all(np.isfinite(y_upr)))
        self.assertTrue(np.all(np.isfinite(mae)))
        self.assertTrue(np.all(y_lwr <= y_upr + 1e-5))
        self.assertTrue(np.all(mae >= 0.0))


class TestCustomUncertaintyRandomForestIntegration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.uniform(-2.0, 2.0, size=(35, 3))
        self.y_train = (
            np.sin(self.X_train[:, 0])
            + 0.5 * self.X_train[:, 1] ** 2
            + np.random.normal(0, 0.05, size=35)
        )
        self.X_test = np.random.uniform(-2.0, 2.0, size=(10, 3))

        self.cs = ConfigurationSpace(seed=42)
        self.cs.add([
            Float("x0", (-2.0, 2.0), default=0.0),
            Float("x1", (-2.0, 2.0), default=0.0),
            Float("x2", (-2.0, 2.0), default=0.0),
        ])

    def test_custom_uncertainty_rf_with_cv_extractors(self):
        for func_name in ["proximity_b_cv", "proximity_ac", "proximity_ac_cv", "proximity_bc_cv"]:
            model = CustomUncertaintyRandomForest(
                uncertainty_func=func_name,
                configspace=self.cs,
                n_trees=10,
                seed=42,
            )
            model.train(self.X_train, self.y_train)

            self.assertIsNotNone(model.uq_extractor, f"UQ extractor for '{func_name}' must be initialized")
            if "_cv" in func_name:
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

            # Check SMAC3 predict
            mean, var = model.predict(self.X_test)
            self.assertEqual(mean.shape, (len(self.X_test), 1))
            self.assertEqual(var.shape, (len(self.X_test), 1))
            self.assertTrue(np.all(np.isfinite(mean)))
            self.assertTrue(np.all(np.isfinite(var)))
            self.assertTrue(np.all(var > 0.0))
