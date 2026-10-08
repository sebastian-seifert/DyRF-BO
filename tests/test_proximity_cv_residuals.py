from __future__ import annotations
import unittest
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ


class TestProximityCVResiduals(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Synthetic dataset with 50 samples, 4 features
        self.X_train = np.random.uniform(-2.0, 2.0, size=(50, 4))
        self.y_train = (
            np.sin(self.X_train[:, 0])
            + 0.5 * self.X_train[:, 1] ** 2
            - 0.3 * self.X_train[:, 2]
            + np.random.normal(0, 0.05, size=50)
        )
        self.X_test = np.random.uniform(-2.0, 2.0, size=(10, 4))

        self.rf = RandomForestRegressor(
            n_estimators=15,
            max_depth=4,
            random_state=42,
            bootstrap=True,
            oob_score=True,
        )
        self.rf.fit(self.X_train, self.y_train)

    def test_backward_compatibility_oob(self):
        """
        Verify residual_mode='oob' produces exact legacy residuals,
        and default initialization uses residual_mode='oob'.
        """
        rf1 = RandomForestRegressor(n_estimators=10, random_state=123, bootstrap=True, oob_score=True)
        rf1.fit(self.X_train, self.y_train)
        rf2 = RandomForestRegressor(n_estimators=10, random_state=123, bootstrap=True, oob_score=True)
        rf2.fit(self.X_train, self.y_train)

        uq_default = GPUProximityRegressionUQ(rf1, self.X_train, self.y_train, device="cpu")
        self.assertEqual(getattr(uq_default, "residual_mode", None), "oob")

        uq_oob = GPUProximityRegressionUQ(
            rf2, self.X_train, self.y_train, device="cpu", residual_mode="oob"
        )
        self.assertEqual(uq_oob.residual_mode, "oob")

        # Check identical residuals
        np.testing.assert_allclose(uq_default.oob_residuals, uq_oob.oob_residuals, atol=1e-7)
        np.testing.assert_array_equal(uq_default.valid_oob_mask, uq_oob.valid_oob_mask)

    def test_5fold_cv_residual_generation(self):
        """
        Verify residual_mode='cv' with 5 folds computes genuine out-of-fold residuals,
        all N residuals are finite and valid, and verify mathematical correctness vs manual fold fitting.
        """
        rf = RandomForestRegressor(n_estimators=10, random_state=42, bootstrap=True, oob_score=True)
        rf.fit(self.X_train, self.y_train)

        uq_cv = GPUProximityRegressionUQ(
            rf, self.X_train, self.y_train, device="cpu", residual_mode="cv", cv_folds=5
        )

        self.assertEqual(uq_cv.residual_mode, "cv")
        self.assertEqual(uq_cv.cv_folds, 5)
        self.assertEqual(len(uq_cv.oob_residuals), len(self.y_train))
        self.assertTrue(np.all(np.isfinite(uq_cv.oob_residuals)))
        self.assertTrue(np.all(uq_cv.valid_oob_mask))
        self.assertEqual(np.sum(uq_cv.valid_oob_mask), len(self.y_train))

        # Mathematical verification: manually replicate 5-fold CV using sklearn clone
        seed = getattr(rf, "random_state", None)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        manual_residuals = np.zeros(len(self.y_train), dtype=np.float32)

        for train_idx, val_idx in kf.split(self.X_train):
            sub_model = clone(rf)
            if hasattr(sub_model, "oob_score"):
                sub_model.set_params(oob_score=False)
            sub_model.fit(self.X_train[train_idx], self.y_train[train_idx])
            y_pred_val = sub_model.predict(self.X_train[val_idx])
            manual_residuals[val_idx] = self.y_train[val_idx] - y_pred_val

        np.testing.assert_allclose(
            uq_cv.oob_residuals,
            manual_residuals,
            atol=1e-5,
            err_msg="CV residuals do not match manual fold fitting!",
        )

        # Confirm CV residuals differ from OOB residuals (genuine out-of-fold predictions)
        rf_oob = RandomForestRegressor(n_estimators=10, random_state=42, bootstrap=True, oob_score=True)
        rf_oob.fit(self.X_train, self.y_train)
        uq_oob = GPUProximityRegressionUQ(
            rf_oob, self.X_train, self.y_train, device="cpu", residual_mode="oob"
        )
        self.assertFalse(np.allclose(uq_cv.oob_residuals, uq_oob.oob_residuals))

    def test_cv_small_n_fallback(self):
        """
        Verify that for N < 5 (e.g. N=3 or 4), residual_mode='cv' safely falls back to OOB residuals
        without throwing an exception.
        """
        X_small = self.X_train[:4]
        y_small = self.y_train[:4]

        rf_small_cv = RandomForestRegressor(n_estimators=10, random_state=42, bootstrap=True, oob_score=True)
        rf_small_cv.fit(X_small, y_small)
        rf_small_oob = RandomForestRegressor(n_estimators=10, random_state=42, bootstrap=True, oob_score=True)
        rf_small_oob.fit(X_small, y_small)

        # Should warn and fallback cleanly without exception
        with pytest.warns(UserWarning, match=r"smaller than cv_folds"):
            uq_cv = GPUProximityRegressionUQ(
                rf_small_cv, X_small, y_small, device="cpu", residual_mode="cv", cv_folds=5
            )

        uq_oob = GPUProximityRegressionUQ(
            rf_small_oob, X_small, y_small, device="cpu", residual_mode="oob"
        )

        np.testing.assert_allclose(uq_cv.oob_residuals, uq_oob.oob_residuals, atol=1e-6)
        np.testing.assert_array_equal(uq_cv.valid_oob_mask, uq_oob.valid_oob_mask)

    def test_cv_seed_reproducibility(self):
        """
        Verify deterministic fold splitting with model random_state.
        """
        rf1 = RandomForestRegressor(n_estimators=10, random_state=123, bootstrap=True, oob_score=True)
        rf1.fit(self.X_train, self.y_train)
        rf2 = RandomForestRegressor(n_estimators=10, random_state=123, bootstrap=True, oob_score=True)
        rf2.fit(self.X_train, self.y_train)
        rf_diff = RandomForestRegressor(n_estimators=10, random_state=999, bootstrap=True, oob_score=True)
        rf_diff.fit(self.X_train, self.y_train)

        uq1 = GPUProximityRegressionUQ(
            rf1, self.X_train, self.y_train, device="cpu", residual_mode="cv", cv_folds=5
        )
        uq2 = GPUProximityRegressionUQ(
            rf2, self.X_train, self.y_train, device="cpu", residual_mode="cv", cv_folds=5
        )
        uq_diff = GPUProximityRegressionUQ(
            rf_diff, self.X_train, self.y_train, device="cpu", residual_mode="cv", cv_folds=5
        )

        # Same random_state produces identical CV residuals
        np.testing.assert_allclose(uq1.oob_residuals, uq2.oob_residuals, atol=1e-6)

        # Different random_state produces different CV residuals
        self.assertFalse(np.allclose(uq1.oob_residuals, uq_diff.oob_residuals))

    def test_cv_prediction_intervals(self):
        """
        Verify predict_with_intervals succeeds and returns valid lower/upper bounds
        and local MAE using CV residuals.
        """
        rf = RandomForestRegressor(n_estimators=10, random_state=42, bootstrap=True, oob_score=True)
        rf.fit(self.X_train, self.y_train)
        uq = GPUProximityRegressionUQ(
            rf, self.X_train, self.y_train, device="cpu", residual_mode="cv", cv_folds=5
        )

        preds = uq.predict_with_intervals(self.X_test, level=0.9, return_mae=True)
        self.assertEqual(len(preds), 4)
        y_lwr, y_pred, y_upr, mae = preds

        self.assertEqual(len(y_lwr), len(self.X_test))
        self.assertEqual(len(y_pred), len(self.X_test))
        self.assertEqual(len(y_upr), len(self.X_test))
        self.assertEqual(len(mae), len(self.X_test))

        self.assertTrue(np.all(np.isfinite(y_lwr)))
        self.assertTrue(np.all(np.isfinite(y_pred)))
        self.assertTrue(np.all(np.isfinite(y_upr)))
        self.assertTrue(np.all(np.isfinite(mae)))

        # Lower bound <= Prediction <= Upper bound
        self.assertTrue(np.all(y_lwr <= y_pred + 1e-6))
        self.assertTrue(np.all(y_pred <= y_upr + 1e-6))
        # MAE is non-negative
        self.assertTrue(np.all(mae >= 0.0))

    def test_cv_folds_validation(self):
        """
        Verify that cv_folds < 2 raises ValueError.
        """
        rf = RandomForestRegressor(n_estimators=10, random_state=42, bootstrap=True)
        with self.assertRaises(ValueError):
            GPUProximityRegressionUQ(rf, self.X_train, self.y_train, cv_folds=1)
        with self.assertRaises(ValueError):
            GPUProximityRegressionUQ(rf, self.X_train, self.y_train, cv_folds=0)

    def test_cv_integer_targets(self):
        """
        Verify that integer-typed y_train does not suffer integer truncation in residuals.
        """
        y_int = np.random.randint(0, 10, size=len(self.X_train))
        rf = RandomForestRegressor(n_estimators=10, random_state=42, bootstrap=True, oob_score=True)
        rf.fit(self.X_train, y_int)
        uq = GPUProximityRegressionUQ(
            rf, self.X_train, y_int, device="cpu", residual_mode="cv", cv_folds=5
        )
        self.assertEqual(uq.oob_residuals.dtype, np.float32)
        # Residues should have fractional parts and not be strictly integers
        self.assertFalse(np.all(np.equal(np.mod(uq.oob_residuals, 1), 0)))
