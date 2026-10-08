from __future__ import annotations
from pathlib import Path
import unittest
import numpy as np
import yaml
from ConfigSpace import ConfigurationSpace, Float

from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
from carps_integration.proximity_lcb import ProximityLowerBoundAcquisition


class TestProximityLCBCVIntegration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.cs = ConfigurationSpace(seed=42)
        self.cs.add([
            Float("x0", (-2.0, 2.0), default=0.0),
            Float("x1", (-2.0, 2.0), default=0.0),
        ])

        # Generate synthetic data
        self.X_all = np.random.uniform(-2.0, 2.0, size=(40, 2))
        self.y_all = (
            np.sin(self.X_all[:, 0])
            + 0.5 * self.X_all[:, 1] ** 2
            + np.random.normal(0, 0.05, size=40)
        )
        self.X_eval = np.random.uniform(-2.0, 2.0, size=(15, 2))

    def test_yaml_config_syntax_and_structure(self):
        """
        Verify smac20_proximity_lcb_cv.yaml exists, loads cleanly,
        and matches CARP-S optimizer schema.
        """
        yaml_path = Path("carps_integration/configs/optimizer/smac20_proximity_lcb_cv.yaml")
        self.assertTrue(yaml_path.exists(), f"Missing config file: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(cfg["optimizer_id"], "SMAC20_ProximityLCB_CV")
        self.assertEqual(cfg["optimizer_container_id"], "SMAC20_ProximityLCB_CV")

        optimizer_section = cfg["optimizer"]
        self.assertEqual(optimizer_section["acq_func_name"], "proximity_lcb")
        self.assertEqual(
            optimizer_section["acq_func_kwargs"],
            {"k": 25, "level": 0.95, "eps": 0.16, "k_warmup": 25},
        )

        smac_cfg = optimizer_section["smac_cfg"]
        self.assertEqual(
            smac_cfg["model_class"],
            "carps_integration.custom_uncertainty_model.CustomUncertaintyRandomForest",
        )
        self.assertEqual(
            smac_cfg["model_kwargs"]["uncertainty_func"],
            "standard_proximity_cv",
        )

    def test_acquisition_warmup_phase1_and_phase2_transition(self):
        """
        Verify transition between warmup (Phase 1) and proximity intervals (Phase 2):
        - When N <= k_warmup (e.g. N=10, k=25): uses standard RF LCB (predict_standard_rf).
        - When N > k_warmup (e.g. N=30, k=25): uses predict_with_intervals on CustomUncertaintyRandomForest
          configured with standard_proximity_cv.
        """
        # Phase 1: N = 10 <= 25
        X_10 = self.X_all[:10]
        y_10 = self.y_all[:10]
        model_10 = CustomUncertaintyRandomForest(
            uncertainty_func="standard_proximity_cv",
            configspace=self.cs,
            n_trees=10,
            seed=42,
        )
        model_10.train(X_10, y_10)

        acq = ProximityLowerBoundAcquisition(k=25, k_warmup=25, eps=0.16, level=0.95)
        acq.update(model=model_10, num_data=10)

        scores_warm = acq._compute(self.X_eval)
        self.assertEqual(scores_warm.shape, (len(self.X_eval), 1))
        self.assertTrue(np.all(np.isfinite(scores_warm)))

        # Also verify SMAC3 Configuration object interface via __call__
        from ConfigSpace import Configuration
        cfg_objs = [Configuration(self.cs, values={"x0": float(x[0]), "x1": float(x[1])}) for x in self.X_eval[:3]]
        cfg_scores = acq(cfg_objs)
        self.assertEqual(cfg_scores.shape, (3, 1))
        self.assertTrue(np.all(np.isfinite(cfg_scores)))
        # Verify __call__ matches _compute on the extracted config vectors
        X_from_cfgs = np.array([c.get_array() for c in cfg_objs], dtype=np.float64)
        np.testing.assert_allclose(cfg_scores, acq._compute(X_from_cfgs), atol=1e-6)

        # Verify manual standard RF LCB formula: -mean_rf + kappa * std_rf
        mean_rf, var_rf = model_10.predict_standard_rf(self.X_eval)
        mean_rf = np.asarray(mean_rf, dtype=np.float64).flatten()
        std_rf = np.sqrt(np.maximum(np.asarray(var_rf, dtype=np.float64).flatten(), 1e-10))
        expected_warm = (-mean_rf + acq._kappa * std_rf).reshape(-1, 1)
        np.testing.assert_allclose(scores_warm, expected_warm, atol=1e-6)

        # Phase 2: N = 30 > 25
        X_30 = self.X_all[:30]
        y_30 = self.y_all[:30]
        model_30 = CustomUncertaintyRandomForest(
            uncertainty_func="standard_proximity_cv",
            configspace=self.cs,
            n_trees=10,
            seed=42,
        )
        model_30.train(X_30, y_30)

        acq.update(model=model_30, num_data=30)
        scores_post = acq._compute(self.X_eval)
        self.assertEqual(scores_post.shape, (len(self.X_eval), 1))
        self.assertTrue(np.all(np.isfinite(scores_post)))

        # In phase 2, scores must reflect predict_with_intervals
        res = model_30.predict_with_intervals(
            self.X_eval,
            n_neighbors=25,
            level=0.95,
            return_mae=True,
            weighting=acq._weighting,
            weighted=acq._weighted,
        )
        y_lwr, y_pred, _, local_mae = res
        delta_floor = 0.16 * acq._kappa * local_mae
        floored_lwr = np.minimum(y_lwr, y_pred - delta_floor)
        expected_post = (-floored_lwr).reshape(-1, 1)
        np.testing.assert_allclose(scores_post, expected_post, atol=1e-5)

        # Confirm post-warmup scores differ from standard RF LCB formula
        mean_rf30, var_rf30 = model_30.predict_standard_rf(self.X_eval)
        std_rf30 = np.sqrt(np.maximum(var_rf30.flatten(), 1e-10))
        std_lcb_30 = (-mean_rf30.flatten() + acq._kappa * std_rf30).reshape(-1, 1)
        self.assertFalse(np.allclose(scores_post, std_lcb_30, atol=1e-3))

    def test_toggle_oob_vs_cv_acquisition_parity(self):
        """
        Verify both standard_proximity (OOB) and standard_proximity_cv (CV) produce
        finite, valid acquisition scores, while reflecting their respective residual sources.
        """
        X_train = self.X_all[:30]
        y_train = self.y_all[:30]

        model_oob = CustomUncertaintyRandomForest(
            uncertainty_func="standard_proximity",
            configspace=self.cs,
            n_trees=10,
            seed=42,
        )
        model_oob.train(X_train, y_train)

        model_cv = CustomUncertaintyRandomForest(
            uncertainty_func="standard_proximity_cv",
            configspace=self.cs,
            n_trees=10,
            seed=42,
        )
        model_cv.train(X_train, y_train)

        self.assertEqual(model_oob.uq_extractor.residual_mode, "oob")
        self.assertEqual(model_cv.uq_extractor.residual_mode, "cv")

        acq_oob = ProximityLowerBoundAcquisition(k=25, k_warmup=25, eps=0.16, level=0.95)
        acq_oob.update(model=model_oob, num_data=30)
        scores_oob = acq_oob._compute(self.X_eval)

        acq_cv = ProximityLowerBoundAcquisition(k=25, k_warmup=25, eps=0.16, level=0.95)
        acq_cv.update(model=model_cv, num_data=30)
        scores_cv = acq_cv._compute(self.X_eval)

        # Both must be valid and finite
        self.assertEqual(scores_oob.shape, (len(self.X_eval), 1))
        self.assertEqual(scores_cv.shape, (len(self.X_eval), 1))
        self.assertTrue(np.all(np.isfinite(scores_oob)))
        self.assertTrue(np.all(np.isfinite(scores_cv)))

        # Residuals in uq_model differ between OOB and 5-Fold CV
        oob_resids = model_oob.uq_extractor.uq_model.oob_residuals
        cv_resids = model_cv.uq_extractor.uq_model.oob_residuals
        self.assertFalse(np.allclose(oob_resids, cv_resids, atol=1e-4))

    def test_closed_loop_bo_stepping(self):
        """
        Verify multi-step closed-loop Bayesian optimization loop using
        CustomUncertaintyRandomForest('standard_proximity_cv') and ProximityLowerBoundAcquisition
        stepping across warmup boundary (N=24 to N=27).
        """
        def objective(x: np.ndarray) -> float:
            return float((x[0] - 0.2) ** 2 + (x[1] + 0.3) ** 2 + 0.1 * np.sin(5.0 * x[0]))

        # Start with N=24 points (just below k_warmup=25)
        X_train = list(self.X_all[:24])
        y_train = [objective(x) for x in X_train]

        acq = ProximityLowerBoundAcquisition(k=25, k_warmup=25, eps=0.16, level=0.95)
        candidate_pool = np.random.uniform(-2.0, 2.0, size=(30, 2))

        n_steps = 3
        for step in range(n_steps):
            current_N = len(X_train)
            model = CustomUncertaintyRandomForest(
                uncertainty_func="standard_proximity_cv",
                configspace=self.cs,
                n_trees=10,
                seed=42 + step,
            )
            model.train(np.array(X_train), np.array(y_train))
            acq.update(model=model, num_data=current_N)

            # Evaluate acquisition on candidates
            acq_vals = acq._compute(candidate_pool).flatten()
            self.assertEqual(len(acq_vals), len(candidate_pool))
            self.assertTrue(np.all(np.isfinite(acq_vals)))

            # Select candidate with highest acquisition value (argmax)
            best_idx = int(np.argmax(acq_vals))
            next_x = candidate_pool[best_idx]
            next_y = objective(next_x)

            X_train.append(next_x)
            y_train.append(next_y)

        self.assertEqual(len(X_train), 24 + n_steps)
        self.assertEqual(len(y_train), 24 + n_steps)
        self.assertTrue(np.all(np.isfinite(np.array(y_train))))


if __name__ == "__main__":
    unittest.main()
