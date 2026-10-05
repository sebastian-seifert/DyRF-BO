from __future__ import annotations
import os
import unittest
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from ConfigSpace import ConfigurationSpace, Float

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
from ep_extractors import UQExtractorRegistry
from ep_extractors.proximity_b import ProximityBExtractor
from ep_extractors.standard_proximity import StandardProximityExtractor
from carps_integration.proximity_lcb import ProximityLowerBoundAcquisition
from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest


class TestUnweightedProximityCore(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Synthetic dataset with 40 samples, 3 features
        self.X_train = np.random.uniform(-2.0, 2.0, size=(40, 3))
        self.y_train = np.sin(self.X_train[:, 0]) + 0.5 * self.X_train[:, 1] ** 2 + np.random.normal(0, 0.05, size=40)
        self.X_test = np.random.uniform(-2.0, 2.0, size=(15, 3))

        # Train a deterministic Random Forest
        self.rf = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42, bootstrap=True)
        self.rf.fit(self.X_train, self.y_train)

    def test_default_initialization_is_unweighted(self):
        """
        Verify that default GPUProximityRegressionUQ defaults to unweighted ('unweighted_all'),
        produces a strictly symmetric proximity matrix P = P^T, and self-identity diag(P) == 1.0.
        Also check with topological tree path decay.
        """
        # Standard binary proximity (no decay)
        uq_default = GPUProximityRegressionUQ(self.rf, self.X_train, self.y_train, device="cpu")
        self.assertEqual(uq_default.weighting, "unweighted_all")

        P_default = uq_default.compute_proximity_matrix(self.X_train)
        n_samples = len(self.X_train)
        self.assertEqual(P_default.shape, (n_samples, n_samples))
        np.testing.assert_allclose(P_default, P_default.T, atol=1e-6, err_msg="Default proximity matrix is not symmetric!")
        np.testing.assert_allclose(np.diag(P_default), 1.0, atol=1e-6, err_msg="Default self-proximity is not 1.0!")

        # Topological proximity
        uq_topo_default = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train, device="cpu", topological_decay_lambda=1.2
        )
        self.assertEqual(uq_topo_default.weighting, "unweighted_all")
        P_topo = uq_topo_default.compute_proximity_matrix(self.X_train)
        np.testing.assert_allclose(P_topo, P_topo.T, atol=1e-6, err_msg="Topological P(X, X) not symmetric!")
        np.testing.assert_allclose(np.diag(P_topo), 1.0, atol=1e-6, err_msg="Topological self-proximity not 1.0!")

    def test_explicit_weighted_flags_produce_legacy_weighted(self):
        """
        Verify that explicit flag weighted=True (or use_leaf_weights=True, or weighting='leaf_normalized')
        produces the legacy weighted matrix (scaled by leaf weights) across all methods.
        """
        uq_weighted = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train, device="cpu", weighted=True
        )
        uq_leaf = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train, device="cpu", use_leaf_weights=True
        )
        uq_norm = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train, device="cpu", weighting="leaf_normalized"
        )

        for uq in (uq_weighted, uq_leaf, uq_norm):
            self.assertEqual(uq.weighting, "leaf_normalized")

        # Proximity matrix on X_test matches across all three
        p_mat_weighted = uq_weighted.compute_proximity_matrix(self.X_test)
        p_mat_leaf = uq_leaf.compute_proximity_matrix(self.X_test)
        p_mat_norm = uq_norm.compute_proximity_matrix(self.X_test)
        np.testing.assert_allclose(p_mat_weighted, p_mat_leaf, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(p_mat_weighted, p_mat_norm, rtol=1e-7, atol=1e-7)

        # compute_uq matches across all three
        uq_sig_weighted = uq_weighted.compute_uq(self.X_test)
        uq_sig_leaf = uq_leaf.compute_uq(self.X_test)
        uq_sig_norm = uq_norm.compute_uq(self.X_test)
        np.testing.assert_allclose(uq_sig_weighted, uq_sig_leaf, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(uq_sig_weighted, uq_sig_norm, rtol=1e-7, atol=1e-7)

        # predict_with_intervals matches across all three
        pred_weighted = uq_weighted.predict_with_intervals(self.X_test, return_mae=True)
        pred_leaf = uq_leaf.predict_with_intervals(self.X_test, return_mae=True)
        pred_norm = uq_norm.predict_with_intervals(self.X_test, return_mae=True)
        for w, l, m in zip(pred_weighted, pred_leaf, pred_norm):
            np.testing.assert_allclose(w, l, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(w, m, rtol=1e-7, atol=1e-7)

        # Confirm that the weighted matrix is scaled by leaf weights (not symmetric, diag != 1.0)
        P_train_weighted = uq_weighted.compute_proximity_matrix(self.X_train)
        self.assertFalse(np.allclose(np.diag(P_train_weighted), 1.0, atol=1e-3))
        self.assertFalse(np.allclose(P_train_weighted, P_train_weighted.T, atol=1e-3))

    def test_option_a_strictly_symmetric_and_self_identity_binary(self):
        """
        Verify Option A (weighting='unweighted_all' or use_leaf_weights=False):
        Standard binary proximity matrix P(X, X) is strictly symmetric and P(x_i, x_i) == 1.0.
        """
        for kwargs in [
            {"weighting": "unweighted_all"},
            {"use_leaf_weights": False},
            {"weighting": "unweighted"},
        ]:
            uq = GPUProximityRegressionUQ(self.rf, self.X_train, self.y_train, device="cpu", **kwargs)
            self.assertEqual(uq.weighting, "unweighted_all")

            # Evaluate proximity on training data itself
            P = uq.compute_proximity_matrix(self.X_train)
            n_samples = len(self.X_train)
            self.assertEqual(P.shape, (n_samples, n_samples))

            # Strictly symmetric within 1e-6
            np.testing.assert_allclose(P, P.T, atol=1e-6, err_msg="Proximity matrix is not symmetric!")

            # Strictly self-identity P(x_i, x_i) == 1.0
            np.testing.assert_allclose(np.diag(P), 1.0, atol=1e-6, err_msg="Self-proximity is not 1.0!")

            # Values must be bounded in [0.0, 1.0]
            self.assertTrue(np.all(P >= -1e-6))
            self.assertTrue(np.all(P <= 1.0 + 1e-6))

    def test_option_a_strictly_symmetric_and_self_identity_topological(self):
        """
        Verify Option A with topological tree path decay:
        P(X, X) remains strictly symmetric and self-identity P(x_i, x_i) == 1.0.
        """
        for normalize_by_depth in (False, True):
            uq = GPUProximityRegressionUQ(
                self.rf,
                self.X_train,
                self.y_train,
                device="cpu",
                topological_decay_lambda=1.5,
                normalize_by_depth=normalize_by_depth,
                weighting="unweighted_all",
            )
            P = uq.compute_proximity_matrix(self.X_train)

            # Strictly symmetric
            np.testing.assert_allclose(P, P.T, atol=1e-6, err_msg=f"Topological P(X, X) not symmetric (norm_depth={normalize_by_depth})")

            # Strictly self-identity P(x_i, x_i) == 1.0 (d(x_i, x_i) = 0, exp(-lambda * 0) = 1)
            np.testing.assert_allclose(np.diag(P), 1.0, atol=1e-6, err_msg=f"Topological self-proximity not 1.0 (norm_depth={normalize_by_depth})")

    def test_option_a_contrast_with_leaf_normalized(self):
        """
        Demonstrate that legacy leaf_normalized is NOT symmetric and NOT self-identity 1.0,
        confirming that Option A provides the intended structural fix.
        """
        uq_legacy = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train, device="cpu", weighting="leaf_normalized"
        )
        P_legacy = uq_legacy.compute_proximity_matrix(self.X_train)

        # Legacy leaf_normalized is divided by leaf sizes of the training points, so diag is < 1.0
        self.assertFalse(np.allclose(np.diag(P_legacy), 1.0, atol=1e-3))
        # Legacy is not symmetric
        self.assertFalse(np.allclose(P_legacy, P_legacy.T, atol=1e-3))

    def test_option_b_unweighted_inbag_behavior(self):
        """
        Verify Option B (weighting='unweighted_inbag'):
        Only in-bag trees contribute, unweighted by leaf counts.
        """
        uq_inbag = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train, device="cpu", weighting="unweighted_inbag"
        )
        self.assertEqual(uq_inbag.weighting, "unweighted_inbag")

        P_inbag = uq_inbag.compute_proximity_matrix(self.X_train)

        # In-bag self identity: for tree t, x_i contributes 1 iff x_i is in-bag for tree t.
        # Thus, P(x_i, x_i) = (sum of in-bag trees for point i) / n_estimators.
        expected_diag = np.mean(uq_inbag.in_bag_indices, axis=1)
        np.testing.assert_allclose(np.diag(P_inbag), expected_diag, atol=1e-6)

        # Verify that for any test sample x, a training sample j only has non-zero contribution
        # from tree t if in_bag_indices[j, t] == 1
        X_single = self.X_test[:2]
        pred_lwr, pred_mid, pred_upr, mae = uq_inbag.predict_with_intervals(X_single, return_mae=True)
        self.assertEqual(len(pred_lwr), 2)
        self.assertEqual(len(pred_mid), 2)
        self.assertTrue(np.all(pred_lwr <= pred_upr))

    def test_runtime_weighting_and_weighted_override(self):
        """
        Verify that compute_proximity_matrix, compute_uq, and predict_with_intervals
        default to unweighted and accept weighted: bool | None = None override.
        """
        uq = GPUProximityRegressionUQ(self.rf, self.X_train, self.y_train, device="cpu")
        self.assertEqual(uq.weighting, "unweighted_all")

        uq_unweighted = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train, device="cpu", weighting="unweighted_all"
        )
        uq_weighted = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train, device="cpu", weighted=True
        )

        # 1. compute_proximity_matrix defaults to unweighted
        p_default = uq.compute_proximity_matrix(self.X_test)
        p_unweighted = uq_unweighted.compute_proximity_matrix(self.X_test)
        p_weighted = uq_weighted.compute_proximity_matrix(self.X_test)
        np.testing.assert_allclose(p_default, p_unweighted, rtol=1e-7, atol=1e-7)
        self.assertFalse(np.allclose(p_default, p_weighted, atol=1e-3))

        # compute_proximity_matrix with weighted=True override
        p_override_weighted = uq.compute_proximity_matrix(self.X_test, weighted=True)
        np.testing.assert_allclose(p_override_weighted, p_weighted, rtol=1e-7, atol=1e-7)

        # compute_proximity_matrix with weighted=False override
        p_override_unweighted = uq.compute_proximity_matrix(self.X_test, weighted=False)
        np.testing.assert_allclose(p_override_unweighted, p_unweighted, rtol=1e-7, atol=1e-7)

        # 2. compute_uq defaults to unweighted
        uq_sig_default = uq.compute_uq(self.X_test)
        uq_sig_unweighted = uq_unweighted.compute_uq(self.X_test)
        uq_sig_weighted = uq_weighted.compute_uq(self.X_test)
        np.testing.assert_allclose(uq_sig_default, uq_sig_unweighted, rtol=1e-7, atol=1e-7)

        uq_sig_override = uq.compute_uq(self.X_test, weighted=True)
        np.testing.assert_allclose(uq_sig_override, uq_sig_weighted, rtol=1e-7, atol=1e-7)

        # 3. predict_with_intervals defaults to unweighted
        pred_default = uq.predict_with_intervals(self.X_test, return_mae=True)
        pred_unweighted = uq_unweighted.predict_with_intervals(self.X_test, return_mae=True)
        pred_weighted = uq_weighted.predict_with_intervals(self.X_test, return_mae=True)
        for d, u in zip(pred_default, pred_unweighted):
            np.testing.assert_allclose(d, u, rtol=1e-7, atol=1e-7)

        pred_override = uq.predict_with_intervals(self.X_test, return_mae=True, weighted=True)
        for o, w in zip(pred_override, pred_weighted):
            np.testing.assert_allclose(o, w, rtol=1e-7, atol=1e-7)

        pred_override_unweighted = uq.predict_with_intervals(self.X_test, return_mae=True, weighted=False)
        for o, u in zip(pred_override_unweighted, pred_unweighted):
            np.testing.assert_allclose(o, u, rtol=1e-7, atol=1e-7)

        # Also verify legacy weighting and use_leaf_weights overrides still work
        res_override = uq.predict_with_intervals(self.X_test, weighting="leaf_normalized")
        for a, b in zip(res_override, pred_weighted):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

        res_override_bool = uq.predict_with_intervals(self.X_test, use_leaf_weights=True)
        for a, b in zip(res_override_bool, pred_weighted):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_extractor_integration_proximity_b_and_standard(self):
        """
        Verify ProximityBExtractor and StandardProximityExtractor:
        - accept weighting and use_leaf_weights
        - register 'proximity_b_unweighted' and 'standard_proximity_unweighted'
        """
        # 1. ProximityBExtractor parameter forwarding
        ext_b_unweighted = ProximityBExtractor(
            self.rf, device="cpu", decay_lambda=1.2, weighting="unweighted_all"
        )
        ext_b_unweighted.fit(self.X_train, self.y_train)
        self.assertEqual(ext_b_unweighted.uq_model.weighting, "unweighted_all")

        ext_b_bool = ProximityBExtractor(
            self.rf, device="cpu", decay_lambda=1.2, use_leaf_weights=False
        )
        ext_b_bool.fit(self.X_train, self.y_train)
        self.assertEqual(ext_b_bool.uq_model.weighting, "unweighted_all")

        sig_b = ext_b_unweighted.extract_epistemic_signal(self.X_test)
        self.assertEqual(len(sig_b), len(self.X_test))

        # 2. StandardProximityExtractor parameter forwarding
        ext_std_unweighted = StandardProximityExtractor(
            self.rf, device="cpu", weighting="unweighted_all"
        )
        ext_std_unweighted.fit(self.X_train, self.y_train)
        self.assertEqual(ext_std_unweighted.uq_model.weighting, "unweighted_all")

        ext_std_bool = StandardProximityExtractor(
            self.rf, device="cpu", use_leaf_weights=False
        )
        ext_std_bool.fit(self.X_train, self.y_train)
        self.assertEqual(ext_std_bool.uq_model.weighting, "unweighted_all")

        sig_std = ext_std_unweighted.extract_epistemic_signal(self.X_test)
        self.assertEqual(len(sig_std), len(self.X_test))

        # 3. Registry checks
        self.assertIn("proximity_b_unweighted", UQExtractorRegistry._registry)
        self.assertIn("standard_proximity_unweighted", UQExtractorRegistry._registry)

        reg_b_unweighted = UQExtractorRegistry.get("proximity_b_unweighted", self.rf, device="cpu")
        reg_b_unweighted.fit(self.X_train, self.y_train)
        self.assertEqual(reg_b_unweighted.uq_model.weighting, "unweighted_all")

        reg_std_unweighted = UQExtractorRegistry.get("standard_proximity_unweighted", self.rf, device="cpu")
        reg_std_unweighted.fit(self.X_train, self.y_train)
        self.assertEqual(reg_std_unweighted.uq_model.weighting, "unweighted_all")

    def test_proximity_lower_bound_acq_unweighted_integration(self):
        """
        Verify ProximityLowerBoundAcquisition accepts weighting='unweighted' and use_leaf_weights=False,
        and forwards weighting to predict_with_intervals.
        """
        # Case 1: weighting="unweighted"
        acq1 = ProximityLowerBoundAcquisition(eps=0.15, level=0.95, k=10, weighting="unweighted")
        self.assertEqual(acq1._weighting, "unweighted")
        self.assertEqual(acq1.meta["weighting"], "unweighted")

        # Case 2: use_leaf_weights=False
        acq2 = ProximityLowerBoundAcquisition(eps=0.15, level=0.95, k=10, use_leaf_weights=False)
        self.assertEqual(acq2._weighting, "unweighted")

        # Case 3: Default is "unweighted"
        acq_def = ProximityLowerBoundAcquisition(eps=0.15, level=0.95, k=10)
        self.assertEqual(acq_def._weighting, "unweighted")
        self.assertFalse(acq_def._weighted)
        self.assertEqual(acq_def.meta["weighting"], "unweighted")
        self.assertFalse(acq_def.meta["weighted"])

        # Mock surrogate to verify arguments passed to predict_with_intervals
        class CallRecordingSurrogate:
            def __init__(self, n_train=50):
                self.n_train = n_train
                self.last_X = np.zeros((n_train, 3))
                self.received_kwargs = {}

            def predict_with_intervals(self, X, **kwargs):
                self.received_kwargs = kwargs
                n = len(X)
                return (
                    np.zeros(n),
                    np.zeros(n),
                    np.ones(n),
                    np.full(n, 0.5),
                )

        mock_model = CallRecordingSurrogate(n_train=50)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(Float("x0", (-2.0, 2.0), default=0.0))
        cs.add_hyperparameter(Float("x1", (-2.0, 2.0), default=0.0))
        cs.add_hyperparameter(Float("x2", (-2.0, 2.0), default=0.0))

        acq1.update(model=mock_model)
        X_eval = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        vals = acq1._compute(X_eval)
        self.assertEqual(vals.shape, (2, 1))
        self.assertIn("weighting", mock_model.received_kwargs)
        self.assertEqual(mock_model.received_kwargs["weighting"], "unweighted")

    def test_config_smac20_proximity_lcb_unweighted_exists_and_valid(self):
        """
        Verify carps_integration/configs/optimizer/smac20_proximity_lcb_unweighted.yaml
        exists, is valid YAML, and has expected acquisition and model kwargs.
        """
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "carps_integration",
            "configs",
            "optimizer",
            "smac20_proximity_lcb_unweighted.yaml",
        )
        self.assertTrue(os.path.exists(config_path), f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.assertIn("optimizer", cfg)
        opt = cfg["optimizer"]
        self.assertEqual(opt["acq_func_name"], "proximity_lcb")
        self.assertEqual(opt["acq_func_kwargs"]["weighting"], "unweighted")

        extractor_kwargs = opt["smac_cfg"]["model_kwargs"]["extractor_kwargs"]
        self.assertEqual(extractor_kwargs["weighting"], "unweighted_all")

    def test_resolve_runtime_weighting_validation(self):
        """
        Verify that _resolve_runtime_weighting defaults to self.weighting ('unweighted_all'),
        handles weighted: bool | None = None, validates the weighting string against known
        options and raises ValueError for unknown weighting schemes.
        """
        uq = GPUProximityRegressionUQ(self.rf, self.X_train, self.y_train, device="cpu")
        self.assertEqual(uq.weighting, "unweighted_all")

        # Default resolves to unweighted_all
        self.assertEqual(uq._resolve_runtime_weighting(), "unweighted_all")

        # weighted boolean overrides
        self.assertEqual(uq._resolve_runtime_weighting(weighted=True), "leaf_normalized")
        self.assertEqual(uq._resolve_runtime_weighting(weighted=False), "unweighted_all")

        # use_leaf_weights overrides
        self.assertEqual(uq._resolve_runtime_weighting(use_leaf_weights=True), "leaf_normalized")
        self.assertEqual(uq._resolve_runtime_weighting(use_leaf_weights=False), "unweighted_all")

        valid_options = {"leaf_normalized", "weighted", "unweighted_all", "unweighted", "unweighted_inbag"}
        for opt in valid_options:
            res = uq._resolve_runtime_weighting(weighting=opt)
            self.assertIn(res, {"leaf_normalized", "unweighted_all", "unweighted_inbag"})

        # Unknown options must raise ValueError
        for invalid_opt in ["unknown", "normalized", "raw", "custom_scheme", ""]:
            with self.assertRaises(ValueError) as ctx:
                uq._resolve_runtime_weighting(weighting=invalid_opt)
            self.assertIn("Unknown weighting scheme", str(ctx.exception))

    def test_custom_uncertainty_rf_proximity_b_unweighted_end_to_end(self):
        """
        End-to-end integration test with real CustomUncertaintyRandomForest
        configured with uncertainty_func='proximity_b_unweighted', fitting on dummy data
        and executing predict_with_intervals().
        """
        cs = ConfigurationSpace(seed=42)
        cs.add([
            Float("x0", (-2.0, 2.0), default=0.0),
            Float("x1", (-2.0, 2.0), default=0.0),
            Float("x2", (-2.0, 2.0), default=0.0),
        ])

        model = CustomUncertaintyRandomForest(
            uncertainty_func="proximity_b_unweighted",
            configspace=cs,
            n_trees=10,
            seed=42,
        )
        model.train(self.X_train, self.y_train)

        self.assertIsNotNone(model.uq_extractor)
        self.assertEqual(model.uq_extractor.weighting, "unweighted_all")

        # Test predict_with_intervals without MAE
        y_lwr, y_mean, y_upr = model.predict_with_intervals(self.X_test, return_mae=False)
        self.assertEqual(y_lwr.shape, (len(self.X_test),))
        self.assertEqual(y_mean.shape, (len(self.X_test),))
        self.assertEqual(y_upr.shape, (len(self.X_test),))
        self.assertTrue(np.all(np.isfinite(y_lwr)))
        self.assertTrue(np.all(np.isfinite(y_mean)))
        self.assertTrue(np.all(np.isfinite(y_upr)))
        self.assertTrue(np.all(y_upr >= y_lwr - 1e-7))

        # Test predict_with_intervals with MAE
        y_lwr_mae, y_mean_mae, y_upr_mae, local_mae = model.predict_with_intervals(self.X_test, return_mae=True)
        self.assertEqual(local_mae.shape, (len(self.X_test),))
        self.assertTrue(np.all(np.isfinite(local_mae)))
        self.assertTrue(np.all(local_mae > 0.0))
        np.testing.assert_allclose(y_lwr, y_lwr_mae, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(y_mean, y_mean_mae, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(y_upr, y_upr_mae, rtol=1e-7, atol=1e-7)

        # Test predict() uncertainty signal through SMAC interface
        mean, var = model.predict(self.X_test)
        self.assertEqual(mean.shape, (len(self.X_test), 1))
        self.assertEqual(var.shape, (len(self.X_test), 1))
        self.assertTrue(np.all(var > 0.0))


if __name__ == "__main__":
    unittest.main()

