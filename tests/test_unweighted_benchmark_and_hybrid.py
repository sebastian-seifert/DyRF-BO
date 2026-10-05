import os
import sys
import unittest
import json
import numpy as np
from unittest.mock import patch
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_schema import ProximityConfig, BenchmarkMasterConfig, RFConfig
import Uncertainty_Quantification as uq
from Hybrid_Proximity_Epistemic_UQ import HybridProximityEpistemicUQ
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ


class TestUnweightedBenchmarkAndHybrid(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_train = np.array([
            [0.0], [0.2], [0.4], [1.0], [1.2], [2.0], [3.0], [3.5], [4.0], [4.2]
        ])
        self.y_train = np.sin(self.X_train.ravel())
        self.rf = RandomForestRegressor(n_estimators=10, min_samples_leaf=1, oob_score=True, random_state=42)
        self.rf.fit(self.X_train, self.y_train)
        self.X_test = np.array([[-0.5], [0.1], [1.1], [2.5], [5.0]])

    def test_config_schema_proximity_weighted(self):
        """Test ProximityConfig defaults to weighted=False and serializes correctly."""
        prox_cfg = ProximityConfig()
        self.assertFalse(prox_cfg.weighted)

        master_cfg = BenchmarkMasterConfig()
        self.assertFalse(master_cfg.proximity.weighted)

        data_dict = master_cfg.to_dict()
        self.assertIn("weighted", data_dict["proximity"])
        self.assertFalse(data_dict["proximity"]["weighted"])

        json_str = master_cfg.to_json()
        parsed = json.loads(json_str)
        self.assertFalse(parsed["proximity"]["weighted"])

        # Test from_dict with weighted=True
        data_dict["proximity"]["weighted"] = True
        reconstructed = BenchmarkMasterConfig.from_dict(data_dict)
        self.assertTrue(reconstructed.proximity.weighted)

        # Test sweep task line generation
        lines_unweighted = master_cfg.generate_sweep_task_lines("ackley_1d")
        prox_line_unweighted = [l for l in lines_unweighted if "--approaches Proximity" in l or "Proximity_Baseline" in l][0]
        self.assertNotIn("--weighted_proximity", prox_line_unweighted)

        reconstructed_lines = reconstructed.generate_sweep_task_lines("ackley_1d")
        prox_line_weighted = [l for l in reconstructed_lines if "--approaches Proximity" in l or "Proximity_Baseline" in l][0]
        self.assertIn("--weighted_proximity", prox_line_weighted)

    def test_cli_parser_weighted_proximity(self):
        """Test Uncertainty_Quantification CLI parser recognizes --weighted_proximity."""
        # Default run: weighted_proximity should be False
        args_default, cfg_default = uq.parse_args_with_config([])
        self.assertFalse(args_default.weighted_proximity)
        self.assertFalse(cfg_default.proximity.weighted)

        # Explicit flag: weighted_proximity should be True
        args_weighted, cfg_weighted = uq.parse_args_with_config(["--weighted_proximity"])
        self.assertTrue(args_weighted.weighted_proximity)
        self.assertTrue(cfg_weighted.proximity.weighted)

    def test_hybrid_proximity_defaults_and_weighted(self):
        """Test HybridProximityEpistemicUQ initializes prox_model with correct weighting."""
        # Default initialization (weighted=False)
        hybrid_default = HybridProximityEpistemicUQ(
            self.rf, self.X_train, self.y_train, base_epistemic_method="likelihood"
        )
        self.assertFalse(hybrid_default.weighted)
        self.assertEqual(hybrid_default.weighting, "unweighted_all")
        self.assertIsNotNone(hybrid_default.prox_model)
        self.assertFalse(hybrid_default.prox_model.weighted)
        self.assertEqual(hybrid_default.prox_model.weighting, "unweighted_all")

        # Weighted initialization (weighted=True)
        hybrid_weighted = HybridProximityEpistemicUQ(
            self.rf, self.X_train, self.y_train, base_epistemic_method="likelihood", weighted=True
        )
        self.assertTrue(hybrid_weighted.weighted)
        self.assertEqual(hybrid_weighted.weighting, "leaf_normalized")
        self.assertIsNotNone(hybrid_weighted.prox_model)
        self.assertTrue(hybrid_weighted.prox_model.weighted)
        self.assertEqual(hybrid_weighted.prox_model.weighting, "leaf_normalized")

    def test_proximity_extraction_matches_unweighted_all(self):
        """Test proximity extraction under default configuration matches GPUProximityRegressionUQ(weighting='unweighted_all')."""
        hybrid_default = HybridProximityEpistemicUQ(
            self.rf, self.X_train, self.y_train, base_epistemic_method="likelihood"
        )
        ref_unweighted = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train,
            device="auto", batch_size="auto",
            use_density_scaling=False,
            topological_decay_lambda=1.0,
            normalize_by_depth=False,
            weighting="unweighted_all",
            weighted=False
        )

        p_hybrid = hybrid_default.prox_model.compute_proximity_matrix(self.X_test)
        p_ref = ref_unweighted.compute_proximity_matrix(self.X_test)
        if hasattr(p_hybrid, "get"):
            p_hybrid = p_hybrid.get()
        if hasattr(p_ref, "get"):
            p_ref = p_ref.get()

        np.testing.assert_allclose(p_hybrid, p_ref, rtol=1e-5, atol=1e-6)

        # Also verify that weighted prox_model matches ref_weighted
        hybrid_weighted = HybridProximityEpistemicUQ(
            self.rf, self.X_train, self.y_train, base_epistemic_method="likelihood", weighted=True
        )
        ref_weighted = GPUProximityRegressionUQ(
            self.rf, self.X_train, self.y_train,
            device="auto", batch_size="auto",
            use_density_scaling=False,
            topological_decay_lambda=1.0,
            normalize_by_depth=False,
            weighting="leaf_normalized",
            weighted=True
        )
        p_hybrid_w = hybrid_weighted.prox_model.compute_proximity_matrix(self.X_test)
        p_ref_w = ref_weighted.compute_proximity_matrix(self.X_test)
        if hasattr(p_hybrid_w, "get"):
            p_hybrid_w = p_hybrid_w.get()
        if hasattr(p_ref_w, "get"):
            p_ref_w = p_ref_w.get()

        np.testing.assert_allclose(p_hybrid_w, p_ref_w, rtol=1e-5, atol=1e-6)

    def test_uq_evaluation_passes_weighting_to_prox_models(self):
        """Test that Uncertainty_Quantification evaluation passes unweighted/weighted configuration correctly."""
        approaches = [
            "Proximity_Baseline",
            "Proximity_Method_B",
            "Proximity_Method_C",
            "Proximity_Method_B_C",
            "Proximity_Method_B_Norm",
            "Proximity_Method_C_Norm",
            "Proximity_Method_B_C_Norm",
            "Proximity_Auto_Lambda"
        ]

        def mock_compute_uq(self, X_test, **kwargs):
            return np.zeros(len(X_test), dtype=np.float32)

        def mock_tune_lambda_oob(self, **kwargs):
            return 1.0

        # Test default (unweighted)
        created_models = []

        def mock_init(self, *args, **kwargs):
            self.weighting = kwargs.get("weighting")
            self.weighted = kwargs.get("weighted")
            created_models.append((kwargs.get("weighting"), kwargs.get("weighted")))

        func_dict = uq.get_1d_functions()
        func_name = "sin"

        with patch.object(GPUProximityRegressionUQ, "__init__", mock_init), \
             patch.object(GPUProximityRegressionUQ, "compute_uq", mock_compute_uq), \
             patch.object(GPUProximityRegressionUQ, "tune_lambda_oob", mock_tune_lambda_oob):
            uq.args, _ = uq.parse_args_with_config([])
            uq.run_single_test(
                func_dict=func_dict,
                func_name=func_name,
                seed=1,
                approaches=approaches,
                rf_config=RFConfig(n_estimators=5, min_samples_leaf=1),
                k_neighbors=5,
                n_jobs=1
            )

        self.assertGreater(len(created_models), 0)
        for weighting, weighted in created_models:
            self.assertEqual(weighting, "unweighted_all")
            self.assertFalse(weighted)

        # Test with weighted_proximity=True
        created_models_weighted = []
        def mock_init_w(self, *args, **kwargs):
            self.weighting = kwargs.get("weighting")
            self.weighted = kwargs.get("weighted")
            created_models_weighted.append((kwargs.get("weighting"), kwargs.get("weighted")))

        with patch.object(GPUProximityRegressionUQ, "__init__", mock_init_w), \
             patch.object(GPUProximityRegressionUQ, "compute_uq", mock_compute_uq), \
             patch.object(GPUProximityRegressionUQ, "tune_lambda_oob", mock_tune_lambda_oob):
            uq.args, _ = uq.parse_args_with_config(["--weighted_proximity"])
            uq.run_single_test(
                func_dict=func_dict,
                func_name=func_name,
                seed=1,
                approaches=approaches,
                rf_config=RFConfig(n_estimators=5, min_samples_leaf=1),
                k_neighbors=5,
                n_jobs=1
            )

        self.assertGreater(len(created_models_weighted), 0)
        for weighting, weighted in created_models_weighted:
            self.assertEqual(weighting, "leaf_normalized")
            self.assertTrue(weighted)


if __name__ == "__main__":
    unittest.main()
