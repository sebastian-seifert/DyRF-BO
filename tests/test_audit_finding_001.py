import os
import sys
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_schema import (
    BenchmarkMasterConfig, DataConfig, RFConfig, ExtractorConfig, ProximityConfig
)
from Uncertainty_Quantification import parse_args_with_config, run_single_test


class TestFinding001RFPresets(unittest.TestCase):
    """Tests that numeric and letter RF presets are resolved correctly and unknown presets rejected."""

    def test_numeric_presets_string_and_int(self):
        # Preset 1 / A: 100 trees, min_samples_leaf=5
        cfg_1_str = RFConfig.from_preset("1")
        self.assertEqual(cfg_1_str.n_estimators, 100)
        self.assertEqual(cfg_1_str.min_samples_leaf, 5)

        cfg_1_int = RFConfig.from_preset(1)
        self.assertEqual(cfg_1_int.n_estimators, 100)
        self.assertEqual(cfg_1_int.min_samples_leaf, 5)

        # Preset 2: 100 trees, min_samples_leaf=25
        cfg_2 = RFConfig.from_preset("2")
        self.assertEqual(cfg_2.n_estimators, 100)
        self.assertEqual(cfg_2.min_samples_leaf, 25)

        # Preset 3: 100 trees, min_samples_leaf=50
        cfg_3 = RFConfig.from_preset("3")
        self.assertEqual(cfg_3.n_estimators, 100)
        self.assertEqual(cfg_3.min_samples_leaf, 50)

        # Preset 4: 300 trees, min_samples_leaf=10
        cfg_4 = RFConfig.from_preset("4")
        self.assertEqual(cfg_4.n_estimators, 300)
        self.assertEqual(cfg_4.min_samples_leaf, 10)

        # Preset 5: 300 trees, min_samples_leaf=30
        cfg_5 = RFConfig.from_preset("5")
        self.assertEqual(cfg_5.n_estimators, 300)
        self.assertEqual(cfg_5.min_samples_leaf, 30)

    def test_letter_presets_case_insensitive(self):
        cfg_a = RFConfig.from_preset("a")
        self.assertEqual(cfg_a.n_estimators, 100)
        self.assertEqual(cfg_a.min_samples_leaf, 5)

        cfg_b = RFConfig.from_preset("B")
        self.assertEqual(cfg_b.n_estimators, 500)
        self.assertEqual(cfg_b.min_samples_leaf, 10)

        cfg_c = RFConfig.from_preset("c")
        self.assertEqual(cfg_c.n_estimators, 1000)
        self.assertEqual(cfg_c.min_samples_leaf, 25)

    def test_unknown_preset_raises_value_error(self):
        with self.assertRaises(ValueError):
            RFConfig.from_preset("unknown_preset")
        with self.assertRaises(ValueError):
            RFConfig.from_preset(99)

    def test_run_single_test_differentiates_presets_1_and_5(self):
        """Verify that run_single_test builds an RF with leaf=5 for preset '1' and leaf=30 for preset '5'."""
        func_dict = {
            "test_func": {
                "fn": lambda x: (x[:, 0]**2),
                "bounds": [(-2.0, 2.0)],
                "dim": 1
            }
        }

        with patch("Uncertainty_Quantification.RandomForestRegressor") as mock_rf_cls, \
             patch("Uncertainty_Quantification.generate_data") as mock_gen_data, \
             patch("Uncertainty_Quantification.LeafCache"), \
             patch("Uncertainty_Quantification.EpistemicQuantifier") as mock_quant_cls:

            import numpy as np
            mock_gen_data.return_value = (
                np.zeros((10, 1)), np.zeros(10),
                np.zeros((5, 1)), np.zeros(5), np.zeros(5)
            )
            mock_rf_instance = MagicMock()
            mock_rf_instance.predict.return_value = np.zeros(5)
            mock_rf_cls.return_value = mock_rf_instance

            mock_quant = MagicMock()
            mock_quant.base_get_aleatoric_variance.return_value = np.ones(5)
            mock_quant.standard_get_epistemic_variance.return_value = np.ones(5)
            mock_quant_cls.return_value = mock_quant

            # Test with string "1"
            run_single_test(
                func_dict, "test_func", seed=42, approaches=["Standard"],
                rf_config="1", k_neighbors="20", gap_type="empty",
                sparse_multiplier=12, scaling_law="linear", debug_timing=False,
                use_density_scaling=False, density_scaling_alpha=1.0,
                topological_decay_lambda=None, n_jobs=1, ood_type="hypercube",
                noise_std=0.1, id_split=0.7
            )
            mock_rf_cls.assert_called_with(
                n_estimators=100, min_samples_leaf=5, min_samples_split=2,
                max_features="sqrt", bootstrap=True, oob_score=True,
                n_jobs=1, random_state=42
            )

            # Test with string "5"
            run_single_test(
                func_dict, "test_func", seed=42, approaches=["Standard"],
                rf_config="5", k_neighbors="20", gap_type="empty",
                sparse_multiplier=12, scaling_law="linear", debug_timing=False,
                use_density_scaling=False, density_scaling_alpha=1.0,
                topological_decay_lambda=None, n_jobs=1, ood_type="hypercube",
                noise_std=0.1, id_split=0.7
            )
            mock_rf_cls.assert_called_with(
                n_estimators=300, min_samples_leaf=30, min_samples_split=2,
                max_features="sqrt", bootstrap=True, oob_score=True,
                n_jobs=1, random_state=42
            )

            # Test unknown preset in run_single_test raises ValueError
            with self.assertRaises(ValueError):
                run_single_test(
                    func_dict, "test_func", seed=42, approaches=["Standard"],
                    rf_config="invalid_99", k_neighbors="20", gap_type="empty",
                    sparse_multiplier=12, scaling_law="linear", debug_timing=False,
                    use_density_scaling=False, density_scaling_alpha=1.0,
                    topological_decay_lambda=None, n_jobs=1, ood_type="hypercube",
                    noise_std=0.1, id_split=0.7
                )


class TestFinding001ConfigResolution(unittest.TestCase):
    """Tests config file parsing, CLI overrides, and proximity snapshot synchronization."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_cli_proximity_parameters_captured_in_master_config(self):
        """CLI lambda, k, and alpha must be reflected in master_config snapshot."""
        args_list = [
            "--topological_decay_lambda", "7.5",
            "--k_neighbors", "3",
            "--density_scaling_alpha", "2.5",
            "--rf_config", "2"
        ]
        args, resolved_cfg = parse_args_with_config(args_list)

        self.assertEqual(resolved_cfg.rf.min_samples_leaf, 25)
        self.assertEqual(resolved_cfg.rf.n_estimators, 100)
        self.assertEqual(resolved_cfg.proximity.topological_decay_lambda, [7.5])
        self.assertEqual(resolved_cfg.proximity.k_neighbors, ["3"])
        self.assertEqual(resolved_cfg.proximity.density_scaling_alpha, [2.5])

    def test_config_file_values_used_when_not_overridden_on_cli(self):
        """Values in config_file must populate master_config and not be replaced by argparse defaults."""
        file_cfg = BenchmarkMasterConfig(
            data=DataConfig(
                gap_type="sparse",
                scaling_law="leaf",
                sparse_multiplier=42,
                noise_std=0.25,
                id_split=0.8,
                ood_type="manifold",
                seed=77
            ),
            rf=RFConfig.from_preset("4"),
            proximity=ProximityConfig(
                topological_decay_lambda=[2.5],
                k_neighbors=["15"],
                density_scaling_alpha=[3.0]
            ),
            extractors=ExtractorConfig(
                approaches=["Standard", "Chen"],
                credal_quadrature_points=40,
                credal_bisection_max_iter=60
            )
        )
        with open(self.config_path, "w") as f:
            f.write(file_cfg.to_json())

        args_list = ["--config_file", self.config_path]
        args, resolved_cfg = parse_args_with_config(args_list)

        self.assertEqual(resolved_cfg.data.gap_type, "sparse")
        self.assertEqual(resolved_cfg.data.scaling_law, "leaf")
        self.assertEqual(resolved_cfg.data.sparse_multiplier, 42)
        self.assertEqual(resolved_cfg.data.noise_std, 0.25)
        self.assertEqual(resolved_cfg.data.id_split, 0.8)
        self.assertEqual(resolved_cfg.data.ood_type, "manifold")
        self.assertEqual(resolved_cfg.data.seed, 77)
        self.assertEqual(resolved_cfg.rf.name, "4")
        self.assertEqual(resolved_cfg.rf.min_samples_leaf, 10)
        self.assertEqual(resolved_cfg.rf.n_estimators, 300)
        self.assertEqual(resolved_cfg.proximity.topological_decay_lambda, [2.5])
        self.assertEqual(resolved_cfg.proximity.k_neighbors, ["15"])
        self.assertEqual(resolved_cfg.proximity.density_scaling_alpha, [3.0])
        self.assertEqual(resolved_cfg.extractors.approaches, ["Standard", "Chen"])

    def test_cli_explicit_override_over_config_file(self):
        """Explicit CLI argument must override config_file setting."""
        file_cfg = BenchmarkMasterConfig(
            data=DataConfig(gap_type="sparse", sparse_multiplier=10),
            rf=RFConfig.from_preset("1")
        )
        with open(self.config_path, "w") as f:
            f.write(file_cfg.to_json())

        args_list = ["--config_file", self.config_path, "--sparse_multiplier", "99", "--rf_config", "5"]
        args, resolved_cfg = parse_args_with_config(args_list)

        self.assertEqual(resolved_cfg.data.sparse_multiplier, 99)
        self.assertEqual(resolved_cfg.data.gap_type, "sparse")
        self.assertEqual(resolved_cfg.rf.name, "5")
        self.assertEqual(resolved_cfg.rf.min_samples_leaf, 30)


class TestFinding001SweepGeneration(unittest.TestCase):
    """Tests that sweep generation includes material data settings."""

    def test_sweep_task_lines_include_material_data_settings(self):
        cfg = BenchmarkMasterConfig(
            data=DataConfig(
                noise_std=0.05,
                id_split=0.6,
                scaling_law="leaf",
                sparse_multiplier=25
            )
        )
        task_lines = cfg.generate_sweep_task_lines(func_name="rastrigin_2d")
        for line in task_lines:
            self.assertIn("--noise_std 0.05", line)
            self.assertIn("--id_split 0.6", line)
            self.assertIn("--scaling_law leaf", line)
            self.assertIn("--sparse_multiplier 25", line)


if __name__ == "__main__":
    unittest.main()
