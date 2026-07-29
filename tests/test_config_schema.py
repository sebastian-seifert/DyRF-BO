import os
import sys
import unittest
import json
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_schema import (
    DataConfig, RFConfig, ExtractorConfig, ProximityConfig,
    AcquisitionConfig, BenchmarkMasterConfig
)

class TestConfigSchema(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_default_config_instantiation(self):
        cfg = BenchmarkMasterConfig()
        self.assertEqual(cfg.data.ndim, 1)
        self.assertEqual(cfg.rf.n_estimators, 100)
        self.assertIn("Standard", cfg.extractors.approaches)
        self.assertTrue(cfg.proximity.use_density_scaling)

    def test_rf_preset_resolution(self):
        rf_a = RFConfig.from_preset("A")
        self.assertEqual(rf_a.n_estimators, 100)
        self.assertEqual(rf_a.min_samples_leaf, 5)

        rf_b = RFConfig.from_preset("B")
        self.assertEqual(rf_b.n_estimators, 500)
        self.assertEqual(rf_b.min_samples_leaf, 10)

        rf_c = RFConfig.from_preset("C")
        self.assertEqual(rf_c.n_estimators, 1000)
        self.assertEqual(rf_c.min_samples_leaf, 25)

    def test_json_serialization_deserialization(self):
        cfg = BenchmarkMasterConfig()
        json_str = cfg.to_json()
        data_dict = json.loads(json_str)
        self.assertEqual(data_dict["data"]["ndim"], 1)

        reconstructed = BenchmarkMasterConfig.from_dict(data_dict)
        self.assertEqual(reconstructed.data.ndim, 1)
        self.assertEqual(reconstructed.rf.n_estimators, 100)

    def test_generate_sweep_task_cli_lines(self):
        cfg = BenchmarkMasterConfig()
        task_lines = cfg.generate_sweep_task_lines(func_name="ackley_1d")
        self.assertGreater(len(task_lines), 0)
        self.assertIn("--function ackley_1d", task_lines[0])

if __name__ == "__main__":
    unittest.main()
