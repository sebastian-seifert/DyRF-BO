import os
import sys
import json
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_schema import BenchmarkMasterConfig, DataConfig, RFConfig

class TestUQConfigSnapshot(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.json")
        self.output_dir = os.path.join(self.test_dir, "raw")

        cfg = BenchmarkMasterConfig(
            data=DataConfig(gap_type="empty", seed=42, noise_std=0.05, id_split=0.6),
            rf=RFConfig.from_preset("A")
        )
        with open(self.config_path, "w") as f:
            f.write(cfg.to_json())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_config_file_argument_parsing(self):
        import argparse
        from Uncertainty_Quantification import parse_args_with_config

        args_list = ["--config_file", self.config_path, "--function", "sin", "--output_dir", self.output_dir]
        parsed_args, resolved_config = parse_args_with_config(args_list)

        self.assertEqual(parsed_args.function, "sin")
        self.assertEqual(resolved_config.data.seed, 42)
        self.assertEqual(resolved_config.data.noise_std, 0.05)
        self.assertEqual(resolved_config.data.id_split, 0.6)

if __name__ == "__main__":
    unittest.main()
