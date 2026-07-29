import os
import json
import tempfile
import unittest

from scripts.generate_highdim_benchmark_tables import parse_highdim_telemetry, format_benchmark_table

class TestGenerateHighDimBenchmarkTables(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.baseline_dir = os.path.join(self.test_dir, "baseline")
        self.ei_dir = os.path.join(self.test_dir, "ei")
        os.makedirs(self.baseline_dir, exist_ok=True)
        os.makedirs(self.ei_dir, exist_ok=True)

        # Write dummy baseline json with trial_value dict
        b_data = {
            "task_name": "cfg_rbv2_super_1040",
            "seed": 1,
            "extractor_name": "smac3_bo",
            "trials": [{"n_trials": 1, "trial_value": {"cost": -0.85}}, {"n_trials": 2, "trial_value": {"cost": -0.90}}]
        }
        with open(os.path.join(self.baseline_dir, "telemetry_smac3_cfg_rbv2_super_1040_seed1.json"), "w") as f:
            json.dump(b_data, f)

        # Write dummy custom approach json with full task path
        c_data = {
            "task_name": "yahpo/so/rbv2_super/1040/None",
            "seed": 1,
            "extractor_name": "Chen",
            "trials": [{"trial_idx": 1, "cost": -0.88}, {"trial_idx": 2, "cost": -0.95}]
        }
        with open(os.path.join(self.ei_dir, "telemetry_Chen_cfg_rbv2_super_1040_seed1.json"), "w") as f:
            json.dump(c_data, f)

    def test_parse_and_format(self):
        results = parse_highdim_telemetry(dirs=[self.baseline_dir, self.ei_dir])
        self.assertIn("cfg_rbv2_super_1040", results)
        self.assertIn("smac3_bo", results["cfg_rbv2_super_1040"])
        self.assertIn("Chen", results["cfg_rbv2_super_1040"])
        self.assertEqual(results["cfg_rbv2_super_1040"]["smac3_bo"][1], -0.90)
        self.assertEqual(results["cfg_rbv2_super_1040"]["Chen"][1], -0.95)

        table_md = format_benchmark_table("cfg_rbv2_super_1040", results["cfg_rbv2_super_1040"])
        self.assertIn("## Benchmark Task: `cfg_rbv2_super_1040`", table_md)
        self.assertIn("| `Chen` |", table_md)
        self.assertIn("| `smac3_bo` |", table_md)

if __name__ == "__main__":
    unittest.main()
