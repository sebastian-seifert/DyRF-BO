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

        from scripts.generate_highdim_benchmark_tables import compute_average_ranks
        avg_ranks = compute_average_ranks(results)
        self.assertEqual(avg_ranks["Chen"], 1.0)
        self.assertEqual(avg_ranks["smac3_bo"], 2.0)

    def test_parse_seed_from_filename_when_missing_in_json(self):
        # Create files without "seed" in JSON content (simulating custom telemetry files)
        for s in range(1, 6):
            c_data = {
                "task_name": "yahpo/so/rbv2_super/1050/None",
                "extractor_name": "standard_proximity",
                "trials": [{"trial_idx": 1, "cost": -0.5 - (s * 0.01)}]
            }
            fname = f"telemetry_epistemic_ei_standard_proximity_cfg_rbv2_super_1050_seed{s}.json"
            with open(os.path.join(self.ei_dir, fname), "w") as f:
                json.dump(c_data, f)

        results = parse_highdim_telemetry(dirs=[self.ei_dir])
        self.assertIn("cfg_rbv2_super_1050", results)
        self.assertIn("standard_proximity", results["cfg_rbv2_super_1050"])
        # Should contain seeds 1 through 5
        seed_map = results["cfg_rbv2_super_1050"]["standard_proximity"]
        self.assertEqual(len(seed_map), 5)
        self.assertEqual(set(seed_map.keys()), {1, 2, 3, 4, 5})

if __name__ == "__main__":
    unittest.main()

