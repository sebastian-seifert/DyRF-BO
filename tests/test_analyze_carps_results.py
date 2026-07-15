import os
import sys
import json
import shutil
import tempfile
import unittest

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_carps_results import parse_telemetry_directory

class TestAnalyzeCARPSResults(unittest.TestCase):
    def setUp(self):
        # Create temp results directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock telemetry JSON files
        # Approach 1: standard_disagreement, Task: mock_task, Seed: 1 & 2
        mock_data_1_1 = {
            "task_name": "mock_task",
            "extractor_name": "standard_disagreement",
            "trials": [
                {"trial_idx": 0, "cost": 0.9},
                {"trial_idx": 1, "cost": 0.5},
                {"trial_idx": 2, "cost": 0.7}
            ]
        }
        mock_data_1_2 = {
            "task_name": "mock_task",
            "extractor_name": "standard_disagreement",
            "trials": [
                {"trial_idx": 0, "cost": 0.8},
                {"trial_idx": 1, "cost": 0.4},
                {"trial_idx": 2, "cost": 0.6}
            ]
        }
        
        # Approach 2: chen_variance, Task: mock_task, Seed: 1 & 2
        mock_data_2_1 = {
            "task_name": "mock_task",
            "extractor_name": "chen_variance",
            "trials": [
                {"trial_idx": 0, "cost": 0.9},
                {"trial_idx": 1, "cost": 0.3},
                {"trial_idx": 2, "cost": 0.5}
            ]
        }
        mock_data_2_2 = {
            "task_name": "mock_task",
            "extractor_name": "chen_variance",
            "trials": [
                {"trial_idx": 0, "cost": 0.7},
                {"trial_idx": 1, "cost": 0.2},
                {"trial_idx": 2, "cost": 0.4}
            ]
        }
        
        with open(os.path.join(self.test_dir, "telemetry_standard_disagreement_mock_task_seed1.json"), "w") as f:
            json.dump(mock_data_1_1, f)
        with open(os.path.join(self.test_dir, "telemetry_standard_disagreement_mock_task_seed2.json"), "w") as f:
            json.dump(mock_data_1_2, f)
        with open(os.path.join(self.test_dir, "telemetry_chen_variance_mock_task_seed1.json"), "w") as f:
            json.dump(mock_data_2_1, f)
        with open(os.path.join(self.test_dir, "telemetry_chen_variance_mock_task_seed2.json"), "w") as f:
            json.dump(mock_data_2_2, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_parse_telemetry_directory(self):
        results = parse_telemetry_directory(self.test_dir)
        
        # We expect a nested dictionary: {task: {approach: [best_cost_seed1, best_cost_seed2]}}
        self.assertIn("mock_task", results)
        self.assertIn("standard_disagreement", results["mock_task"])
        self.assertIn("chen_variance", results["mock_task"])
        
        # Best costs for standard_disagreement should be min([0.9, 0.5, 0.7]) = 0.5, and min([0.8, 0.4, 0.6]) = 0.4
        self.assertEqual(sorted(results["mock_task"]["standard_disagreement"]), [0.4, 0.5])
        
        # Best costs for chen_variance should be min([0.9, 0.3, 0.5]) = 0.3, and min([0.7, 0.2, 0.4]) = 0.2
        self.assertEqual(sorted(results["mock_task"]["chen_variance"]), [0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
