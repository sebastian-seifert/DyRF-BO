import os
import sys
import unittest
import numpy as np
import tempfile
import shutil
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_anytime_performance_plot import (
    make_step_coords, compute_anytime_trajectories
)

class TestAnytimePerformancePlot(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.baseline_dir = os.path.join(self.test_dir, "baseline")
        self.ei_dir = os.path.join(self.test_dir, "ei")
        os.makedirs(self.baseline_dir, exist_ok=True)
        os.makedirs(self.ei_dir, exist_ok=True)

        # Create dummy telemetry JSONs
        dummy_telemetry_1 = {
            "task_name": "task1",
            "seed": 1,
            "extractor_name": "smac3_bo",
            "trials": [
                {"n_trials": 1, "trial_value": {"cost": 10.0}},
                {"n_trials": 2, "trial_value": {"cost": 8.0}},
                {"n_trials": 3, "trial_value": {"cost": 9.0}}, # cumulative min stays 8.0
                {"n_trials": 4, "trial_value": {"cost": 5.0}},
            ]
        }
        dummy_telemetry_2 = {
            "task_name": "task1",
            "seed": 2,
            "extractor_name": "smac3_bo",
            "trials": [
                {"n_trials": 1, "trial_value": {"cost": 12.0}},
                {"n_trials": 2, "trial_value": {"cost": 6.0}},
                {"n_trials": 3, "trial_value": {"cost": 7.0}},
                {"n_trials": 4, "trial_value": {"cost": 4.0}},
            ]
        }

        with open(os.path.join(self.baseline_dir, "telemetry_smac3_seed1.json"), "w") as f:
            json.dump(dummy_telemetry_1, f)
        with open(os.path.join(self.baseline_dir, "telemetry_smac3_seed2.json"), "w") as f:
            json.dump(dummy_telemetry_2, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_make_step_coords_rechtsstetig(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([10.0, 8.0, 8.0, 5.0])
        x_step, y_step = make_step_coords(x, y)

        # For "where='post'" (right-continuous step function):
        # x_step: [1, 2, 2, 3, 3, 4, 4]
        # y_step: [10, 10, 8, 8, 8, 5, 5]
        self.assertEqual(len(x_step), 2 * len(x) - 1)
        self.assertEqual(len(y_step), 2 * len(y) - 1)
        np.testing.assert_array_equal(x_step, np.array([1, 2, 2, 3, 3, 4, 4]))
        np.testing.assert_array_equal(y_step, np.array([10.0, 10.0, 8.0, 8.0, 8.0, 8.0, 5.0]))

    def test_compute_anytime_trajectories(self):
        trajectories = compute_anytime_trajectories(self.test_dir, task_filter="task1")
        self.assertIn("smac3_bo", trajectories)

        data = trajectories["smac3_bo"]
        self.assertIn("mean", data)
        self.assertIn("sem", data)
        self.assertIn("iterations", data)

        # Iteration 1: min values are 10.0 and 12.0 -> mean = 11.0, std = 1.414, sem = 1.414 / sqrt(2) = 1.0
        # Iteration 2: min values are 8.0 and 6.0 -> mean = 7.0, sem = 1.0
        # Iteration 4: min values are 5.0 and 4.0 -> mean = 4.5, sem = 0.5
        self.assertAlmostEqual(data["mean"][0], 11.0)
        self.assertAlmostEqual(data["mean"][1], 7.0)
        self.assertAlmostEqual(data["mean"][3], 4.5)
        self.assertAlmostEqual(data["sem"][0], 1.0)

if __name__ == "__main__":
    unittest.main()
