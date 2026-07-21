import os
import sys
import json
import shutil
import tempfile
import unittest

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_carps_results import (
    parse_telemetry_directory,
    parse_array_logs,
    parse_bo_histories,
    compute_anytime_stats,
    generate_benchmark_tables,
    generate_anytime_plots,
)

class TestAnalyzeCARPSResults(unittest.TestCase):
    def setUp(self):
        # Create temp results directory
        self.test_dir = tempfile.mkdtemp()
        self.out_dir = os.path.join(self.test_dir, "carps_summary")
        
        # Create mock telemetry JSON files
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

        # Create mock array log files
        log_content_1 = (
            "==================================================\n"
            "Array Job ID: 100 | Task Index: 1\n"
            "Running arguments: +optimizer=dyrf_epistemic_hpobench optimizer.extractor_name=standard_disagreement +task/HPOBench/blackbox/tabular/ml=mock_task task.optimization_resources.n_trials=3 seed=1\n"
            "==================================================\n"
            "[INFO][file_logger.py:201] n_trials: 1, n_function_calls: 1, config: [], cost: 0.9\n"
            "[INFO][file_logger.py:201] n_trials: 2, n_function_calls: 2, config: [], cost: 0.5\n"
            "[INFO][file_logger.py:201] n_trials: 3, n_function_calls: 3, config: [], cost: 0.7\n"
            "Solution found:\n"
            "TrialValue(\n"
            "    cost=0.5,\n"
            "    time=0.01\n"
            ")\n"
        )
        log_content_2 = (
            "==================================================\n"
            "Array Job ID: 100 | Task Index: 2\n"
            "Running arguments: +optimizer/smac20=hpo +task/YAHPO/SO=mock_task task.optimization_resources.n_trials=3 seed=1\n"
            "==================================================\n"
            "[INFO][file_logger.py:201] n_trials: 1, n_function_calls: 1, config: [], cost: 0.8\n"
            "[INFO][file_logger.py:201] n_trials: 2, n_function_calls: 2, config: [], cost: 0.3\n"
            "[INFO][file_logger.py:201] n_trials: 3, n_function_calls: 3, config: [], cost: 0.4\n"
            "Solution found:\n"
            "TrialValue(\n"
            "    cost=0.3,\n"
            "    time=0.01\n"
            ")\n"
        )
        with open(os.path.join(self.test_dir, "array_100_1.log"), "w") as f:
            f.write(log_content_1)
        with open(os.path.join(self.test_dir, "array_100_2.log"), "w") as f:
            f.write(log_content_2)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_parse_telemetry_directory(self):
        results = parse_telemetry_directory(self.test_dir)
        self.assertIn("mock_task", results)
        self.assertIn("standard_disagreement", results["mock_task"])
        self.assertIn("chen_variance", results["mock_task"])
        self.assertEqual(sorted(results["mock_task"]["standard_disagreement"]), [0.4, 0.5])
        self.assertEqual(sorted(results["mock_task"]["chen_variance"]), [0.2, 0.3])

    def test_parse_array_logs(self):
        final_costs = parse_array_logs(self.test_dir)
        self.assertIn("mock_task", final_costs)
        self.assertIn("standard_disagreement", final_costs["mock_task"])
        self.assertEqual(final_costs["mock_task"]["standard_disagreement"][1], 0.5)
        self.assertIn("smac3_bo", final_costs["mock_task"])
        self.assertEqual(final_costs["mock_task"]["smac3_bo"][1], 0.3)

    def test_compute_anytime_stats(self):
        # Seed 1: [0.9, 0.5, 0.7] -> incumbent: [0.9, 0.5, 0.5]
        # Seed 2: [0.8, 0.4, 0.6] -> incumbent: [0.8, 0.4, 0.4]
        histories = {
            1: [0.9, 0.5, 0.7],
            2: [0.8, 0.4, 0.6]
        }
        indices, mean_traj, se_traj = compute_anytime_stats(histories)
        self.assertEqual(list(indices), [1, 2, 3])
        # Mean incumbent at step 1: (0.9 + 0.8)/2 = 0.85
        # Step 2: (0.5 + 0.4)/2 = 0.45
        # Step 3: (0.5 + 0.4)/2 = 0.45
        self.assertAlmostEqual(mean_traj[0], 0.85)
        self.assertAlmostEqual(mean_traj[1], 0.45)
        self.assertAlmostEqual(mean_traj[2], 0.45)
        self.assertTrue(se_traj[0] > 0)

    def test_generate_benchmark_tables(self):
        final_costs = {
            "mock_task": {
                "standard_disagreement": {1: 0.5, 2: 0.4},
                "chen_variance": {1: 0.3, 2: 0.2}
            }
        }
        report_md = generate_benchmark_tables(final_costs, self.out_dir)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "tables", "mock_task_comparison.csv")))
        self.assertIn("chen_variance", report_md)
        self.assertIn("standard_disagreement", report_md)

    def test_generate_anytime_plots(self):
        bo_histories = {
            "mock_task": {
                "standard_disagreement": {
                    1: [0.9, 0.5, 0.7],
                    2: [0.8, 0.4, 0.6]
                },
                "chen_variance": {
                    1: [0.9, 0.3, 0.5],
                    2: [0.7, 0.2, 0.4]
                }
            }
        }
        plots_created = generate_anytime_plots(bo_histories, self.out_dir)
        self.assertEqual(len(plots_created), 1)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "plots", "mock_task_anytime.png")))

    def test_main_cli_arguments(self):
        from scripts.analyze_carps_results import main
        custom_out = os.path.join(self.test_dir, "custom_carps_summary")
        test_args = ["analyze_carps_results.py", "--results_dir", self.test_dir, "--output_dir", custom_out]
        orig_argv = sys.argv
        try:
            sys.argv = test_args
            main()
            self.assertTrue(os.path.exists(os.path.join(custom_out, "summary_report.md")))
        finally:
            sys.argv = orig_argv

if __name__ == "__main__":
    unittest.main()

