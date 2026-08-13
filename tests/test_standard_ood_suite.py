import unittest
import numpy as np
from ep_extractors.synthetic_standard_benchmarks import run_standard_benchmark_experiment

class TestStandardOODSuite(unittest.TestCase):
    def test_standard_experiment_1d_sin(self):
        results = run_standard_benchmark_experiment(
            func_name="sin",
            gap_type="empty",
            approach="proximity_bc",
            seed=1
        )
        self.assertIn("auroc", results)
        self.assertIn("spearman", results)
        self.assertGreaterEqual(results["auroc"], 0.0)

    def test_standard_experiment_10d_quadratic(self):
        results = run_standard_benchmark_experiment(
            func_name="quadratic_10d",
            gap_type="sparse",
            approach="chen_variance",
            seed=1
        )
        self.assertIn("auroc", results)
        self.assertIn("spearman", results)
        self.assertGreaterEqual(results["auroc"], 0.0)

if __name__ == "__main__":
    unittest.main()
