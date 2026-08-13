import unittest
import numpy as np
from ep_extractors.synthetic_ood_benchmarks import run_ood_detection_experiment

class TestOODDetectionSuite(unittest.TestCase):
    def test_ood_experiment_ackley_2d_empty_gap(self):
        """Test single OOD benchmark run on ackley_2d empty gap across standard extractors."""
        results = run_ood_detection_experiment(
            func_name="ackley_2d",
            gap_type="empty",
            approach="proximity_bc",
            seed=1,
            noise_std=0.01,
            id_split=0.7
        )
        
        self.assertIn("auroc", results)
        self.assertIn("aupr", results)
        self.assertIn("spearman", results)
        self.assertIn("aurc", results)
        self.assertIn("brier", results)
        
        # AUROC must be between 0.0 and 1.0
        self.assertGreaterEqual(results["auroc"], 0.0)
        self.assertLessEqual(results["auroc"], 1.0)
        
        # Spearman correlation must be in [-1, 1]
        self.assertGreaterEqual(results["spearman"], -1.0)
        self.assertLessEqual(results["spearman"], 1.0)

    def test_ood_experiment_rosenbrock_2d_sparse_gap(self):
        """Test single OOD benchmark run on rosenbrock_2d sparse gap."""
        results = run_ood_detection_experiment(
            func_name="rosenbrock_2d",
            gap_type="sparse",
            approach="chen_variance",
            seed=2,
            noise_std=0.01,
            id_split=0.7
        )
        
        self.assertIn("auroc", results)
        self.assertIn("spearman", results)
        self.assertGreaterEqual(results["auroc"], 0.0)

    def test_ood_experiment_hartmann_6d_empty_gap(self):
        """Test high-dimensional 6D OOD benchmark run on hartmann_6d."""
        results = run_ood_detection_experiment(
            func_name="hartmann_6d",
            gap_type="empty",
            approach="proximity_auto_lambda",
            seed=3,
            noise_std=0.01,
            id_split=0.7
        )
        
        self.assertIn("auroc", results)
        self.assertIn("spearman", results)
        self.assertGreaterEqual(results["auroc"], 0.0)

if __name__ == "__main__":
    unittest.main()
