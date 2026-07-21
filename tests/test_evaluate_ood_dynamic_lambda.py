import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_ood_dynamic_lambda import run_ood_evaluation

class TestEvaluateOODDynamicLambda(unittest.TestCase):
    def test_run_ood_evaluation_hypercube_and_manifold(self):
        # Run a quick evaluation with 1 seed on sin_1d for both ood_types
        results = run_ood_evaluation(
            funcs=["sin_1d"],
            seeds=[42],
            ood_types=["hypercube", "manifold"],
            gap_types=["empty"]
        )
        self.assertIn("hypercube", results)
        self.assertIn("manifold", results)
        
        for ood_type in ["hypercube", "manifold"]:
            self.assertIn("sin_1d", results[ood_type])
            func_res = results[ood_type]["sin_1d"]
            self.assertIn("standard_rf", func_res)
            self.assertIn("proximity_auto_lambda", func_res)
            
            for app in ["standard_rf", "proximity_auto_lambda"]:
                metrics = func_res[app]
                self.assertIn("auroc", metrics)
                self.assertIn("auprc", metrics)
                self.assertIn("fpr95", metrics)
                self.assertIn("jsd", metrics)
                self.assertIn("nmi", metrics)
                self.assertIn("naurc", metrics)
                
                self.assertIsInstance(metrics["naurc"], float)
                self.assertFalse(np.isnan(metrics["naurc"]))

if __name__ == "__main__":
    unittest.main()
