import os
import sys
import unittest
import pandas as pd

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.compare_auto_lambda_carps import merge_carps_results


class TestCompareAutoLambdaCARPS(unittest.TestCase):
    """
    Test suite for merging auto-lambda CARP-S results with previous CARP-S summary baselines.
    """

    def test_merge_carps_results(self):
        prev_dir = "results/carps_summary_21072026/tables"
        auto_dir = "results/carps_auto_lambda_summary_22072026/tables"
        out_dir = "results/carps_auto_lambda_comparison_22072026/tables"

        if os.path.exists(prev_dir) and os.path.exists(auto_dir):
            df_ranks, task_tables = merge_carps_results(prev_dir, auto_dir, out_dir)
            self.assertEqual(len(task_tables), 26)
            self.assertIn("proximity_auto_lambda", df_ranks["Approach"].values)
            self.assertIn("smac3_bo", df_ranks["Approach"].values)
            self.assertIn("proximity_bc", df_ranks["Approach"].values)
            self.assertEqual(len(df_ranks), 9)


if __name__ == "__main__":
    unittest.main()
