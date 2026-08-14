import unittest
import os
import json
import tempfile
import pandas as pd
from scripts.parse_standard_sweep_results import parse_standard_sweep_results, extract_dim_from_func_name

class TestParseStandardSweep(unittest.TestCase):
    def test_extract_dim(self):
        self.assertEqual(extract_dim_from_func_name("sin"), 1)
        self.assertEqual(extract_dim_from_func_name("quadratic"), 2)
        self.assertEqual(extract_dim_from_func_name("ackley_4d"), 4)
        self.assertEqual(extract_dim_from_func_name("hartmann_6d"), 6)
        self.assertEqual(extract_dim_from_func_name("sin_cos_15d"), 15)

    def test_parser_with_dummy_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy JSON records
            for seed in [1, 2]:
                rec = {
                    "func_name": "sin_cos_4d",
                    "gap_type": "empty",
                    "approach": "proximity_bc",
                    "seed": seed,
                    "auroc": 0.85 + seed * 0.01,
                    "fpr95": 0.20,
                    "aupr": 0.90,
                    "spearman": 0.75,
                    "aurc": 0.12,
                    "oracle_aurc": 0.08,
                    "jsd": 0.45,
                    "mi": 0.55,
                    "nlpd": 0.30,
                    "brier": 0.15,
                    "n_train": 100,
                    "n_test": 500
                }
                with open(os.path.join(tmpdir, f"dummy_seed_{seed}.json"), "w") as f:
                    json.dump(rec, f)

            df_raw, df_dim = parse_standard_sweep_results(results_dir=tmpdir, output_dir=tmpdir)
            self.assertIsNotNone(df_raw)
            self.assertIsNotNone(df_dim)
            self.assertIn("dim", df_raw.columns)
            self.assertIn("auroc", df_dim.columns)
            self.assertIn("spearman", df_dim.columns)
            self.assertIn("aurc", df_dim.columns)
            self.assertIn("oracle_aurc", df_dim.columns)
            self.assertIn("jsd", df_dim.columns)
            self.assertIn("mi", df_dim.columns)
            self.assertIn("brier", df_dim.columns)
            self.assertIn("nlpd", df_dim.columns)

if __name__ == "__main__":
    unittest.main()
