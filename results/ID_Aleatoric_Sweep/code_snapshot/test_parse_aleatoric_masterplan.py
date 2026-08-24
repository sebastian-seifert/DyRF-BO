import os
import json
import shutil
import tempfile
import unittest
import pandas as pd
from scripts.parse_aleatoric_masterplan_results import parse_aleatoric_masterplan_results

class TestParseAleatoricMasterplan(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.tmp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Create mock JSON result files for testing
        mock_data_1 = {
            "standard_ari_var": {
                "spearman_true": 0.5, "spearman_resid": 0.4, "log_pearson_true": 0.45,
                "mse_var": 0.1, "rmse_var": 0.3162, "nlpd_aleatoric": 1.2
            },
            "shaker_entropy": {
                "spearman_true": 0.7, "spearman_resid": 0.6, "log_pearson_true": 0.65,
                "mse_var": 0.05, "rmse_var": 0.2236, "nlpd_aleatoric": 0.9
            },
            "shaker_geom_var": {
                "spearman_true": 0.68, "spearman_resid": 0.58, "log_pearson_true": 0.63,
                "mse_var": 0.06, "rmse_var": 0.2449, "nlpd_aleatoric": 0.95
            }
        }
        mock_data_2 = {
            "standard_ari_var": {
                "spearman_true": 0.48, "spearman_resid": 0.38, "log_pearson_true": 0.42,
                "mse_var": 0.12, "rmse_var": 0.3464, "nlpd_aleatoric": 1.25
            },
            "shaker_entropy": {
                "spearman_true": 0.72, "spearman_resid": 0.62, "log_pearson_true": 0.68,
                "mse_var": 0.04, "rmse_var": 0.2, "nlpd_aleatoric": 0.85
            },
            "shaker_geom_var": {
                "spearman_true": 0.70, "spearman_resid": 0.60, "log_pearson_true": 0.64,
                "mse_var": 0.05, "rmse_var": 0.2236, "nlpd_aleatoric": 0.92
            }
        }

        with open(os.path.join(self.tmp_dir, "res_sin_1d_hetero_linear_RF_Default_seed1.json"), "w") as f:
            json.dump(mock_data_1, f)

        with open(os.path.join(self.tmp_dir, "res_sin_cos_2d_homoscedastic_low_RF_Overfit_Leaf1_seed1.json"), "w") as f:
            json.dump(mock_data_2, f)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_parse_aleatoric_masterplan_results(self):
        df, summaries = parse_aleatoric_masterplan_results(
            results_dir=self.tmp_dir,
            output_dir=self.output_dir
        )

        self.assertIsNotNone(df)
        self.assertIsNotNone(summaries)
        self.assertIn("noise", summaries)
        self.assertIn("dim", summaries)
        self.assertIn("rf_config", summaries)
        self.assertIn("wilcoxon", summaries)

        # Check generated output files
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "aleatoric_masterplan_full_records.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "aleatoric_masterplan_by_noise.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "aleatoric_masterplan_by_dim.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "aleatoric_masterplan_by_rf_config.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "aleatoric_masterplan_wilcoxon.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "aleatoric_masterplan_analysis_report.md")))

        # Check report content
        with open(os.path.join(self.output_dir, "aleatoric_masterplan_analysis_report.md"), "r") as f:
            report_text = f.read()

        self.assertIn("Grand Summary Performance Across All Experiments", report_text)
        self.assertIn("Breakdown by Noise Regime", report_text)
        self.assertIn("Breakdown by Target Function & Dimensionality", report_text)
        self.assertIn("Breakdown by Random Forest Configuration", report_text)
        self.assertIn("Statistical Significance & Win/Tie/Loss Matrix", report_text)

if __name__ == "__main__":
    unittest.main()
