import os
import json
import pytest
import pandas as pd

def test_parse_aleatoric_ood_results(tmp_path):
    from scripts.parse_aleatoric_ood_results import parse_aleatoric_ood_results
    
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    
    # Create mock JSON results
    mock_record = {
        "shaker_geom_var": {
            "global_spearman_true": 0.85,
            "global_spearman_resid": 0.45,
            "global_log_pearson_true": 0.82,
            "global_mse_var": 0.01,
            "global_rmse_var": 0.10,
            "global_nlpd_aleatoric": 0.15,
            "id_only_spearman_true": 0.88,
            "id_only_spearman_resid": 0.48,
            "id_only_log_pearson_true": 0.86,
            "id_only_mse_var": 0.008,
            "id_only_rmse_var": 0.09,
            "id_only_nlpd_aleatoric": 0.12,
            "ood_only_spearman_true": 0.40,
            "ood_only_spearman_resid": 0.20,
            "ood_only_log_pearson_true": 0.35,
            "ood_only_mse_var": 0.05,
            "ood_only_rmse_var": 0.22,
            "ood_only_nlpd_aleatoric": 0.45,
            "ood_id_variance_ratio": 1.05
        },
        "standard_disagreement": {
            "global_spearman_true": 0.70,
            "global_spearman_resid": 0.35,
            "global_log_pearson_true": 0.65,
            "global_mse_var": 0.12,
            "global_rmse_var": 0.34,
            "global_nlpd_aleatoric": 0.80,
            "id_only_spearman_true": 0.72,
            "id_only_spearman_resid": 0.38,
            "id_only_log_pearson_true": 0.68,
            "id_only_mse_var": 0.10,
            "id_only_rmse_var": 0.31,
            "id_only_nlpd_aleatoric": 0.75,
            "ood_only_spearman_true": 0.30,
            "ood_only_spearman_resid": 0.15,
            "ood_only_log_pearson_true": 0.25,
            "ood_only_mse_var": 0.25,
            "ood_only_rmse_var": 0.50,
            "ood_only_nlpd_aleatoric": 1.20,
            "ood_id_variance_ratio": 4.50
        }
    }
    
    # Write mock files
    with open(json_dir / "res_sin_cos_1d_hetero_ood_step_double_RF_Default_seed1.json", "w") as fp:
        json.dump(mock_record, fp)
    with open(json_dir / "res_sin_cos_2d_hetero_ood_step_double_RF_Default_seed1.json", "w") as fp:
        json.dump(mock_record, fp)
        
    parse_aleatoric_ood_results(json_dir=str(json_dir), output_dir=str(out_dir))
    
    expected_files = [
        "aleatoric_ood_masterplan_full_records.csv",
        "aleatoric_ood_masterplan_grand_summary.csv",
        "aleatoric_ood_masterplan_by_noise.csv",
        "aleatoric_ood_masterplan_by_dim.csv",
        "aleatoric_ood_masterplan_by_rf_config.csv",
        "aleatoric_ood_masterplan_analysis_report.md"
    ]
    for ef in expected_files:
        p = out_dir / ef
        assert p.exists(), f"Missing output file {ef}"
        
    df_full = pd.read_csv(out_dir / "aleatoric_ood_masterplan_full_records.csv")
    assert "ood_id_variance_ratio" in df_full.columns
    assert "id_only_spearman_true" in df_full.columns
    assert "ood_only_spearman_true" in df_full.columns
