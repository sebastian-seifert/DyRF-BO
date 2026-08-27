import pytest
import os
import pandas as pd
from pathlib import Path
from scripts.run_carps_hypothesis_analysis import partition_carps_logs, get_hypothesis_subgroups

def test_hypothesis_subgroup_definitions():
    """Verify definitions for all 5 hypothesis subgroups."""
    subgroups = get_hypothesis_subgroups()
    
    assert "additive_vs_baseline" in subgroups
    assert "direct_vs_baseline" in subgroups
    assert "proximity_family_head_to_head" in subgroups
    assert "information_disagreement_family" in subgroups
    assert "top_finalists" in subgroups
    
    # Check K=8 for additive vs baseline
    assert len(subgroups["additive_vs_baseline"]) == 8
    assert "SMAC3_HPOFacade_ei" in subgroups["additive_vs_baseline"]
    
    # Check K=8 for direct vs baseline
    assert len(subgroups["direct_vs_baseline"]) == 8
    assert "SMAC3_HPOFacade_ei" in subgroups["direct_vs_baseline"]

def test_partition_carps_logs_output(tmp_path):
    """Verify partition_carps_logs filters the parquet dataset into valid sub-datasets."""
    # Create dummy DataFrame matching CARP-S logs schema
    records = []
    opt_ids = [
        "CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal",
        "CARPSDynamicRF_AdditiveEpistemic_ei_standard_proximity",
        "SMAC20_CustomUncertainty_ei_likelihood_credal",
        "SMAC3_HPOFacade_ei"
    ]
    for opt in opt_ids:
        for seed in [1, 2]:
            records.append({
                "task_id": "test_task_1",
                "optimizer_id": opt,
                "seed": seed,
                "trial_value__cost": 0.5,
                "n_trials": 50,
                "benchmark_id": "test",
                "task_type": "blackbox",
                "subset_id": "dev"
            })
    df = pd.DataFrame(records)
    input_parquet = str(tmp_path / "dummy_logs.parquet")
    df.to_parquet(input_parquet)
    
    subgroups = {
        "test_subgroup": [
            "CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal",
            "SMAC3_HPOFacade_ei"
        ]
    }
    
    out_dir = str(tmp_path / "partitions")
    partition_map = partition_carps_logs(input_parquet, out_dir, subgroups=subgroups)
    
    assert "test_subgroup" in partition_map
    sub_path = partition_map["test_subgroup"]
    assert os.path.exists(sub_path)
    
    sub_df = pd.read_parquet(sub_path)
    assert set(sub_df["optimizer_id"].unique()) == set(subgroups["test_subgroup"])
