import pytest
import os
import pandas as pd
import numpy as np
from scripts.compute_pairwise_wilcoxon_suite import (
    compute_wilcoxon_suite,
    apply_holm_bonferroni,
    calculate_cliffs_delta
)

def test_cliffs_delta_calculation():
    """Verify Cliff's delta non-parametric effect size calculation."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    delta = calculate_cliffs_delta(x, y)
    # x is systematically smaller than y, so delta should be negative
    assert delta < 0.0
    assert -1.0 <= delta <= 1.0

def test_holm_bonferroni_adjustment():
    """Verify step-down Holm-Bonferroni adjustment."""
    raw_p = [0.01, 0.04, 0.03, 0.20]
    adjusted = apply_holm_bonferroni(raw_p)
    
    assert len(adjusted) == len(raw_p)
    # Adjusted p-values must be >= raw p-values and <= 1.0
    for r, a in zip(raw_p, adjusted):
        assert a >= r
        assert a <= 1.0

def test_compute_wilcoxon_suite_end_to_end(tmp_path):
    """Verify compute_wilcoxon_suite processes a dataframe and writes .md, .csv, and .tex files."""
    records = []
    opt_ids = [
        "CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal",
        "SMAC20_CustomUncertainty_ei_likelihood_credal",
        "SMAC3_HPOFacade_ei"
    ]
    # Create 10 dummy tasks with 5 seeds each
    for t_idx in range(10):
        task_id = f"task_{t_idx}"
        for seed in range(1, 6):
            # Baseline has slightly higher cost
            records.append({
                "task_id": task_id,
                "optimizer_id": "SMAC3_HPOFacade_ei",
                "seed": seed,
                "trial_value__cost_inc_norm": 0.05 + 0.01 * t_idx + 0.002 * seed,
                "n_trials": 50
            })
            # Additive has lowest cost
            records.append({
                "task_id": task_id,
                "optimizer_id": "CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal",
                "seed": seed,
                "trial_value__cost_inc_norm": 0.02 + 0.005 * t_idx + 0.001 * seed,
                "n_trials": 50
            })
            # Direct has medium cost
            records.append({
                "task_id": task_id,
                "optimizer_id": "SMAC20_CustomUncertainty_ei_likelihood_credal",
                "seed": seed,
                "trial_value__cost_inc_norm": 0.03 + 0.008 * t_idx + 0.001 * seed,
                "n_trials": 50
            })
            
    df = pd.DataFrame(records)
    parquet_path = str(tmp_path / "dummy_logs.parquet")
    df.to_parquet(parquet_path)
    
    out_dir = str(tmp_path / "stats_output")
    res_vs_base, res_h2h = compute_wilcoxon_suite(
        input_parquet=parquet_path,
        output_dir=out_dir,
        baseline_id="SMAC3_HPOFacade_ei"
    )
    
    # Check return DataFrames
    assert len(res_vs_base) == 2  # 2 optimizers vs baseline
    assert len(res_h2h) == 1     # 1 head-to-head comparison
    
    # Check generated files
    assert os.path.exists(os.path.join(out_dir, "wilcoxon_tests_vs_baseline.md"))
    assert os.path.exists(os.path.join(out_dir, "wilcoxon_tests_vs_baseline.csv"))
    assert os.path.exists(os.path.join(out_dir, "wilcoxon_tests_vs_baseline.tex"))
    assert os.path.exists(os.path.join(out_dir, "wilcoxon_head_to_head_additive_vs_direct.md"))
    assert os.path.exists(os.path.join(out_dir, "wilcoxon_head_to_head_additive_vs_direct.tex"))
