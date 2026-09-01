#!/usr/bin/env python3
"""TDD Test Suite for 1v1 Wilcoxon Statistical Analysis Suite.

Tests mathematical correctness of Wilcoxon Signed-Rank tests, Rank-Biserial correlation (r_rb),
Cliff's delta, paired Cohen's d, multiple comparison corrections (Holm-Bonferroni, Benjamini-Hochberg),
seed alignment, zero-difference edge cases, and report exporters.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure scripts module is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_1v1_wilcoxon_analysis import (
    StatisticalMathEngine,
    DataLoader,
    PairingEngine,
    StatisticalAnalysisEngine,
    ReportExporter,
    TaskWilcoxonResult,
    AggregateSummary
)


@pytest.fixture
def synthetic_task_data():
    """Generates synthetic dataframe with 2 tasks, 5 seeds, 1 candidate, and 1 baseline."""
    rows = []
    # Task 1: Candidate is strictly better on all seeds
    for s in range(1, 6):
        # 3 trials per run
        for trial in [10, 30, 50]:
            rows.append({
                "task_id": "task_easy",
                "optimizer_id": "Cand_Good",
                "seed": s,
                "n_trials": trial,
                "trial_value__cost_inc": 0.10 * (50 - trial + 1) / 50.0  # Final = 0.002
            })
            rows.append({
                "task_id": "task_easy",
                "optimizer_id": "Base_Ref",
                "seed": s,
                "n_trials": trial,
                "trial_value__cost_inc": 0.50 * (50 - trial + 1) / 50.0  # Final = 0.010
            })
            
    # Task 2: Exact tie (identical values)
    for s in range(1, 6):
        for trial in [10, 30, 50]:
            rows.append({
                "task_id": "task_tied",
                "optimizer_id": "Cand_Good",
                "seed": s,
                "n_trials": trial,
                "trial_value__cost_inc": 0.25
            })
            rows.append({
                "task_id": "task_tied",
                "optimizer_id": "Base_Ref",
                "seed": s,
                "n_trials": trial,
                "trial_value__cost_inc": 0.25
            })
    return pd.DataFrame(rows)


def test_r_rb_perfect_separation():
    """Candidate strictly lower than baseline -> r_rb = -1.0, W+ > 0, W- = 0."""
    x_cand = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_base = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # diff = -1.0 everywhere
    w_plus, w_minus, r_rb = StatisticalMathEngine.calculate_r_rb(x_cand, y_base)
    assert w_plus == 15.0  # 1+2+3+4+5
    assert w_minus == 0.0
    assert r_rb == -1.0  # Polarity: negative favors candidate (lower cost)


def test_r_rb_symmetric():
    """Candidate and baseline identical -> r_rb = 0.0, W+ = 0, W- = 0."""
    x_cand = np.array([2.0, 2.0, 2.0])
    y_base = np.array([2.0, 2.0, 2.0])
    w_plus, w_minus, r_rb = StatisticalMathEngine.calculate_r_rb(x_cand, y_base)
    assert w_plus == 0.0
    assert w_minus == 0.0
    assert r_rb == 0.0


def test_wilcoxon_zero_difference_safety():
    """Identical arrays do not raise ValueError and return W=0, p=1.0."""
    x = np.array([0.42, 0.42, 0.42, 0.42])
    y = np.array([0.42, 0.42, 0.42, 0.42])
    w_stat, p_val = StatisticalMathEngine.safe_paired_wilcoxon(x, y)
    assert w_stat == 0.0
    assert p_val == 1.0


def test_corrections_bounds_and_monotonicity():
    """Holm-Bonferroni and Benjamini-Hochberg adjustments preserve bounds and order."""
    p_raw = [0.001, 0.04, 0.02, 0.50]
    p_holm, p_bh = StatisticalMathEngine.apply_corrections(p_raw)
    
    assert len(p_holm) == 4
    assert len(p_bh) == 4
    for p_h, p_b, raw in zip(p_holm, p_bh, p_raw):
        assert 0.0 <= p_h <= 1.0
        assert 0.0 <= p_b <= 1.0
        assert p_h >= raw
        assert p_b >= raw

    # Check monotonicity along sorted order
    sort_idx = np.argsort(p_raw)
    sorted_holm = [p_holm[i] for i in sort_idx]
    sorted_bh = [p_bh[i] for i in sort_idx]
    for i in range(1, len(sorted_holm)):
        assert sorted_holm[i] >= sorted_holm[i-1]
        assert sorted_bh[i] >= sorted_bh[i-1]


def test_seed_pairing_alignment():
    """PairingEngine matches identical seeds and drops unmatched seeds safely."""
    df_unaligned = pd.DataFrame([
        {"task_id": "t1", "optimizer_id": "cand", "seed": 1, "trial_value__cost": 0.1, "n_trials": 10},
        {"task_id": "t1", "optimizer_id": "cand", "seed": 2, "trial_value__cost": 0.2, "n_trials": 10},
        {"task_id": "t1", "optimizer_id": "cand", "seed": 3, "trial_value__cost": 0.3, "n_trials": 10},
        {"task_id": "t1", "optimizer_id": "cand", "seed": 4, "trial_value__cost": 0.4, "n_trials": 10},
        # Baseline has seeds 2, 3, 4, 5 (seed 1 missing in base, seed 5 missing in cand)
        {"task_id": "t1", "optimizer_id": "base", "seed": 2, "trial_value__cost": 0.25, "n_trials": 10},
        {"task_id": "t1", "optimizer_id": "base", "seed": 3, "trial_value__cost": 0.35, "n_trials": 10},
        {"task_id": "t1", "optimizer_id": "base", "seed": 4, "trial_value__cost": 0.45, "n_trials": 10},
        {"task_id": "t1", "optimizer_id": "base", "seed": 5, "trial_value__cost": 0.55, "n_trials": 10},
    ])
    terminal = DataLoader.extract_terminal_costs(df_unaligned)
    paired = PairingEngine.pair_seeds(terminal, "cand", "base")
    
    assert "t1" in paired
    c_vals, b_vals = paired["t1"]
    assert len(c_vals) == 3
    assert len(b_vals) == 3
    # Matched seeds are 2, 3, 4
    assert np.allclose(c_vals, [0.2, 0.3, 0.4])
    assert np.allclose(b_vals, [0.25, 0.35, 0.45])


def test_cliffs_delta_and_cohens_d():
    """Test effect size calculations (Cliff's delta and Cohen's d)."""
    x_cand = np.array([1.0, 2.0, 3.0])
    y_base = np.array([2.0, 3.0, 4.0])
    delta = StatisticalMathEngine.calculate_cliffs_delta(x_cand, y_base)
    d_z = StatisticalMathEngine.calculate_cohens_d(x_cand, y_base)
    assert delta < 0  # Cand is lower (better)
    assert d_z < 0


def test_end_to_end_pipeline(synthetic_task_data):
    """End-to-end pipeline: synthetic DataFrame -> TaskWilcoxonResult list and AggregateSummary."""
    terminal_df = DataLoader.extract_terminal_costs(synthetic_task_data)
    paired = PairingEngine.pair_seeds(terminal_df, "Cand_Good", "Base_Ref")
    
    engine = StatisticalAnalysisEngine(alpha=0.05)
    task_results, aggregate = engine.analyze_1v1(paired, "Cand_Good", "Base_Ref")
    
    assert len(task_results) == 2
    res_easy = next(r for r in task_results if r.task_id == "task_easy")
    res_tied = next(r for r in task_results if r.task_id == "task_tied")
    
    # Task easy has significant advantage (lower cost)
    assert res_easy.mean_cand_cost < res_easy.mean_base_cost
    assert res_easy.r_rb < 0
    assert res_easy.decision in ["WIN", "TIE"]
    
    # Task tied is exact tie
    assert res_tied.decision == "TIE"
    assert res_tied.r_rb == 0.0
    assert res_tied.p_raw == 1.0
    
    # Aggregate stats
    assert aggregate.candidate_id == "Cand_Good"
    assert aggregate.baseline_id == "Base_Ref"
    assert aggregate.n_tasks == 2
    assert aggregate.mean_rank_cand <= aggregate.mean_rank_base


def test_exporters(synthetic_task_data, tmp_path):
    """Exporters successfully write Markdown, LaTeX, CSV, and Rich console string without error."""
    terminal_df = DataLoader.extract_terminal_costs(synthetic_task_data)
    paired = PairingEngine.pair_seeds(terminal_df, "Cand_Good", "Base_Ref")
    engine = StatisticalAnalysisEngine(alpha=0.05)
    task_results, aggregate = engine.analyze_1v1(paired, "Cand_Good", "Base_Ref")
    
    exporter = ReportExporter(output_dir=str(tmp_path))
    results_map = {"Cand_Good": (task_results, aggregate)}
    
    # Test file exports
    md_path = exporter.export_markdown(results_map)
    tex_path = exporter.export_latex(results_map)
    csv_path = exporter.export_csv(results_map)
    
    assert os.path.exists(md_path)
    assert os.path.exists(tex_path)
    assert os.path.exists(csv_path)
    
    with open(md_path, "r") as f:
        md_content = f.read()
        assert "task_easy" in md_content
        assert "Cand_Good" in md_content
        
    with open(tex_path, "r") as f:
        tex_content = f.read()
        assert "\\begin{table*}" in tex_content
        assert "\\toprule" in tex_content
        
    with open(csv_path, "r") as f:
        csv_content = f.read()
        assert "candidate_id" in csv_content
        assert "Cand_Good" in csv_content

    # Test Rich console string generation
    rich_output = exporter.render_rich_table(results_map)
    assert isinstance(rich_output, str)
    assert "Cand_Good" in rich_output
    assert "Base_Ref" in rich_output


def test_decision_uses_holm_bonferroni_adjusted_p():
    """Verify that WIN/LOSS decisions strictly use Holm-Bonferroni adjusted p-values (p_holm < alpha)."""
    # Create 3 tasks:
    # Task 1: highly significant win (p_raw very small -> p_holm < 0.05 -> WIN)
    # Task 2: marginal difference with 0.03 < p_raw < 0.05. With 3 tasks, p_holm = 0.03 * 2 = 0.06 > 0.05 -> TIE
    # Task 3: exact tie (p_raw = 1.0)
    paired_data = {
        "task_highly_sig": (
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        ),
        "task_marginal": (
            np.array([0.48, 0.49, 0.47, 0.46, 0.45, 0.44, 0.50, 0.43]),
            np.array([0.52, 0.51, 0.53, 0.54, 0.55, 0.56, 0.50, 0.57])
        ),
        "task_tied": (
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.5, 0.5, 0.5, 0.5])
        )
    }
    
    engine = StatisticalAnalysisEngine(alpha=0.05)
    task_results, aggregate = engine.analyze_1v1(paired_data, "Cand", "Base")
    
    res_dict = {r.task_id: r for r in task_results}
    
    # Check task_highly_sig
    assert res_dict["task_highly_sig"].p_holm < 0.05
    assert res_dict["task_highly_sig"].decision == "WIN"
    
    # Check that any task where p_raw < 0.05 but p_holm >= 0.05 is classified as TIE
    for r in task_results:
        if r.p_holm >= 0.05:
            assert r.decision == "TIE", f"Task {r.task_id} with p_holm={r.p_holm:.4f} >= 0.05 must be TIE, got {r.decision}"
        else:
            assert r.decision in ["WIN", "LOSS"], f"Task {r.task_id} with p_holm={r.p_holm:.4f} < 0.05 must be WIN or LOSS, got {r.decision}"
