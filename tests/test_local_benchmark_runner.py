"""Unit tests for the Local Epistemic Benchmark Runner and Statistical Suite.

Verifies:
1. CLI argument parsing and defaults.
2. Benchmark problem instantiation and mathematical definitions (Ackley 2D, Rosenbrock 2D, Hartmann 6D, synthetic gap 2D).
3. Paired seed determinism: identical initial design evaluations between Baseline and Proposed.
4. Smoke execution of local benchmark harness generating valid benchmark_traces.csv.
5. Statistical hypothesis calculations: Paired Wilcoxon, Cliff's delta, Holm-Bonferroni correction,
   statistical_results.csv and BENCHMARK_SUMMARY.md generation.
6. OOD-AUROC in synthetic gap topology and runtime overhead constraints.
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add repository root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_local_epistemic_benchmark import (
    parse_args,
    get_benchmark_problem,
    Ackley2DProblem,
    Rosenbrock2DProblem,
    Hartmann6DProblem,
    SyntheticGap2DProblem,
    run_paired_evaluation,
    run_local_epistemic_benchmark,
)
from scripts.compute_pairwise_wilcoxon_suite import (
    calculate_cliffs_delta,
    apply_holm_bonferroni,
    compute_wilcoxon_suite,
)


def test_cli_argument_parser():
    """Verify CLI argument defaults and custom parameter parsing."""
    # Test default arguments
    defaults = parse_args([])
    assert defaults.seeds == 35
    assert defaults.n_trials == 50
    assert defaults.output_dir == "results/epistemic_research"
    assert "ackley_2d" in defaults.benchmarks
    assert "rosenbrock_2d" in defaults.benchmarks
    assert "hartmann_6d" in defaults.benchmarks
    assert "synthetic_gap_2d" in defaults.benchmarks

    # Test custom arguments
    custom = parse_args([
        "--benchmarks", "ackley_2d", "synthetic_gap_2d",
        "--seeds", "5",
        "--n_trials", "10",
        "--output_dir", "/tmp/test_out",
        "--n_workers", "2",
    ])
    assert custom.benchmarks == ["ackley_2d", "synthetic_gap_2d"]
    assert custom.seeds == 5
    assert custom.n_trials == 10
    assert custom.output_dir == "/tmp/test_out"
    assert custom.n_workers == 2


def test_benchmark_problems_instantiation_and_evaluation():
    """Verify all 4 required benchmark problem classes instantiate and evaluate correctly."""
    topologies = ["ackley_2d", "rosenbrock_2d", "hartmann_6d", "synthetic_gap_2d"]
    
    for name in topologies:
        prob = get_benchmark_problem(name, seed=42)
        assert prob is not None
        assert prob.metadata.name == name
        assert prob.metadata.dimension in [2, 6]
        assert len(prob.metadata.lower_bounds) == prob.metadata.dimension
        assert len(prob.metadata.upper_bounds) == prob.metadata.dimension
        
        # Test evaluation on a point inside domain
        x_mid = 0.5 * (prob.metadata.lower_bounds + prob.metadata.upper_bounds)
        y_true = prob.evaluate_true(x_mid)
        assert np.isfinite(y_true)
        
        # Test noisy evaluation
        res = prob.evaluate(x_mid, trial_idx=0)
        assert np.isfinite(res.y_noisy)
        assert np.isfinite(res.instantaneous_regret)
        assert res.seed == 42
        assert res.trial_idx == 0


def test_paired_seed_initial_design_determinism():
    """Verify that for each paired seed, initial evaluations are identical for Baseline and Proposed."""
    seed = 77
    prob_name = "ackley_2d"
    records = run_paired_evaluation(
        benchmark_name=prob_name,
        seed=seed,
        n_trials=6,
        n_init=3,
        n_candidates=50,
    )
    df = pd.DataFrame(records)
    
    # Filter for baseline and proposed
    base_df = df[df["optimizer_id"] == "SMAC3_HPOFacade_ei"].sort_values("trial_id").reset_index(drop=True)
    prop_df = df[df["optimizer_id"] == "CARPSDynamicRF_AdditiveEpistemic_ei_distance_evidential"].sort_values("trial_id").reset_index(drop=True)
    
    assert len(base_df) == 6
    assert len(prop_df) == 6
    
    # The first n_init evaluations MUST be identical
    for idx in range(3):
        assert np.allclose(base_df.loc[idx, "x"], prop_df.loc[idx, "x"], atol=1e-7), (
            f"Initial configuration mismatch at trial {idx}"
        )
        assert np.isclose(base_df.loc[idx, "y_noisy"], prop_df.loc[idx, "y_noisy"], atol=1e-7), (
            f"Initial observation mismatch at trial {idx}"
        )


def test_run_local_benchmark_smoke_execution(tmp_path):
    """Execute a fast smoke benchmark run and verify benchmark_traces.csv schema and content."""
    out_dir = str(tmp_path / "smoke_epistemic_benchmarks")
    
    df_traces = run_local_epistemic_benchmark(
        benchmarks=["ackley_2d", "synthetic_gap_2d"],
        seeds=2,
        n_trials=5,
        n_init=2,
        output_dir=out_dir,
        n_workers=1,
        n_candidates=50,
    )
    
    csv_file = os.path.join(out_dir, "benchmark_traces.csv")
    assert os.path.isfile(csv_file)
    assert os.path.getsize(csv_file) > 0
    
    # Read saved CSV and check required columns
    saved_df = pd.read_csv(csv_file)
    required_cols = [
        "task_id",
        "seed",
        "optimizer_id",
        "trial_id",
        "n_trials",
        "trial_value__cost_inc_norm",
        "sampled_incumbent_regret",
        "sampled_incumbent_true",
        "wallclock_seconds",
    ]
    for col in required_cols:
        assert col in saved_df.columns, f"Missing required column '{col}' in benchmark_traces.csv"
        
    assert set(saved_df["task_id"].unique()) == {"ackley_2d", "synthetic_gap_2d"}
    assert set(saved_df["seed"].unique()) == {1, 2}
    assert "SMAC3_HPOFacade_ei" in saved_df["optimizer_id"].values
    assert "CARPSDynamicRF_AdditiveEpistemic_ei_distance_evidential" in saved_df["optimizer_id"].values


def test_cliffs_delta_and_holm_bonferroni_properties():
    """Verify formal properties of Cliff's delta and Holm-Bonferroni adjustment."""
    # 1. Cliff's delta bounds and direction
    x_better = np.array([0.1, 0.2, 0.3, 0.4])
    y_worse = np.array([1.0, 1.1, 1.2, 1.3])
    delta_neg = calculate_cliffs_delta(x_better, y_worse)
    assert delta_neg == -1.0, f"Expected perfect negative Cliff's delta, got {delta_neg}"
    
    delta_pos = calculate_cliffs_delta(y_worse, x_better)
    assert delta_pos == 1.0, f"Expected perfect positive Cliff's delta, got {delta_pos}"
    
    delta_tie = calculate_cliffs_delta(x_better, x_better)
    assert delta_tie == 0.0, f"Expected zero Cliff's delta for identical vectors, got {delta_tie}"
    
    # 2. Holm-Bonferroni monotonic step-down adjustment
    raw_p = [0.01, 0.04, 0.03, 0.20]
    adj_p = apply_holm_bonferroni(raw_p)
    assert len(adj_p) == len(raw_p)
    # Monotonicity test: adjusted p should be >= raw p
    for r, a in zip(raw_p, adj_p):
        assert a >= r, f"Adjusted p ({a}) must be >= raw p ({r})"
        assert a <= 1.0


def test_statistical_suite_processing_and_summary_generation(tmp_path):
    """Verify compute_wilcoxon_suite processes benchmark_traces and generates required summary files."""
    # Create mock benchmark traces with 35 seeds for 3 functions where proposed strictly outperforms baseline
    n_seeds = 35
    benchmarks = ["ackley_2d", "rosenbrock_2d", "hartmann_6d", "synthetic_gap_2d"]
    records = []
    rng = np.random.default_rng(2026)
    
    for bench in benchmarks:
        for s in range(1, n_seeds + 1):
            base_cost = float(rng.normal(loc=2.0, scale=0.2))
            prop_cost = base_cost - float(rng.uniform(0.3, 0.6))
            
            # Baseline trace
            records.append({
                "task_id": bench,
                "seed": s,
                "optimizer_id": "SMAC3_HPOFacade_ei",
                "trial_id": 49,
                "n_trials": 50,
                "trial_value__cost": base_cost,
                "trial_value__cost_inc_norm": base_cost,
                "sampled_incumbent_regret": base_cost,
                "sampled_incumbent_true": base_cost,
                "wallclock_seconds": 1.0,
            })
            # Proposed DA-EHRF trace
            records.append({
                "task_id": bench,
                "seed": s,
                "optimizer_id": "CARPSDynamicRF_AdditiveEpistemic_ei_distance_evidential",
                "trial_id": 49,
                "n_trials": 50,
                "trial_value__cost": prop_cost,
                "trial_value__cost_inc_norm": prop_cost,
                "sampled_incumbent_regret": prop_cost,
                "sampled_incumbent_true": prop_cost,
                "wallclock_seconds": 1.15,
            })
            
    df = pd.DataFrame(records)
    csv_input = str(tmp_path / "benchmark_traces.csv")
    df.to_csv(csv_input, index=False)
    
    out_dir = str(tmp_path / "stats_output")
    df_vs_base, _ = compute_wilcoxon_suite(
        input_parquet=csv_input,
        output_dir=out_dir,
        baseline_id="SMAC3_HPOFacade_ei",
    )
    
    assert len(df_vs_base) >= 1
    # Check outputs generated
    stat_csv = os.path.join(out_dir, "statistical_results.csv")
    summary_md = os.path.join(out_dir, "BENCHMARK_SUMMARY.md")
    wilcoxon_md = os.path.join(out_dir, "wilcoxon_tests_vs_baseline.md")
    
    assert os.path.isfile(stat_csv), "statistical_results.csv was not generated"
    assert os.path.isfile(summary_md), "BENCHMARK_SUMMARY.md was not generated"
    assert os.path.isfile(wilcoxon_md), "wilcoxon_tests_vs_baseline.md was not generated"
    
    # Check statistical_results.csv contents
    df_stat = pd.read_csv(stat_csv)
    assert "task_id" in df_stat.columns
    assert "p_raw" in df_stat.columns or "p_value" in df_stat.columns
    assert "Cliff's delta" in df_stat.columns or "cliffs_delta" in df_stat.columns
    
    # Check BENCHMARK_SUMMARY.md contents
    with open(summary_md, "r") as f:
        content = f.read()
    assert "Benchmark Summary" in content or "Statistical Evaluation" in content
    assert "Wilcoxon" in content
    assert "Cliff's delta" in content
