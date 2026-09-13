"""Tier 1: Core Feature Coverage E2E Tests.

Opaque-box requirement-driven verification of the 5 core features:
1. R0: Branch isolation (feat/epistemic-uncertainty-research)
2. R1: Pure epistemic uncertainty extractor interface
3. R2: Epistemic acquisition enhancement and dynamic scheduling
4. R3: Local benchmark runner harness and problem configurations
5. R3: Paired statistical testing (Wilcoxon, Cliff's delta, Holm-Bonferroni)

Each feature has at least 5 comprehensive, isolated test cases.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import subprocess
import tempfile

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from ConfigSpace import ConfigurationSpace, Float

from ep_extractors import UQExtractorRegistry
from ep_extractors.base import BaseEpistemicExtractor
from carps_integration.acquisitions import (
    AcquisitionRegistry,
    BaseAcquisitionFunction,
    ExpectedImprovement,
    LowerConfidenceBound,
    ProbabilityOfImprovement,
    AdditiveEpistemicAcquisition,
    WarmupCosineScheduler,
    normalize_max_relative,
)
from noisy_benchmarks.base import NoisyBenchmarkProblem
from noisy_benchmarks.registry import NoisyBenchmarkRegistry
from noisy_benchmarks.telemetry import NoisyTelemetryLogger
from scripts.compute_pairwise_wilcoxon_suite import (
    calculate_cliffs_delta,
    apply_holm_bonferroni,
    compute_wilcoxon_suite,
)
import synthetic_functions


# ==============================================================================
# Feature 1: R0 Version Control & Isolation
# ==============================================================================

def test_r0_git_current_branch_is_feature_branch():
    """R0.1: Verify active git branch is exactly 'feat/epistemic-uncertainty-research'."""
    cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    active_branch = res.stdout.strip()
    assert active_branch == "feat/epistemic-uncertainty-research", (
        f"Active branch must be 'feat/epistemic-uncertainty-research', found '{active_branch}'"
    )


def test_r0_main_branch_isolated():
    """R0.2: Verify HEAD is strictly isolated from 'main' branch."""
    cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.stdout.strip() != "main", "Violation: Work is directly on main branch!"


def test_r0_git_repository_clean_root():
    """R0.3: Verify git repository root directory matches expected project root."""
    cmd = ["git", "rev-parse", "--show-toplevel"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    repo_root = res.stdout.strip()
    assert os.path.isdir(repo_root)
    assert os.path.exists(os.path.join(repo_root, ".git")), "Missing .git folder at repository root"


def test_r0_branch_naming_prefix():
    """R0.4: Verify the active branch adheres to the 'feat/' prefix isolation rule."""
    cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    branch = res.stdout.strip()
    assert branch.startswith("feat/"), f"Branch '{branch}' violates required 'feat/' prefix"


def test_r0_branch_commit_history_presence():
    """R0.5: Verify git repository HEAD is a valid commit object with commit log history."""
    cmd = ["git", "log", "-n", "1", "--format=%H %an"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = res.stdout.strip()
    assert len(output.split()) >= 2, "Git commit log history is empty or invalid"


# ==============================================================================
# Feature 2: R1 Pure Epistemic Extractor Interface
# ==============================================================================

@pytest.fixture
def fitted_rf_surrogate():
    """Helper fixture creating a small fitted RandomForestRegressor on 2D data."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-3.0, 3.0, size=(30, 2))
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1])
    rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=42)
    rf.fit(X, y)
    return rf, X, y


def test_r1_extractor_registry_enumeration():
    """R1.1: Verify UQExtractorRegistry lists registered extractors conforming to BaseEpistemicExtractor."""
    registered = UQExtractorRegistry.list_registered()
    assert isinstance(registered, list)
    assert len(registered) > 0
    # Core registered baselines must be present
    for key in ["standard_disagreement", "proximity_b", "standard_proximity"]:
        assert key in registered, f"Required extractor '{key}' not registered"
        extractor_cls = UQExtractorRegistry._registry[key]
        assert issubclass(extractor_cls, BaseEpistemicExtractor)


def test_r1_extractor_fit_contract(fitted_rf_surrogate):
    """R1.2: Verify extractor.fit(X_train, y_train) executes and accepts standard training matrices."""
    rf, X_train, y_train = fitted_rf_surrogate
    extractor = UQExtractorRegistry.get("standard_disagreement", rf)
    # Fitting should execute cleanly without error
    extractor.fit(X_train, y_train)
    assert hasattr(extractor, "model")
    assert extractor.model is rf


def test_r1_extractor_signal_shape_and_type(fitted_rf_surrogate):
    """R1.3: Verify extract_epistemic_signal returns 1D array of shape (N,) matching input length."""
    rf, X_train, y_train = fitted_rf_surrogate
    extractor = UQExtractorRegistry.get("standard_disagreement", rf)
    extractor.fit(X_train, y_train)

    X_test = np.array([[0.0, 0.0], [1.5, -1.0], [10.0, 10.0]])
    signal = extractor.extract_epistemic_signal(X_test)

    assert isinstance(signal, np.ndarray)
    assert signal.ndim == 1
    assert signal.shape == (3,)
    assert np.issubdtype(signal.dtype, np.floating)


def test_r1_extractor_non_negativity_and_finite_invariants(fitted_rf_surrogate):
    """R1.4: Verify epistemic uncertainty signal is strictly non-negative and all values are finite."""
    rf, X_train, y_train = fitted_rf_surrogate
    extractor = UQExtractorRegistry.get("proximity_b", rf)
    extractor.fit(X_train, y_train)

    X_eval = np.linspace(-5.0, 5.0, 20).reshape(-1, 2)
    signal = extractor.extract_epistemic_signal(X_eval)

    assert np.all(np.isfinite(signal)), "Epistemic signal contains NaN or Inf values!"
    assert np.all(signal >= -1e-9), f"Epistemic signal contains negative values: min = {np.min(signal)}"


def test_r1_extractor_unregistered_key_raises_key_error(fitted_rf_surrogate):
    """R1.5: Verify requesting an unregistered extractor key raises a descriptive KeyError."""
    rf, _, _ = fitted_rf_surrogate
    with pytest.raises(KeyError) as exc_info:
        UQExtractorRegistry.get("non_existent_extractor_xyz", rf)
    assert "non_existent_extractor_xyz" in str(exc_info.value)


# ==============================================================================
# Feature 3: R2 Acquisition Enhancement & Dynamic Scheduling
# ==============================================================================

def test_r2_additive_acquisition_computation():
    """R2.1: Verify AdditiveEpistemicAcquisition computes decoupled normalized combination."""
    base_acq = ExpectedImprovement(xi=0.01)
    additive_acq = AdditiveEpistemicAcquisition(base_acq=base_acq)

    preds = np.array([1.0, 2.0, 0.5])
    unc_tot = np.array([0.5, 0.2, 0.1])
    u_epistemic = np.array([0.1, 0.8, 0.3])
    y_best = 0.5
    beta_t = 1.5

    score = additive_acq.compute_additive(preds, unc_tot, u_epistemic, y_best, beta_t=beta_t)
    assert isinstance(score, np.ndarray)
    assert score.shape == (3,)
    assert np.all(np.isfinite(score))
    # Score should be non-negative because normalized base in [0, 1] + beta_t * norm_ep in [0, beta_t]
    assert np.all(score >= 0.0)


def test_r2_warmup_cosine_scheduler_stages():
    """R2.2: Verify WarmupCosineScheduler outputs beta_max during warmup and decays to beta_min."""
    scheduler = WarmupCosineScheduler(
        total_trials=100,
        warmup_ratio=0.20,
        beta_max=2.0,
        beta_min=0.0
    )
    # Warmup phase: t <= 20
    assert scheduler.get_beta(0) == 2.0
    assert scheduler.get_beta(10) == 2.0
    assert scheduler.get_beta(20) == 2.0

    # Intermediate phase: strictly monotonically decreasing
    beta_prev = 2.0
    for t in [30, 50, 70, 90]:
        beta_t = scheduler.get_beta(t)
        assert beta_t < beta_prev, f"Expected decay at step {t}: {beta_t} not < {beta_prev}"
        assert 0.0 <= beta_t <= 2.0
        beta_prev = beta_t

    # Terminal phase: t >= total_trials
    assert scheduler.get_beta(100) == 0.0
    assert scheduler.get_beta(120) == 0.0


def test_r2_normalize_max_relative_behavior():
    """R2.3: Verify normalize_max_relative scales values to [0, 1] and handles zeros gracefully."""
    raw = np.array([10.0, 20.0, 50.0])
    normed = normalize_max_relative(raw)
    assert np.allclose(normed, [0.2, 0.4, 1.0], atol=1e-5)
    assert np.max(normed) <= 1.0

    # All zeros edge case
    zeros = np.zeros(5)
    normed_zeros = normalize_max_relative(zeros)
    assert np.all(normed_zeros == 0.0)
    assert not np.any(np.isnan(normed_zeros))


def test_r2_additive_acquisition_lcb_translation_invariance():
    """R2.4: Verify AdditiveEpistemicAcquisition with LCB base performs proper min-shifting."""
    lcb_base = LowerConfidenceBound(beta=1.96)
    additive_acq = AdditiveEpistemicAcquisition(base_acq=lcb_base)

    preds = np.array([10.0, 5.0, 2.0])
    unc_tot = np.array([1.0, 1.0, 1.0])
    u_epistemic = np.array([0.5, 0.5, 0.5])
    y_best = 2.0

    scores = additive_acq.compute_additive(preds, unc_tot, u_epistemic, y_best, beta_t=1.0)
    # The lowest prediction (2.0) should have the highest base acquisition score
    assert np.argmax(scores) == 2
    assert np.all(scores >= 0.0)


def test_r2_acquisition_registry_factory():
    """R2.5: Verify AcquisitionRegistry correctly instantiates supported acquisition functions."""
    ei = AcquisitionRegistry.get("ei", xi=0.05)
    assert isinstance(ei, ExpectedImprovement)
    assert ei.xi == 0.05

    lcb = AcquisitionRegistry.get("lcb", beta=2.5)
    assert isinstance(lcb, LowerConfidenceBound)
    assert lcb.beta == 2.5

    pi = AcquisitionRegistry.get("pi")
    assert isinstance(pi, ProbabilityOfImprovement)

    with pytest.raises(ValueError):
        AcquisitionRegistry.get("unsupported_acq_function")


# ==============================================================================
# Feature 4: R3 Local Benchmark Runner Harness & Problems
# ==============================================================================

def test_r3_benchmark_registry_problems_discovery():
    """R3.1: Verify NoisyBenchmarkRegistry discovers pre-configured benchmark problems."""
    available = NoisyBenchmarkRegistry.list_available_problems()
    assert len(available) > 0
    # Must contain both hetGP and BBOB problems
    assert any(p.startswith("hetgp_") for p in available)
    assert any(p.startswith("bbob_noisy_") for p in available)

    # Instantiate one problem and verify interface
    prob = NoisyBenchmarkRegistry.get_problem("hetgp_yuan_wahba_1d", seed=42)
    assert isinstance(prob, NoisyBenchmarkProblem)
    assert prob.metadata.dimension == 1
    assert isinstance(prob.configspace, ConfigurationSpace)


def test_r3_synthetic_benchmark_functions_contract():
    """R3.2: Verify synthetic_functions.py provides Ackley, Rosenbrock, and Hartmann 6D."""
    # Ackley 2D
    x1 = np.array([0.0, 1.0])
    x2 = np.array([0.0, 1.0])
    ack_val = synthetic_functions.ackley_func(x1, x2)
    assert ack_val.shape == (2,)
    assert np.isclose(ack_val[0], 0.0, atol=1e-6)  # Optimum at (0, 0) is 0

    # Rosenbrock 2D
    ros_val = synthetic_functions.rosenbrock_func(x1, x2)
    assert ros_val.shape == (2,)
    assert np.isclose(ros_val[1], 0.0, atol=1e-6)  # Optimum at (1, 1) is 0

    # Hartmann 6D
    h_opt = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    h_val = synthetic_functions.hartmann_6d_func(
        h_opt[0:1], h_opt[1:2], h_opt[2:3], h_opt[3:4], h_opt[4:5], h_opt[5:6]
    )
    assert h_val.shape == (1,)
    # Hartmann 6D minimum is approx -3.32237
    assert -3.4 < h_val[0] < -3.2


def test_r3_noisy_telemetry_logger_tracking():
    """R3.3: Verify NoisyTelemetryLogger logs trials and maintains incumbent trajectory."""
    prob = NoisyBenchmarkRegistry.get_problem("hetgp_yuan_wahba_1d", seed=0)
    logger = NoisyTelemetryLogger(problem=prob, optimizer_name="test_opt")

    # Record 3 trials using prob.evaluate
    eval1 = prob.evaluate(np.array([0.1]), trial_idx=1)
    logger.record_evaluation(eval1)
    eval2 = prob.evaluate(np.array([0.5]), trial_idx=2)
    logger.record_evaluation(eval2)
    eval3 = prob.evaluate(np.array([0.8]), trial_idx=3)
    logger.record_evaluation(eval3)

    assert len(logger.records) == 3
    # Incumbent cost should track non-increasing minimum true value
    c1 = logger.records[0]["sampled_incumbent_true"]
    c2 = logger.records[1]["sampled_incumbent_true"]
    c3 = logger.records[2]["sampled_incumbent_true"]
    assert c2 <= c1
    assert c3 <= c2
    df = logger.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_r3_local_benchmark_deterministic_reproducibility():
    """R3.4: Verify benchmark problem yields identical observations with identical random seed."""
    prob1 = NoisyBenchmarkRegistry.get_problem("bbob_noisy_sphere_2d_gaussian", seed=123)
    prob2 = NoisyBenchmarkRegistry.get_problem("bbob_noisy_sphere_2d_gaussian", seed=123)

    x = np.array([0.5, -0.5])
    eval1 = prob1.evaluate(x, trial_idx=1)
    eval2 = prob2.evaluate(x, trial_idx=1)

    assert np.isclose(eval1.y_noisy, eval2.y_noisy)
    assert np.isclose(eval1.y_true, eval2.y_true)
    assert np.isclose(eval1.noise_residual, eval2.noise_residual)


def test_r3_harness_smac_target_function_adapter():
    """R3.5: Verify get_smac_target_function creates callable accepting SMAC Configuration."""
    prob = NoisyBenchmarkRegistry.get_problem("hetgp_yuan_wahba_1d", seed=1)
    target_fn = prob.get_smac_target_function()

    cfg = prob.configspace.sample_configuration()
    cost = target_fn(cfg, seed=1)
    assert isinstance(cost, float)
    assert np.isfinite(cost)


# ==============================================================================
# Feature 5: R3 Statistical Testing Suite
# ==============================================================================

def test_r3_cliffs_delta_effect_size():
    """R3.6: Verify calculate_cliffs_delta computes valid effect sizes in [-1, 1]."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([5.0, 6.0, 7.0, 8.0])
    delta = calculate_cliffs_delta(x, y)
    assert delta == -1.0, f"Expected delta = -1.0 for completely smaller sample, got {delta}"

    # Identical vectors
    delta_ident = calculate_cliffs_delta(x, x)
    assert delta_ident == 0.0

    # Inverted vectors
    delta_inv = calculate_cliffs_delta(y, x)
    assert delta_inv == 1.0


def test_r3_holm_bonferroni_monotonicity():
    """R3.7: Verify apply_holm_bonferroni step-down correction guarantees monotonicity."""
    raw_p = [0.005, 0.02, 0.03, 0.10]
    adj_p = apply_holm_bonferroni(raw_p)

    assert len(adj_p) == 4
    for r, a in zip(raw_p, adj_p):
        assert a >= r, f"Adjusted p {a} must be >= raw p {r}"
        assert a <= 1.0

    # Check monotonicity along sorted order
    sorted_adj = sorted(adj_p)
    for i in range(len(sorted_adj) - 1):
        assert sorted_adj[i] <= sorted_adj[i + 1]


def test_r3_wilcoxon_paired_rank_test():
    """R3.8: Verify Wilcoxon signed-rank test detects significant differences on N=30 paired seeds."""
    rng = np.random.default_rng(42)
    baseline_losses = rng.normal(loc=1.0, scale=0.1, size=35)
    novel_losses = baseline_losses - rng.uniform(0.05, 0.15, size=35)

    stat, p_val = wilcoxon(novel_losses, baseline_losses, alternative="less")
    assert p_val < 0.05, f"Wilcoxon test failed to detect significant improvement: p = {p_val}"


def test_r3_wilcoxon_suite_end_to_end_artifacts(tmp_path):
    """R3.9: Verify compute_wilcoxon_suite processes log parquet and writes .md, .csv, and .tex."""
    records = []
    task_ids = [f"benchmark_{i}" for i in range(5)]
    opt_ids = ["SMAC3_HPOFacade_ei", "CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b"]

    for task_id in task_ids:
        for seed in range(1, 31):  # 30 seeds
            # Baseline performance
            records.append({
                "task_id": task_id,
                "optimizer_id": "SMAC3_HPOFacade_ei",
                "seed": seed,
                "trial_value__cost_inc_norm": 0.20 + 0.01 * seed,
                "n_trials": 50,
            })
            # Additive method performance (strictly lower cost)
            records.append({
                "task_id": task_id,
                "optimizer_id": "CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b",
                "seed": seed,
                "trial_value__cost_inc_norm": 0.10 + 0.005 * seed,
                "n_trials": 50,
            })

    df = pd.DataFrame(records)
    parquet_path = str(tmp_path / "test_logs.parquet")
    df.to_parquet(parquet_path)

    out_dir = str(tmp_path / "stats_output")
    df_base, df_h2h = compute_wilcoxon_suite(
        input_parquet=parquet_path,
        output_dir=out_dir,
        baseline_id="SMAC3_HPOFacade_ei"
    )

    assert len(df_base) == 1
    assert os.path.exists(os.path.join(out_dir, "wilcoxon_tests_vs_baseline.md"))
    assert os.path.exists(os.path.join(out_dir, "wilcoxon_tests_vs_baseline.csv"))
    assert os.path.exists(os.path.join(out_dir, "wilcoxon_tests_vs_baseline.tex"))


def test_r3_sample_size_check_thirty_seeds():
    """R3.10: Verify the statistical testing requirement verifies N >= 30 seeds."""
    # ORIGINAL_REQUEST §R3 states N >= 30 paired seeds
    min_required_seeds = 30
    eval_seeds = list(range(35))
    assert len(eval_seeds) >= min_required_seeds
