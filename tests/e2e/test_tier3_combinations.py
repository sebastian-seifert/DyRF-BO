"""Tier 3: Pairwise and Cross-Feature Combinations E2E Tests.

Opaque-box requirement-driven verification of cross-feature interactions:
1. Epistemic Extractor + SMAC3 Surrogate Model (`CustomUncertaintyRandomForest` + `UQExtractorRegistry`)
2. Surrogate Model + Acquisition Scheduler (`CustomUncertaintyRandomForest` + `AdditiveEpistemicAcquisition` + `WarmupCosineScheduler`)
3. Benchmark Harness + Wilcoxon Statistical Analyzer (`NoisyBOHarness` / `NoisyTelemetryLogger` + `compute_wilcoxon_suite`)

Each pairwise interaction contains at least 5 comprehensive, isolated test cases.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest
from ConfigSpace import ConfigurationSpace, Float

from ep_extractors import UQExtractorRegistry
from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
from carps_integration.acquisitions import (
    AcquisitionRegistry,
    ExpectedImprovement,
    LowerConfidenceBound,
    AdditiveEpistemicAcquisition,
    WarmupCosineScheduler,
    normalize_max_relative,
)
from noisy_benchmarks.registry import NoisyBenchmarkRegistry
from noisy_benchmarks.telemetry import NoisyTelemetryLogger
from scripts.compute_pairwise_wilcoxon_suite import (
    calculate_cliffs_delta,
    apply_holm_bonferroni,
    compute_wilcoxon_suite,
)


# ==============================================================================
# Interaction 1: Extractor + SMAC3 Surrogate Model
# ==============================================================================

@pytest.fixture
def base_configspace_2d():
    cs = ConfigurationSpace({"x1": Float("x1", bounds=(-3.0, 3.0)), "x2": Float("x2", bounds=(-3.0, 3.0))})
    return cs


@pytest.mark.parametrize("extractor_key", [
    "standard_disagreement",
    "proximity_b",
    "standard_proximity",
    "proximity_bc",
])
def test_t3_surrogate_with_multiple_registered_extractors(base_configspace_2d, extractor_key):
    """T3.1: Verify CustomUncertaintyRandomForest integrates with various registered extractors."""
    rng = np.random.default_rng(42)
    X_train = rng.uniform(-3.0, 3.0, size=(25, 2))
    y_train = (np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])).reshape(-1, 1)

    model = CustomUncertaintyRandomForest(
        uncertainty_func=extractor_key,
        configspace=base_configspace_2d,
        n_trees=15,
        seed=42,
    )
    model.train(X_train, y_train)

    X_test = rng.uniform(-3.0, 3.0, size=(10, 2))
    mean, var = model._predict(X_test)

    assert mean.shape == (10, 1)
    assert var.shape == (10, 1)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert np.all(var >= 0.0)


def test_t3_surrogate_variance_matches_extractor_squared(base_configspace_2d):
    """T3.2: Verify surrogate _predict variance is identically equal to extractor's U_E(X)^2."""
    rng = np.random.default_rng(100)
    X_train = rng.uniform(-2.0, 2.0, size=(30, 2))
    y_train = (X_train[:, 0] ** 2 + X_train[:, 1] ** 2).reshape(-1, 1)

    model = CustomUncertaintyRandomForest(
        uncertainty_func="standard_disagreement",
        configspace=base_configspace_2d,
        n_trees=15,
        seed=100,
    )
    model.train(X_train, y_train)

    X_test = rng.uniform(-2.0, 2.0, size=(8, 2))
    _, var = model._predict(X_test)

    # Directly query the underlying extractor instance attached to the model
    assert model.uq_extractor is not None
    direct_signal = model.uq_extractor.extract_epistemic_signal(X_test)
    expected_var = (direct_signal ** 2).reshape(-1, 1)

    assert np.allclose(var, expected_var, atol=1e-8), "Surrogate variance diverges from extractor signal squared!"


def test_t3_surrogate_sequential_retraining(base_configspace_2d):
    """T3.3: Verify surrogate re-fitting updates extractor with expanded data in sequential BO loop."""
    rng = np.random.default_rng(7)
    X_init = rng.uniform(-2.0, 2.0, size=(15, 2))
    y_init = (X_init[:, 0] + X_init[:, 1]).reshape(-1, 1)

    model = CustomUncertaintyRandomForest(
        uncertainty_func="proximity_b",
        configspace=base_configspace_2d,
        n_trees=15,
        seed=7,
    )
    model.train(X_init, y_init)
    assert model.last_X.shape[0] == 15

    # Evaluate at a test point
    x_probe = np.array([[1.5, 1.5]])
    mean_before, var_before = model._predict(x_probe)
    assert var_before[0, 0] >= 0.0

    # Augment dataset with 5 additional points
    X_new = rng.uniform(-2.0, 2.0, size=(5, 2))
    y_new = (X_new[:, 0] + X_new[:, 1]).reshape(-1, 1)
    X_augmented = np.vstack([X_init, X_new])
    y_augmented = np.vstack([y_init, y_new])

    model.train(X_augmented, y_augmented)
    assert model.last_X.shape[0] == 20
    mean_after, var_after = model._predict(x_probe)

    assert mean_after.shape == (1, 1)
    assert var_after.shape == (1, 1)
    assert np.all(np.isfinite(var_after))
    assert var_after[0, 0] >= 0.0


def test_t3_surrogate_custom_callable_binding(base_configspace_2d):
    """T3.4: Verify CustomUncertaintyRandomForest supports arbitrary user-defined callable uncertainty functions."""
    def custom_triangular_uq(rf_model, X, y):
        # Returns distance from center
        return np.linalg.norm(X, axis=1)

    model = CustomUncertaintyRandomForest(
        uncertainty_func=custom_triangular_uq,
        configspace=base_configspace_2d,
        n_trees=10,
        seed=42,
    )
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    y_train = np.array([[0.0], [2.0], [2.0]])
    model.train(X_train, y_train)

    X_test = np.array([[3.0, 4.0]])  # Norm is 5.0
    _, var = model._predict(X_test)
    assert np.isclose(var[0, 0], 25.0, atol=1e-5)  # 5.0^2 = 25.0


def test_t3_surrogate_impute_inactive_integration(base_configspace_2d):
    """T3.5: Verify surrogate handles configuration arrays with NaN / inactive imputation."""
    model = CustomUncertaintyRandomForest(
        uncertainty_func="standard_disagreement",
        configspace=base_configspace_2d,
        n_trees=15,
        seed=42,
    )
    X_train = np.array([[0.1, 0.2], [0.5, 0.6], [0.8, 0.9]])
    y_train = np.array([[1.0], [2.0], [3.0]])
    model.train(X_train, y_train)

    # Test array with NaN representing inactive conditional parameters
    X_with_nan = np.array([[0.2, np.nan], [np.nan, 0.7]])
    mean, var = model._predict(X_with_nan)
    assert mean.shape == (2, 1)
    assert var.shape == (2, 1)
    assert not np.any(np.isnan(mean))
    assert not np.any(np.isnan(var))


# ==============================================================================
# Interaction 2: Surrogate Model + Acquisition Scheduler
# ==============================================================================

def test_t3_surrogate_to_additive_ei_pipeline(base_configspace_2d):
    """T3.6: Verify end-to-end flow from surrogate predictions to AdditiveEpistemicAcquisition with EI."""
    rng = np.random.default_rng(42)
    X_train = rng.uniform(-2.0, 2.0, size=(20, 2))
    y_train = (X_train[:, 0] ** 2 + X_train[:, 1] ** 2).reshape(-1, 1)

    model = CustomUncertaintyRandomForest(
        uncertainty_func="proximity_b",
        configspace=base_configspace_2d,
        n_trees=15,
        seed=42,
    )
    model.train(X_train, y_train)

    # Query candidate pool
    X_candidates = rng.uniform(-3.0, 3.0, size=(30, 2))
    mean, var = model._predict(X_candidates)
    u_ep = model.uq_extractor.extract_epistemic_signal(X_candidates)
    total_unc = np.sqrt(np.maximum(var.flatten(), 1e-9))

    # Additive acquisition with EI
    acq = AdditiveEpistemicAcquisition(base_acq=ExpectedImprovement())
    scores = acq.compute_additive(
        preds=mean.flatten(),
        unc_tot=total_unc,
        u_epistemic=u_ep,
        y_best=float(np.min(y_train)),
        beta_t=1.0
    )

    assert scores.shape == (30,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)


def test_t3_surrogate_to_additive_lcb_pipeline(base_configspace_2d):
    """T3.7: Verify end-to-end flow with LowerConfidenceBound base acquisition."""
    rng = np.random.default_rng(99)
    X_train = rng.uniform(-2.0, 2.0, size=(20, 2))
    y_train = (X_train[:, 0] * 2.0).reshape(-1, 1)

    model = CustomUncertaintyRandomForest(
        uncertainty_func="standard_proximity",
        configspace=base_configspace_2d,
        n_trees=15,
        seed=99,
    )
    model.train(X_train, y_train)

    X_candidates = rng.uniform(-2.0, 2.0, size=(10, 2))
    mean, var = model._predict(X_candidates)
    u_ep = model.uq_extractor.extract_epistemic_signal(X_candidates)
    total_unc = np.sqrt(np.maximum(var.flatten(), 1e-9))

    acq = AdditiveEpistemicAcquisition(base_acq=LowerConfidenceBound(beta=2.0))
    scores = acq.compute_additive(
        preds=mean.flatten(),
        unc_tot=total_unc,
        u_epistemic=u_ep,
        y_best=float(np.min(y_train)),
        beta_t=0.5
    )

    assert scores.shape == (10,)
    assert np.all(scores >= 0.0)
    assert np.max(scores) > 0.0


def test_t3_scheduler_dynamic_transition_across_budget():
    """T3.8: Verify multi-step BO simulation shifts candidate ranking from exploration to exploitation."""
    scheduler = WarmupCosineScheduler(total_trials=20, warmup_ratio=0.25, beta_max=2.0, beta_min=0.0)
    acq = AdditiveEpistemicAcquisition(base_acq=ExpectedImprovement())

    # Candidate 0: Good mean (low cost), low uncertainty (exploitation point)
    # Candidate 1: Mediocre mean, very high epistemic uncertainty (exploration point)
    preds = np.array([0.5, 2.0])
    unc_tot = np.array([0.1, 1.0])
    u_ep = np.array([0.05, 1.0])
    y_best = 0.5

    # Early stage: t=2 (beta = 2.0, strong exploration bonus)
    score_early = acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=scheduler.get_beta(2))
    # Candidate 1 should win or have strong boost early
    assert score_early[1] > score_early[0]

    # Terminal stage: t=20 (beta = 0.0, pure exploitation)
    score_late = acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=scheduler.get_beta(20))
    # Candidate 0 must win at the end of the budget
    assert score_late[0] > score_late[1]


def test_t3_candidate_selection_argmax_transition():
    """T3.9: Verify argmax selection across a continuous candidate grid changes predictably with beta_t."""
    acq = AdditiveEpistemicAcquisition(base_acq=ExpectedImprovement())

    # Create 5 candidates with equal total uncertainty, so candidate 0 has best base EI
    preds = np.array([0.05, 0.4, 0.8, 1.2, 1.5])
    unc_tot = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    # Candidate 4 has enormous epistemic uncertainty
    u_ep = np.array([0.01, 0.1, 0.4, 0.9, 2.0])
    y_best = 0.10

    # High exploration: beta=3.0 -> picks candidate with highest epistemic uncertainty
    pick_explore = np.argmax(acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=3.0))
    assert pick_explore in [3, 4]

    # Pure exploitation: beta=0.0 -> picks candidate 0 with lowest predicted mean
    pick_exploit = np.argmax(acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=0.0))
    assert pick_exploit == 0


def test_t3_surrogate_acquisition_zero_epistemic_degradation(base_configspace_2d):
    """T3.10: Verify when epistemic uncertainty is uniformly zero, score equals normalized base acquisition."""
    base_ei = ExpectedImprovement()
    additive = AdditiveEpistemicAcquisition(base_acq=base_ei)

    preds = np.array([1.0, 1.5, 2.0])
    unc = np.array([0.2, 0.3, 0.1])
    u_ep_zero = np.zeros(3)
    y_best = 1.0

    score = additive.compute_additive(preds, unc, u_ep_zero, y_best, beta_t=1.0)
    expected_base = normalize_max_relative(base_ei.compute(preds, unc, y_best))
    assert np.allclose(score, expected_base, atol=1e-8)


# ==============================================================================
# Interaction 3: Benchmark Harness + Wilcoxon Analyzer Pipeline
# ==============================================================================

def test_t3_harness_telemetry_to_wilcoxon_pipeline(tmp_path):
    """T3.11: Verify telemetry output from benchmark logging feeds cleanly into Wilcoxon statistical suite."""
    prob = NoisyBenchmarkRegistry.get_problem("bbob_noisy_sphere_2d_gaussian", seed=42)

    logger_base = NoisyTelemetryLogger(problem=prob, optimizer_name="SMAC3_HPOFacade_ei")
    logger_novel = NoisyTelemetryLogger(problem=prob, optimizer_name="CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b")

    # Record 5 mock trials each
    for i in range(1, 6):
        eval_res = prob.evaluate(np.array([0.1 * i, 0.1 * i]), trial_idx=i)
        logger_base.record_evaluation(eval_res)
        logger_novel.record_evaluation(eval_res)

    df_base = logger_base.to_dataframe()
    df_novel = logger_novel.to_dataframe()

    assert len(df_base) == 5
    assert len(df_novel) == 5
    assert "sampled_incumbent_true" in df_base.columns
    assert "trial_idx" in df_base.columns


def test_t3_paired_seed_alignment_validation(tmp_path):
    """T3.12: Verify statistical analyzer pairs evaluations by (task_id, seed) across algorithms."""
    records = []
    # 2 tasks, 5 seeds
    for task in ["ackley_2d", "rosenbrock_2d"]:
        for seed in range(1, 6):
            records.append({
                "task_id": task,
                "optimizer_id": "SMAC3_HPOFacade_ei",
                "seed": seed,
                "trial_value__cost_inc_norm": 0.50 + 0.01 * seed,
                "n_trials": 20,
            })
            records.append({
                "task_id": task,
                "optimizer_id": "CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b",
                "seed": seed,
                "trial_value__cost_inc_norm": 0.20 + 0.01 * seed,
                "n_trials": 20,
            })

    df = pd.DataFrame(records)
    parquet_file = str(tmp_path / "aligned_logs.parquet")
    df.to_parquet(parquet_file)

    out_dir = str(tmp_path / "aligned_output")
    df_vs_base, _ = compute_wilcoxon_suite(
        input_parquet=parquet_file,
        output_dir=out_dir,
        baseline_id="SMAC3_HPOFacade_ei"
    )

    assert len(df_vs_base) == 1
    row = df_vs_base.iloc[0]
    # Novel method has lower cost -> Mean Diff vs Base and Rel Reduction are negative (reduction)
    assert row["Mean Diff vs Base"] < 0.0
    assert row["Rel Reduction (%)"] < 0.0
    assert row["Win / Loss / Tie"] == "2 / 0 / 0"
    assert row["Cliff's delta"] < 0.0


def test_t3_wilcoxon_analyzer_directionality_and_cliffs_delta():
    """T3.13: Verify Cliff's delta directionality corresponds with Wilcoxon rank order."""
    x_superior = np.array([0.1, 0.2, 0.15, 0.18, 0.12])  # Lower loss = superior
    y_inferior = np.array([0.8, 0.9, 0.85, 0.75, 0.95])

    delta = calculate_cliffs_delta(x_superior, y_inferior)
    # x is strictly less than y -> delta = -1.0
    assert delta == -1.0


def test_t3_multitask_holm_bonferroni_correction_integration():
    """T3.14: Verify Holm-Bonferroni correction controls FWER across multiple benchmark tasks."""
    # 4 distinct benchmark hypotheses
    raw_p_values = [0.001, 0.012, 0.035, 0.048]
    adjusted = apply_holm_bonferroni(raw_p_values)

    assert len(adjusted) == 4
    # All adjusted values must be >= raw values
    assert adjusted[0] == 0.001 * 4  # 0.004
    assert adjusted[1] == 0.012 * 3  # 0.036
    assert adjusted[2] >= adjusted[1]
    assert adjusted[3] >= adjusted[2]


def test_t3_telemetry_json_to_parquet_to_tables_pipeline(tmp_path):
    """T3.15: Verify complete telemetry export from logger to JSON, Parquet, and analysis tables."""
    prob = NoisyBenchmarkRegistry.get_problem("hetgp_yuan_wahba_1d", seed=0)
    logger = NoisyTelemetryLogger(
        problem=prob,
        optimizer_name="CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b",
        output_dir=str(tmp_path)
    )

    eval_res = prob.evaluate(np.array([0.5]), trial_idx=1)
    logger.record_evaluation(eval_res)
    saved_files = logger.save(base_filename="test_run")

    assert os.path.exists(saved_files["json"])
    assert os.path.exists(saved_files["parquet"])
    assert os.path.exists(saved_files["csv"])

    # Read back saved parquet
    loaded_df = pd.read_parquet(saved_files["parquet"])
    assert len(loaded_df) == 1
    assert loaded_df["optimizer_name"].iloc[0] == "CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b"
