"""Tier 2: Boundary and Corner Case E2E Tests.

Opaque-box requirement-driven verification of 5 boundary categories:
1. Zero Noise (deterministic limit, duplicates, flat surfaces, zero variance)
2. Extreme Extrapolation Gap (far OOD, empty gaps, asymptotic bounds, disjoint clusters)
3. High Dimensions (D=20..100, N << D regime, irrelevant features, distance scaling)
4. Single-Sample Leaf / Degenerate Trees (N=1, n_estimators=1, deep unconstrained trees)
5. NaN/Inf Bounds & Numerical Stability (nan/inf handling, extreme scales 10^-8 to 10^8, dtype invariance)

Each category contains at least 5 comprehensive, isolated test cases.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from ConfigSpace import ConfigurationSpace, Float

from ep_extractors import UQExtractorRegistry
from carps_integration.acquisitions import (
    ExpectedImprovement,
    LowerConfidenceBound,
    AdditiveEpistemicAcquisition,
    WarmupCosineScheduler,
    normalize_max_relative,
)
from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest


# ==============================================================================
# Boundary 1: Zero Noise & Deterministic Limit
# ==============================================================================

def test_b1_zero_noise_constant_flat_surface():
    """B1.1: Verify extractor on flat constant target (y_i = c) produces finite non-negative uncertainty."""
    X_train = np.linspace(-5.0, 5.0, 20).reshape(-1, 1)
    y_train = np.full(len(X_train), 42.0)

    rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_disagreement", rf)
    extractor.fit(X_train, y_train)

    X_test = np.array([[-10.0], [0.0], [10.0]])
    signal = extractor.extract_epistemic_signal(X_test)

    assert np.all(np.isfinite(signal)), "Signal contains NaN/Inf on flat target"
    assert np.all(signal >= 0.0), "Signal contains negative values"
    # For identical constant targets across all trees, disagreement is zero
    assert np.allclose(signal, 0.0, atol=1e-6)


def test_b1_zero_noise_duplicate_points():
    """B1.2: Verify extractor handles exact duplicate observations without numerical collapse."""
    X_train = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]])
    y_train = np.array([5.0, 5.0, 5.0, 10.0, 10.0])

    rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_proximity", rf)
    extractor.fit(X_train, y_train)

    signal = extractor.extract_epistemic_signal(np.array([[1.0, 2.0], [2.0, 3.0], [10.0, 10.0]]))
    assert np.all(np.isfinite(signal))
    assert np.all(signal >= 0.0)


def test_b1_zero_noise_acquisition_finite():
    """B1.3: Verify acquisition functions handle near-zero uncertainty without NaN/Inf."""
    ei = ExpectedImprovement(xi=0.0)
    preds = np.array([1.0, 2.0, 0.5])
    unc_zero = np.array([1e-15, 0.0, 1e-12])
    y_best = 1.0

    scores = ei.compute(preds, unc_zero, y_best)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    # Candidate 2 with mean 0.5 < y_best 1.0 should have deterministic improvement 0.5
    assert np.isclose(scores[2], 0.5, atol=1e-5)


def test_b1_zero_noise_scheduler_terminal_decay():
    """B1.4: Verify additive acquisition with beta_min = 0.0 decouples epistemic bonus at budget limit."""
    base_acq = ExpectedImprovement()
    additive = AdditiveEpistemicAcquisition(base_acq=base_acq)

    preds = np.array([1.0, 2.0])
    unc = np.array([0.5, 0.5])
    u_ep = np.array([10.0, 100.0])  # Disproportionate epistemic signal
    y_best = 1.0

    # At beta_t = 0.0, epistemic bonus must not alter relative base ranking
    score_zero = additive.compute_additive(preds, unc, u_ep, y_best, beta_t=0.0)
    score_base = normalize_max_relative(base_acq.compute(preds, unc, y_best))
    assert np.allclose(score_zero, score_base, atol=1e-7)


def test_b1_zero_noise_deterministic_noiseless_target():
    """B1.5: Fit CustomUncertaintyRandomForest on noiseless target and verify finite variance."""
    cs = ConfigurationSpace({"x": Float("x", bounds=(-3.0, 3.0))})
    X_train = np.linspace(-3.0, 3.0, 25).reshape(-1, 1)
    y_train = (X_train[:, 0] ** 2).reshape(-1, 1)

    model = CustomUncertaintyRandomForest(
        uncertainty_func="standard_disagreement",
        configspace=cs,
        n_trees=10,
        seed=42,
    )
    model.train(X_train, y_train)

    X_test = np.linspace(-4.0, 4.0, 15).reshape(-1, 1)
    mu, var = model._predict(X_test)
    assert mu.shape == (15, 1)
    assert var.shape == (15, 1)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(var))
    assert np.all(var >= 0.0)


# ==============================================================================
# Boundary 2: Extreme Extrapolation Gap
# ==============================================================================

def test_b2_extreme_far_ood_bounded():
    """B2.1: Verify query points at 10x, 100x, 1000x distance remain bounded and non-overflowing."""
    X_train = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y_train = np.sin(X_train[:, 0])

    rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("proximity_b", rf)
    extractor.fit(X_train, y_train)

    X_far = np.array([[10.0], [100.0], [1000.0], [-1000.0]])
    signal = extractor.extract_epistemic_signal(X_far)

    assert np.all(np.isfinite(signal))
    assert np.all(signal >= 0.0)
    assert np.all(signal < 1e12), "Uncertainty exploded to unphysically large values"


def test_b2_empty_gap_uncertainty_higher_than_in_distribution():
    """B2.2: Verify that in a synthetic gap [3, 7], the gap midpoint has higher uncertainty than sampled points."""
    # Sample points only in [0, 3] and [7, 10]
    X_left = np.linspace(0.0, 3.0, 15)
    X_right = np.linspace(7.0, 10.0, 15)
    X_train = np.concatenate([X_left, X_right]).reshape(-1, 1)
    y_train = np.sin(X_train[:, 0])

    rf = RandomForestRegressor(n_estimators=15, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("proximity_b", rf)
    extractor.fit(X_train, y_train)

    # Compare midpoint of gap (x=5.0) against well-sampled point (x=1.5)
    u_dense = extractor.extract_epistemic_signal(np.array([[1.5]]))[0]
    u_gap = extractor.extract_epistemic_signal(np.array([[5.0]]))[0]

    assert u_gap > u_dense, f"Gap uncertainty ({u_gap}) must exceed dense region uncertainty ({u_dense})"


def test_b2_asymptotic_upper_bound_scaling():
    """B2.3: Verify uncertainty does not diverge as distance grows infinitely."""
    X_train = np.random.default_rng(42).uniform(-2.0, 2.0, (20, 2))
    y_train = X_train[:, 0] + X_train[:, 1]

    rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_proximity", rf)
    extractor.fit(X_train, y_train)

    u_10 = extractor.extract_epistemic_signal(np.array([[10.0, 10.0]]))[0]
    u_100 = extractor.extract_epistemic_signal(np.array([[100.0, 100.0]]))[0]
    u_1000 = extractor.extract_epistemic_signal(np.array([[1000.0, 1000.0]]))[0]

    # As distance grows, proximity uncertainty saturates rather than exploding
    assert np.isfinite(u_10) and np.isfinite(u_100) and np.isfinite(u_1000)
    assert abs(u_1000 - u_100) <= max(abs(u_100 - u_10), 1e-4)


def test_b2_gap_additive_acquisition_preference():
    """B2.4: Verify additive acquisition prioritizes high-uncertainty gap candidate when base scores are close."""
    base_acq = ExpectedImprovement()
    additive = AdditiveEpistemicAcquisition(base_acq=base_acq)

    preds = np.array([2.0, 2.05])       # Candidate 0 is slightly better in mean
    unc_tot = np.array([0.1, 0.1])      # Total variance similar
    u_epistemic = np.array([0.05, 0.95])# Candidate 1 is in a large exploration gap!
    y_best = 2.0

    # With exploration weight beta_t = 1.0, candidate 1 in gap should win
    scores = additive.compute_additive(preds, unc_tot, u_epistemic, y_best, beta_t=1.0)
    assert np.argmax(scores) == 1, "Additive acquisition failed to prioritize high-epistemic gap candidate"


def test_b2_disjoint_multi_cluster_extrapolation():
    """B2.5: Verify disjoint multi-cluster domain identifies inter-cluster valley with positive uncertainty."""
    cluster1 = np.random.default_rng(0).normal(loc=-5.0, scale=0.5, size=(15, 2))
    cluster2 = np.random.default_rng(1).normal(loc=5.0, scale=0.5, size=(15, 2))
    X_train = np.vstack([cluster1, cluster2])
    y_train = np.sin(X_train[:, 0]) + 2.0

    rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("proximity_b", rf)
    extractor.fit(X_train, y_train)

    # Evaluate at center between clusters: (0, 0)
    u_inter_cluster = extractor.extract_epistemic_signal(np.array([[0.0, 0.0]]))[0]
    u_centroid1 = extractor.extract_epistemic_signal(np.array([[-5.0, -5.0]]))[0]

    assert u_inter_cluster >= u_centroid1
    assert u_inter_cluster > 0.0


# ==============================================================================
# Boundary 3: High Dimensions
# ==============================================================================

@pytest.mark.parametrize("dim", [10, 25, 50])
def test_b3_high_dimension_extractor_scaling(dim):
    """B3.1: Verify extractor operates correctly across varying dimensions D in [10, 25, 50]."""
    rng = np.random.default_rng(42)
    X_train = rng.uniform(-2.0, 2.0, size=(25, dim))
    y_train = np.sum(X_train[:, :2] ** 2, axis=1)

    rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_disagreement", rf)
    extractor.fit(X_train, y_train)

    X_test = rng.uniform(-2.0, 2.0, size=(10, dim))
    signal = extractor.extract_epistemic_signal(X_test)

    assert signal.shape == (10,)
    assert np.all(np.isfinite(signal))
    assert np.all(signal >= 0.0)


def test_b3_high_dimension_few_samples_regime():
    """B3.2: Verify N << D regime (N=5 points in D=20) fits and predicts without rank errors."""
    rng = np.random.default_rng(42)
    X_train = rng.uniform(-1.0, 1.0, size=(5, 20))
    y_train = rng.uniform(0.0, 1.0, size=5)

    rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_disagreement", rf)
    extractor.fit(X_train, y_train)

    X_test = rng.uniform(-1.0, 1.0, size=(3, 20))
    signal = extractor.extract_epistemic_signal(X_test)
    assert signal.shape == (3,)
    assert np.all(np.isfinite(signal))


def test_b3_high_dimension_irrelevant_features():
    """B3.3: Verify data with mostly uninformative noise dimensions does not corrupt uncertainty."""
    rng = np.random.default_rng(42)
    N, D = 30, 20
    X_train = rng.uniform(-1.0, 1.0, size=(N, D))
    # Target depends solely on feature 0
    y_train = 5.0 * X_train[:, 0]

    rf = RandomForestRegressor(n_estimators=15, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("proximity_b", rf)
    extractor.fit(X_train, y_train)

    X_test = rng.uniform(-1.0, 1.0, size=(5, D))
    signal = extractor.extract_epistemic_signal(X_test)
    assert len(signal) == 5
    assert np.all(np.isfinite(signal))
    assert np.all(signal >= 0.0)


def test_b3_high_dimension_large_candidate_batch():
    """B3.4: Verify vectorized evaluation of large candidate pool (500 points in D=15)."""
    rng = np.random.default_rng(42)
    X_train = rng.uniform(0.0, 1.0, size=(40, 15))
    y_train = np.sum(X_train, axis=1)

    rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_disagreement", rf)
    extractor.fit(X_train, y_train)

    X_candidates = rng.uniform(0.0, 1.0, size=(500, 15))
    signal = extractor.extract_epistemic_signal(X_candidates)
    assert signal.shape == (500,)
    assert np.all(np.isfinite(signal))


def test_b3_high_dimension_distance_overflow_protection():
    """B3.5: Verify high-dimensional Euclidean distances don't overflow or produce NaNs."""
    X_train = np.ones((5, 50))
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_proximity", rf)
    extractor.fit(X_train, y_train)

    # Point placed far away in 50D space
    X_far = 100.0 * np.ones((2, 50))
    signal = extractor.extract_epistemic_signal(X_far)
    assert not np.any(np.isnan(signal))
    assert np.all(signal >= 0.0)


# ==============================================================================
# Boundary 4: Single-Sample Leaf / Degenerate Trees
# ==============================================================================

def test_b4_single_sample_dataset():
    """B4.1: Verify extractor handles minimal dataset with N=1 training sample."""
    X_train = np.array([[1.0, 2.0]])
    y_train = np.array([5.0])

    rf = RandomForestRegressor(n_estimators=10, oob_score=False, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_disagreement", rf)
    extractor.fit(X_train, y_train)

    signal = extractor.extract_epistemic_signal(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert len(signal) == 2
    assert np.all(np.isfinite(signal))
    assert np.all(signal >= 0.0)


def test_b4_identical_leaves_zero_variance():
    """B4.2: Verify trees with identical predictions produce well-defined non-negative uncertainty."""
    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([10.0, 10.0])

    rf = RandomForestRegressor(n_estimators=5, oob_score=False, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_disagreement", rf)
    extractor.fit(X_train, y_train)

    signal = extractor.extract_epistemic_signal(np.array([[0.5]]))
    assert np.all(signal >= 0.0)
    assert np.all(np.isfinite(signal))


def test_b4_single_tree_forest():
    """B4.3: Verify ensemble disagreement on a single tree (M=1) does not divide by (M-1)=0."""
    X_train = np.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    y_train = X_train[:, 0]

    rf = RandomForestRegressor(n_estimators=1, oob_score=False, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_disagreement", rf)
    extractor.fit(X_train, y_train)

    signal = extractor.extract_epistemic_signal(np.array([[0.0], [0.5]]))
    assert len(signal) == 2
    assert np.all(np.isfinite(signal))
    assert np.all(signal == 0.0)  # Variance across 1 tree is 0.0


def test_b4_leaf_sample_count_inverse_bounded():
    """B4.4: Verify leaf sample count term 1/N_m(x) is finite and bounded when N_m=1."""
    # When min_samples_leaf=1, trees can isolate single samples in leaves
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([0.0, 1.0, 4.0, 9.0])

    rf = RandomForestRegressor(n_estimators=10, min_samples_leaf=1, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("proximity_b", rf)
    extractor.fit(X_train, y_train)

    signal = extractor.extract_epistemic_signal(X_train)
    assert np.all(np.isfinite(signal))
    assert np.all(signal >= 0.0)


def test_b4_deep_unconstrained_trees():
    """B4.5: Verify fully unpruned trees (max_depth=None) maintain numerical stability."""
    rng = np.random.default_rng(42)
    X_train = rng.uniform(-5.0, 5.0, size=(50, 3))
    y_train = np.sum(X_train**2, axis=1)

    rf = RandomForestRegressor(
        n_estimators=10,
        max_depth=None,
        min_samples_split=2,
        oob_score=True,
        random_state=42
    )
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_disagreement", rf)
    extractor.fit(X_train, y_train)

    X_test = rng.uniform(-10.0, 10.0, size=(20, 3))
    signal = extractor.extract_epistemic_signal(X_test)
    assert np.all(np.isfinite(signal))
    assert np.all(signal >= 0.0)


# ==============================================================================
# Boundary 5: NaN/Inf Bounds & Numerical Stability
# ==============================================================================

def test_b5_normalize_max_relative_nan_inf_replacement():
    """B5.1: Verify normalize_max_relative cleans NaN and Inf values without propagating them."""
    dirty_arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
    cleaned = normalize_max_relative(dirty_arr)
    assert not np.any(np.isnan(cleaned)), "NaN leaked through normalize_max_relative"
    assert not np.any(np.isinf(cleaned)), "Inf leaked through normalize_max_relative"
    assert np.all(cleaned >= 0.0)
    assert np.max(cleaned) <= 1.0


def test_b5_extreme_target_scales_micro():
    """B5.2: Verify microscopic target magnitudes (y ~ 10^-8) do not underflow to negative values."""
    X_train = np.linspace(0.0, 1.0, 15).reshape(-1, 1)
    y_train = 1e-8 * np.sin(2 * np.pi * X_train[:, 0])

    rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_disagreement", rf)
    extractor.fit(X_train, y_train)

    signal = extractor.extract_epistemic_signal(np.array([[0.25], [0.75]]))
    assert np.all(np.isfinite(signal))
    assert np.all(signal >= 0.0)
    assert np.max(signal) < 1e-5


def test_b5_extreme_target_scales_macro():
    """B5.3: Verify macroscopic target magnitudes (y ~ 10^8) do not overflow float64."""
    X_train = np.linspace(0.0, 1.0, 15).reshape(-1, 1)
    y_train = 1e8 * np.sin(2 * np.pi * X_train[:, 0])

    rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_disagreement", rf)
    extractor.fit(X_train, y_train)

    signal = extractor.extract_epistemic_signal(np.array([[0.25], [0.75]]))
    assert np.all(np.isfinite(signal))
    assert np.all(signal >= 0.0)


def test_b5_float32_input_dtype_compatibility():
    """B5.4: Verify extractor processes float32 numpy arrays seamlessly."""
    X_train = np.array([[0.1, 0.2], [0.5, 0.6]], dtype=np.float32)
    y_train = np.array([1.0, 2.0], dtype=np.float32)

    rf = RandomForestRegressor(n_estimators=5, oob_score=False, random_state=42)
    rf.fit(X_train, y_train)

    extractor = UQExtractorRegistry.get("standard_disagreement", rf)
    extractor.fit(X_train, y_train)

    X_test = np.array([[0.3, 0.4]], dtype=np.float32)
    signal = extractor.extract_epistemic_signal(X_test)
    assert isinstance(signal, np.ndarray)
    assert np.all(np.isfinite(signal))
    assert np.all(signal >= 0.0)


def test_b5_near_zero_variance_clipping_guard():
    """B5.5: Verify numerical jitter does not produce negative variance in CustomUncertaintyRandomForest."""
    cs = ConfigurationSpace({"x": Float("x", bounds=(0.0, 1.0))})
    model = CustomUncertaintyRandomForest(
        uncertainty_func="standard_disagreement",
        configspace=cs,
        n_trees=15,
        seed=42,
    )
    # Train on identical outputs
    X_train = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    y_train = np.array([[5.0], [5.0], [5.0], [5.0], [5.0]])
    model.train(X_train, y_train)

    _, var = model._predict(X_train)
    assert np.all(var >= 0.0), f"Variance contains negative values: {var}"
    assert not np.any(np.isnan(var))
