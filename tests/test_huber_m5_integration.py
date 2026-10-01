"""
Unit and integration tests for Milestone 5: Integration into Epistemic_Quantifier.py.

Reference:
- Huber, Bailey, Durrant-Whyte, Hanebeck (IEEE MFI 2008):
  "On Entropy Approximation for Gaussian Mixture Random Vectors"
- Specification: detailed_huber_implementation_plan.md (Milestone 5)
"""

import inspect
import os
import sys

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from Epistemic_Quantifier import EpistemicQuantifier, LeafCache
from ep_extractors.shaker_entropy import ShakerEntropyExtractor


@pytest.fixture
def synthetic_rf_data():
    """Generates synthetic dataset and trained RandomForestRegressor for tests."""
    rng = np.random.default_rng(42)
    n_train = 60
    n_test = 25
    n_dim = 3

    X_train = rng.uniform(-2.0, 2.0, size=(n_train, n_dim))
    y_train = (
        np.sin(X_train[:, 0])
        + 0.5 * np.cos(2.0 * X_train[:, 1])
        + rng.normal(0, 0.1, size=n_train)
    )

    X_test = rng.uniform(-3.0, 3.0, size=(n_test, n_dim))

    rf = RandomForestRegressor(
        n_estimators=12,
        min_samples_leaf=2,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    return rf, X_train, y_train, X_test


def test_default_method_is_huber():
    """Asserts that method='huber' is the default parameter across public signatures."""
    sig_entropy = inspect.signature(EpistemicQuantifier.shaker_get_epistemic_entropy)
    assert sig_entropy.parameters["method"].default == "huber"
    assert "enable_splitting" in sig_entropy.parameters
    assert sig_entropy.parameters["enable_splitting"].default is False
    assert "return_bounds" in sig_entropy.parameters
    assert sig_entropy.parameters["return_bounds"].default is False

    sig_var = inspect.signature(EpistemicQuantifier.shaker_get_epistemic_variance)
    assert sig_var.parameters["method"].default == "huber"
    assert "enable_splitting" in sig_var.parameters
    assert sig_var.parameters["enable_splitting"].default is False
    assert "return_bounds" in sig_var.parameters
    assert sig_var.parameters["return_bounds"].default is False

    sig_tot_var = inspect.signature(EpistemicQuantifier.shaker_get_total_variance)
    assert sig_tot_var.parameters["method"].default == "huber"
    assert "enable_splitting" in sig_tot_var.parameters
    assert sig_tot_var.parameters["enable_splitting"].default is False
    assert "return_bounds" in sig_tot_var.parameters
    assert sig_tot_var.parameters["return_bounds"].default is False

    sig_calc_tot = inspect.signature(EpistemicQuantifier._shaker_calc_total_entropy)
    assert sig_calc_tot.parameters["method"].default == "huber"
    assert "enable_splitting" in sig_calc_tot.parameters
    assert sig_calc_tot.parameters["enable_splitting"].default is False
    assert "return_bounds" in sig_calc_tot.parameters
    assert sig_calc_tot.parameters["return_bounds"].default is False

    sig_extractor = inspect.signature(ShakerEntropyExtractor.__init__)
    assert sig_extractor.parameters["method"].default == "huber"


def test_backward_compatible_signatures(synthetic_rf_data):
    """
    Fits RandomForestRegressor on synthetic data, instantiates EpistemicQuantifier;
    calls shaker_get_epistemic_variance(X_test), shaker_get_total_variance(X_test),
    shaker_get_epistemic_entropy(X_test) with default arguments.
    Verifies shape (N,), all elements finite, and all elements non-negative.
    """
    rf, X_train, y_train, X_test = synthetic_rf_data
    quantifier = EpistemicQuantifier(rf, X_train, y_train)
    n_test = len(X_test)

    # 1. Epistemic variance with default arguments (now Huber H_2)
    ep_var = quantifier.shaker_get_epistemic_variance(X_test)
    assert isinstance(ep_var, np.ndarray)
    assert ep_var.shape == (n_test,)
    assert np.all(np.isfinite(ep_var))
    assert np.all(ep_var >= 0.0)

    # 2. Total variance with default arguments
    tot_var = quantifier.shaker_get_total_variance(X_test)
    assert isinstance(tot_var, np.ndarray)
    assert tot_var.shape == (n_test,)
    assert np.all(np.isfinite(tot_var))
    assert np.all(tot_var >= 0.0)

    # 3. Epistemic entropy with default arguments
    ep_entropy = quantifier.shaker_get_epistemic_entropy(X_test)
    assert isinstance(ep_entropy, np.ndarray)
    assert ep_entropy.shape == (n_test,)
    assert np.all(np.isfinite(ep_entropy))
    assert np.all(ep_entropy >= 0.0)

    # Upper bound invariant: MI <= log2(n_trees)
    max_bound = float(np.log2(len(rf.estimators_)))
    assert np.all(ep_entropy <= max_bound + 1e-12)


def test_leaf_cache_integration(synthetic_rf_data):
    """
    Compares EpistemicQuantifier with and without LeafCache.
    Verifies that cached and uncached predictions are identical within 1e-14.
    """
    rf, X_train, y_train, X_test = synthetic_rf_data

    # Uncached quantifier
    eq_uncached = EpistemicQuantifier(rf, X_train, y_train)
    entropy_uncached = eq_uncached.shaker_get_epistemic_entropy(X_test)
    var_uncached = eq_uncached.shaker_get_epistemic_variance(X_test)
    tot_var_uncached = eq_uncached.shaker_get_total_variance(X_test)

    # Cached quantifier
    cache = LeafCache(rf, X_test, X_train=X_train)
    eq_cached = EpistemicQuantifier(rf, X_train, y_train, leaf_cache=cache)
    entropy_cached = eq_cached.shaker_get_epistemic_entropy(X_test)
    var_cached = eq_cached.shaker_get_epistemic_variance(X_test)
    tot_var_cached = eq_cached.shaker_get_total_variance(X_test)

    np.testing.assert_allclose(
        entropy_cached,
        entropy_uncached,
        atol=1e-14,
        rtol=1e-14,
        err_msg="LeafCache produced divergent epistemic entropy from uncached pass",
    )
    np.testing.assert_allclose(
        var_cached,
        var_uncached,
        atol=1e-14,
        rtol=1e-14,
        err_msg="LeafCache produced divergent epistemic variance from uncached pass",
    )
    np.testing.assert_allclose(
        tot_var_cached,
        tot_var_uncached,
        atol=1e-14,
        rtol=1e-14,
        err_msg="LeafCache produced divergent total variance from uncached pass",
    )


def test_epistemic_bounds_hierarchy(synthetic_rf_data):
    """
    Calls shaker_get_epistemic_entropy(X_test, return_bounds=True).
    Asserts returns tuple of 3 arrays (MI_l, MI, MI_u) where:
    0 <= MI_l <= MI <= MI_u <= log2(n_trees) pointwise.
    """
    rf, X_train, y_train, X_test = synthetic_rf_data
    quantifier = EpistemicQuantifier(rf, X_train, y_train)
    n_trees = len(rf.estimators_)
    max_bound = float(np.log2(n_trees))

    bounds_result = quantifier.shaker_get_epistemic_entropy(
        X_test, return_bounds=True
    )

    assert isinstance(bounds_result, tuple)
    assert len(bounds_result) == 3

    mi_l, mi, mi_u = bounds_result
    assert mi_l.shape == (len(X_test),)
    assert mi.shape == (len(X_test),)
    assert mi_u.shape == (len(X_test),)

    # Pointwise bounds hierarchy: 0 <= MI_l <= MI <= MI_u <= log2(n_trees)
    assert np.all(mi_l >= 0.0), f"MI_l has negative elements: {mi_l[mi_l < 0.0]}"
    assert np.all(mi_l <= mi + 1e-12), f"MI_l > MI detected: max diff = {np.max(mi_l - mi)}"
    assert np.all(mi <= mi_u + 1e-12), f"MI > MI_u detected: max diff = {np.max(mi - mi_u)}"
    assert np.all(mi_u <= max_bound + 1e-12), f"MI_u > log2(T) detected: max = {np.max(mi_u)}"

    # Also test that return_bounds=True works on variance methods
    ep_var_bounds = quantifier.shaker_get_epistemic_variance(X_test, return_bounds=True)
    assert isinstance(ep_var_bounds, tuple)
    assert len(ep_var_bounds) == 3
    var_l, var_2, var_u = ep_var_bounds
    assert np.all(0.0 <= var_l)
    assert np.all(var_l <= var_2 + 1e-12)
    assert np.all(var_2 <= var_u + 1e-12)

    tot_var_bounds = quantifier.shaker_get_total_variance(X_test, return_bounds=True)
    assert isinstance(tot_var_bounds, tuple)
    assert len(tot_var_bounds) == 3
    tot_l, tot_2, tot_u = tot_var_bounds
    assert np.all(0.0 <= tot_l)
    assert np.all(tot_l <= tot_2 + 1e-12)
    assert np.all(tot_2 <= tot_u + 1e-12)


def test_legacy_methods_preserved(synthetic_rf_data):
    """
    Tests method="gauss_hermite" and method="monte_carlo";
    verifies both run cleanly and return (N,) arrays.
    """
    rf, X_train, y_train, X_test = synthetic_rf_data
    quantifier = EpistemicQuantifier(rf, X_train, y_train)
    n_test = len(X_test)

    # Gauss-Hermite legacy method
    gh_entropy = quantifier.shaker_get_epistemic_entropy(
        X_test, method="gauss_hermite", n_quadrature_points=16
    )
    assert isinstance(gh_entropy, np.ndarray)
    assert gh_entropy.shape == (n_test,)
    assert np.all(np.isfinite(gh_entropy))
    assert np.all(gh_entropy >= 0.0)

    gh_var = quantifier.shaker_get_epistemic_variance(
        X_test, method="gauss_hermite", n_quadrature_points=16
    )
    assert isinstance(gh_var, np.ndarray)
    assert gh_var.shape == (n_test,)
    assert np.all(np.isfinite(gh_var))
    assert np.all(gh_var >= 0.0)

    # Monte Carlo legacy method
    mc_entropy = quantifier.shaker_get_epistemic_entropy(
        X_test, method="monte_carlo", num_samples=1000, random_state=42
    )
    assert isinstance(mc_entropy, np.ndarray)
    assert mc_entropy.shape == (n_test,)
    assert np.all(np.isfinite(mc_entropy))
    assert np.all(mc_entropy >= 0.0)

    mc_var = quantifier.shaker_get_epistemic_variance(
        X_test, method="monte_carlo", num_samples=1000, random_state=42
    )
    assert isinstance(mc_var, np.ndarray)
    assert mc_var.shape == (n_test,)
    assert np.all(np.isfinite(mc_var))
    assert np.all(mc_var >= 0.0)


def test_splitting_parameter_integration(synthetic_rf_data):
    """
    Tests that enable_splitting=True runs cleanly through EpistemicQuantifier
    and produces finite non-negative results bounded by log2(n_trees).
    """
    rf, X_train, y_train, X_test = synthetic_rf_data
    quantifier = EpistemicQuantifier(rf, X_train, y_train)
    n_trees = len(rf.estimators_)
    max_bound = float(np.log2(n_trees))

    # Split entropy
    split_entropy = quantifier.shaker_get_epistemic_entropy(
        X_test, enable_splitting=True
    )
    assert isinstance(split_entropy, np.ndarray)
    assert split_entropy.shape == (len(X_test),)
    assert np.all(np.isfinite(split_entropy))
    assert np.all(split_entropy >= 0.0)
    assert np.all(split_entropy <= max_bound + 1e-12)

    # Split variance
    split_var = quantifier.shaker_get_epistemic_variance(
        X_test, enable_splitting=True
    )
    assert isinstance(split_var, np.ndarray)
    assert split_var.shape == (len(X_test),)
    assert np.all(np.isfinite(split_var))
    assert np.all(split_var >= 0.0)

    # Split total variance
    split_tot_var = quantifier.shaker_get_total_variance(
        X_test, enable_splitting=True
    )
    assert isinstance(split_tot_var, np.ndarray)
    assert split_tot_var.shape == (len(X_test),)
    assert np.all(np.isfinite(split_tot_var))
    assert np.all(split_tot_var >= 0.0)

    # Split with LeafCache
    cache = LeafCache(rf, X_test, X_train=X_train)
    eq_cached = EpistemicQuantifier(rf, X_train, y_train, leaf_cache=cache)
    split_cached = eq_cached.shaker_get_epistemic_entropy(
        X_test, enable_splitting=True
    )
    np.testing.assert_allclose(split_cached, split_entropy, atol=1e-14, rtol=1e-14)


def test_shaker_extractor_uses_huber(synthetic_rf_data):
    """
    Tests ShakerEntropyExtractor end-to-end with default method='huber'.
    """
    rf, X_train, y_train, X_test = synthetic_rf_data
    extractor = ShakerEntropyExtractor(rf)
    assert extractor.method == "huber"

    extractor.fit(X_train, y_train)
    signal = extractor.extract_epistemic_signal(X_test)

    assert isinstance(signal, np.ndarray)
    assert signal.shape == (len(X_test),)
    assert np.all(np.isfinite(signal))
    assert np.all(signal >= 0.0)


def test_return_bounds_unsupported_method_raises(synthetic_rf_data):
    """
    Asserts that passing return_bounds=True with method="gauss_hermite"
    or method="monte_carlo" raises a ValueError.
    """
    rf, X_train, y_train, X_test = synthetic_rf_data
    quantifier = EpistemicQuantifier(rf, X_train, y_train)

    with pytest.raises(
        ValueError, match="return_bounds=True is only supported when method='huber'"
    ):
        quantifier.shaker_get_epistemic_entropy(
            X_test, method="gauss_hermite", return_bounds=True
        )

    with pytest.raises(
        ValueError, match="return_bounds=True is only supported when method='huber'"
    ):
        quantifier.shaker_get_epistemic_variance(
            X_test, method="gauss_hermite", return_bounds=True
        )

    with pytest.raises(
        ValueError, match="return_bounds=True is only supported when method='huber'"
    ):
        quantifier.shaker_get_total_variance(
            X_test, method="gauss_hermite", return_bounds=True
        )

    with pytest.raises(
        ValueError, match="return_bounds=True is only supported when method='huber'"
    ):
        quantifier.shaker_get_epistemic_entropy(
            X_test, method="monte_carlo", return_bounds=True
        )


def test_1d_input_produces_1d_output_shape(synthetic_rf_data):
    """
    Verifies that passing a 1D array X_test of shape (n_features,) produces
    a 1D array of shape (1,) for all public Shaker quantifier methods.
    """
    rf, X_train, y_train, _ = synthetic_rf_data
    quantifier = EpistemicQuantifier(rf, X_train, y_train)
    n_features = X_train.shape[1]

    # Single test point with 1D shape (n_features,)
    x_single = np.ones(n_features, dtype=np.float64)
    assert x_single.ndim == 1
    assert x_single.shape == (n_features,)

    # 1. Epistemic entropy
    ent = quantifier.shaker_get_epistemic_entropy(x_single)
    assert isinstance(ent, np.ndarray)
    assert ent.shape == (1,)
    assert np.all(np.isfinite(ent))

    # 2. Epistemic entropy with return_bounds=True
    ent_bounds = quantifier.shaker_get_epistemic_entropy(x_single, return_bounds=True)
    assert isinstance(ent_bounds, tuple) and len(ent_bounds) == 3
    for b in ent_bounds:
        assert isinstance(b, np.ndarray)
        assert b.shape == (1,)
        assert np.all(np.isfinite(b))

    # 3. Epistemic variance
    var = quantifier.shaker_get_epistemic_variance(x_single)
    assert isinstance(var, np.ndarray)
    assert var.shape == (1,)
    assert np.all(np.isfinite(var))

    # 4. Total variance
    tot_var = quantifier.shaker_get_total_variance(x_single)
    assert isinstance(tot_var, np.ndarray)
    assert tot_var.shape == (1,)
    assert np.all(np.isfinite(tot_var))

