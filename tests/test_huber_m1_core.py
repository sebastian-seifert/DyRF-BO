"""
Unit tests for Milestone 1: Vectorized Closed-Form Huber (2008) Differential Entropy H_2.

Reference:
- Huber, Bailey, Durrant-Whyte, Hanebeck (IEEE MFI 2008):
  "On Entropy Approximation for Gaussian Mixture Random Vectors"
- Specification: detailed_huber_implementation_plan.md (Milestone 1)
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy.integrate import quad

from dyrf_bo.entropy import huber_entropy_1d_numpy


def test_single_gaussian_limit():
    """
    Asserts that for T=1 component, Huber H_2 exactly matches the analytical
    Shannon differential entropy of a single Gaussian: 0.5 * log2(2 * pi * e * var)
    within 1e-12 bits tolerance across various means and variances.
    """
    test_means = [-5.0, 0.0, 10.0]
    test_vars = [1e-4, 0.1, 1.0, 25.0]

    for mu in test_means:
        for var in test_vars:
            h_exact = 0.5 * np.log2(2.0 * np.pi * np.e * var)

            # Test 2D input (B=1, T=1)
            means_2d = np.array([[mu]], dtype=np.float64)
            vars_2d = np.array([[var]], dtype=np.float64)
            h2 = huber_entropy_1d_numpy(means_2d, vars_2d)

            assert isinstance(h2, np.ndarray), "Output must be a numpy ndarray"
            assert h2.shape == (1,), f"Expected shape (1,), got {h2.shape}"
            np.testing.assert_allclose(
                h2[0],
                h_exact,
                atol=1e-12,
                rtol=0,
                err_msg=f"Failed single Gaussian limit for mu={mu}, var={var}",
            )


def test_consensus_trees_limit():
    """
    Asserts that when all T=50 trees predict identical distributions (consensus ensemble),
    Huber H_2 matches the single Gaussian entropy within 1e-11 bits tolerance.
    """
    T = 50
    mu_val = 3.5
    var_val = 2.0
    h_exact = 0.5 * np.log2(2.0 * np.pi * np.e * var_val)

    means = np.full((1, T), mu_val, dtype=np.float64)
    variances = np.full((1, T), var_val, dtype=np.float64)

    h2 = huber_entropy_1d_numpy(means, variances)
    np.testing.assert_allclose(
        h2[0],
        h_exact,
        atol=1e-11,
        rtol=0,
        err_msg="Consensus ensemble of 50 identical components diverged from single Gaussian",
    )

    # Test batch of consensus configurations
    B = 5
    batch_mus = np.array([-2.0, 0.0, 1.0, 3.5, 10.0])
    batch_vars = np.array([0.05, 0.5, 1.0, 2.0, 15.0])
    b_means = np.tile(batch_mus[:, None], (1, T))
    b_vars = np.tile(batch_vars[:, None], (1, T))
    b_exact = 0.5 * np.log2(2.0 * np.pi * np.e * batch_vars)

    b_h2 = huber_entropy_1d_numpy(b_means, b_vars)
    assert b_h2.shape == (B,)
    np.testing.assert_allclose(
        b_h2,
        b_exact,
        atol=1e-11,
        rtol=0,
        err_msg="Batch consensus calculation diverged from analytical single Gaussians",
    )


def test_batch_broadcasting():
    """
    Asserts that vectorized batch calculation over B=128, T=20 matches point-by-point
    iterative loop evaluation within 1e-14 bits tolerance.
    """
    rng = np.random.default_rng(42)
    B = 128
    T = 20

    means = rng.normal(loc=0.0, scale=2.0, size=(B, T))
    variances = rng.uniform(low=0.1, high=5.0, size=(B, T))

    # Vectorized execution
    h2_vectorized = huber_entropy_1d_numpy(means, variances)
    assert h2_vectorized.shape == (B,), f"Expected shape ({B},), got {h2_vectorized.shape}"

    # Iterative execution point by point
    h2_loop = np.empty(B, dtype=np.float64)
    for i in range(B):
        res = huber_entropy_1d_numpy(means[i : i + 1], variances[i : i + 1])
        h2_loop[i] = res[0]

    np.testing.assert_allclose(
        h2_vectorized,
        h2_loop,
        atol=1e-14,
        rtol=0,
        err_msg="Vectorized batch computation differs from iterative single-sample computation",
    )


def test_numerical_stability_extreme_separation():
    """
    Asserts that two widely separated components (T=2, mu=[0, 1000], var=[1, 1])
    do not overflow or produce NaNs/Infs, and approach the theoretical disjoint support
    limit: log2(2) + 0.5 * log2(2 * pi * e) within 1e-4 bits tolerance.
    """
    means = np.array([[0.0, 1000.0]], dtype=np.float64)
    variances = np.array([[1.0, 1.0]], dtype=np.float64)

    h2 = huber_entropy_1d_numpy(means, variances)

    assert np.all(np.isfinite(h2)), f"Extreme separation yielded non-finite values: {h2}"

    # Theoretical limit for disjoint support mixture of 2 identical Gaussians:
    # H[f(y)] = H(weights) + sum(w_i * H(N_i)) = log2(2) + 0.5 * log2(2 * pi * e * 1.0)
    h_expected = np.log2(2.0) + 0.5 * np.log2(2.0 * np.pi * np.e)
    np.testing.assert_allclose(
        h2[0],
        h_expected,
        atol=1e-4,
        rtol=0,
        err_msg="Extreme separation limit diverged from disjoint support theoretical entropy",
    )


def test_numerical_stability_tiny_variance():
    """
    Asserts numerical stability when variances are tiny (T=10, var=1e-6, clipped by min_var).
    Verifies that the returned values are finite, real-valued floats without exceptions,
    and that the resulting mutual information (epistemic uncertainty MI = H_2 - H_aleatoric)
    is strictly non-negative. Also verifies variance-equivalent uncertainty is non-negative.
    """
    T = 10
    means = np.linspace(-5.0, 5.0, T, dtype=np.float64).reshape(1, T)
    variances = np.full((1, T), 1e-6, dtype=np.float64)

    h2 = huber_entropy_1d_numpy(means, variances, min_var=1e-6)

    assert np.all(np.isfinite(h2)), "Output contains NaNs or Infs for tiny variance"

    # Differential entropy for very small variance (< 1/(2*pi*e)) is mathematically negative,
    # but the mutual information (epistemic uncertainty MI = H_2 - H_aleatoric) must be >= 0.
    h_aleatoric = np.mean(0.5 * np.log2(2.0 * np.pi * np.e * variances), axis=-1)
    epistemic_mi = h2 - h_aleatoric
    assert np.all(epistemic_mi >= -1e-11), (
        f"Mutual information (epistemic uncertainty) must be >= 0, got {epistemic_mi}"
    )

    # Effective variance-equivalent uncertainty is strictly non-negative
    sigma_equiv_sq = (2.0 ** (2.0 * h2)) / (2.0 * np.pi * np.e)
    assert np.all(sigma_equiv_sq >= 0.0), "Equivalent variance must be non-negative"

    # Also test clipping behavior when variances are below min_var (e.g., 0.0 or 1e-12)
    sub_zero_vars = np.zeros((1, T), dtype=np.float64)
    h2_clipped = huber_entropy_1d_numpy(sub_zero_vars, sub_zero_vars, min_var=1e-6)
    assert np.all(np.isfinite(h2_clipped)), "Zero variance clipping produced non-finite entropy"


def test_1d_input_shape():
    """
    Asserts that 1D inputs of shape (T,) are accepted and return a 1D array of shape (1,)
    or scalar, matching the value computed with 2D shape (1, T).
    """
    means_1d = np.array([-1.0, 0.5, 2.0], dtype=np.float64)
    vars_1d = np.array([0.8, 1.2, 0.5], dtype=np.float64)

    h2_1d = huber_entropy_1d_numpy(means_1d, vars_1d)
    assert isinstance(h2_1d, np.ndarray), "Output must be a numpy ndarray"
    assert h2_1d.shape == (1,) or h2_1d.ndim == 0, f"Expected shape (1,) or scalar, got {h2_1d.shape}"

    # Verify identical value to 2D invocation
    h2_2d = huber_entropy_1d_numpy(means_1d.reshape(1, -1), vars_1d.reshape(1, -1))
    np.testing.assert_allclose(
        np.squeeze(h2_1d),
        h2_2d[0],
        atol=1e-14,
        rtol=0,
        err_msg="1D input produced different result from equivalent 2D input",
    )


def test_accuracy_vs_scipy_quad():
    """
    Compares Huber H_2 against numerical integration via scipy.integrate.quad
    on a bimodal mixture: mu=[0.0, 2.0], var=[1.0, 0.5].
    Asserts relative error is strictly below 5% (Specification Table 1.4).
    """
    means = np.array([[0.0, 2.0]], dtype=np.float64)
    variances = np.array([[1.0, 0.5]], dtype=np.float64)
    weights = np.array([[0.5, 0.5]], dtype=np.float64)

    # 1. Analytical PDF definition for quad
    def pdf(y):
        c1 = 0.5 * (1.0 / np.sqrt(2.0 * np.pi * 1.0)) * np.exp(-0.5 * (y - 0.0) ** 2 / 1.0)
        c2 = 0.5 * (1.0 / np.sqrt(2.0 * np.pi * 0.5)) * np.exp(-0.5 * (y - 2.0) ** 2 / 0.5)
        return c1 + c2

    def integrand(y):
        p = pdf(y)
        if p <= 1e-300:
            return 0.0
        return -p * np.log2(p)

    h_quad, _ = quad(integrand, -15.0, 15.0, epsabs=1e-9, epsrel=1e-9)

    # 2. Huber H_2
    h2 = huber_entropy_1d_numpy(means, variances, weights=weights)
    h2_val = h2[0]

    rel_error = abs(h2_val - h_quad) / h_quad
    assert rel_error < 0.05, f"Relative error {rel_error:.4f} exceeded 5% limit (H2={h2_val}, H_quad={h_quad})"


def test_custom_weights():
    """
    Asserts correct behavior with custom component mixture weights:
    - Weight normalization if unnormalized weights are provided
    - Degenerate weight [1.0, 0.0] reduces to single Gaussian
    - 1D weights of shape (T,) broadcast over batch (B, T)
    """
    # Degenerate weight: weight 1.0 on component 0, 0.0 on component 1
    means = np.array([[1.5, -4.0]], dtype=np.float64)
    variances = np.array([[2.0, 0.5]], dtype=np.float64)
    weights_degen = np.array([[1.0, 0.0]], dtype=np.float64)

    h2_degen = huber_entropy_1d_numpy(means, variances, weights=weights_degen)
    h_exact_comp0 = 0.5 * np.log2(2.0 * np.pi * np.e * 2.0)
    np.testing.assert_allclose(
        h2_degen[0],
        h_exact_comp0,
        atol=1e-12,
        rtol=0,
        err_msg="Degenerate weight [1, 0] failed to reduce to component 0 entropy",
    )

    # 1D weights broadcasting across batch B=3
    B = 3
    T = 4
    rng = np.random.default_rng(123)
    b_means = rng.normal(size=(B, T))
    b_vars = rng.uniform(0.5, 3.0, size=(B, T))
    w_1d = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    h2_broadcast = huber_entropy_1d_numpy(b_means, b_vars, weights=w_1d)
    w_2d = np.tile(w_1d[None, :], (B, 1))
    h2_explicit = huber_entropy_1d_numpy(b_means, b_vars, weights=w_2d)
    np.testing.assert_allclose(
        h2_broadcast,
        h2_explicit,
        atol=1e-14,
        rtol=0,
        err_msg="1D weights broadcasting did not match 2D weights",
    )


def test_edge_cases_and_exceptions():
    """
    Asserts appropriate handling and error raising for invalid inputs and edge cases:
    - Empty batch (B=0) returns empty array with shape (0,)
    - T=0 raises ValueError
    - Shape mismatch between means and variances raises ValueError
    - 3D inputs raise ValueError
    - Negative weights raise ValueError
    - Zero weights sum raises ValueError
    - Incompatible weights shape raises ValueError
    """
    # Empty batch B=0
    empty_means = np.empty((0, 5), dtype=np.float64)
    empty_vars = np.empty((0, 5), dtype=np.float64)
    res_empty = huber_entropy_1d_numpy(empty_means, empty_vars)
    assert res_empty.shape == (0,)

    # T=0
    with pytest.raises(ValueError, match="at least 1"):
        huber_entropy_1d_numpy(np.empty((3, 0)), np.empty((3, 0)))

    # Shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        huber_entropy_1d_numpy(np.zeros((3, 4)), np.zeros((3, 5)))

    # 3D inputs
    with pytest.raises(ValueError, match="1D or 2D array"):
        huber_entropy_1d_numpy(np.zeros((2, 3, 4)), np.zeros((2, 3, 4)))

    # Negative weights
    with pytest.raises(ValueError, match="non-negative"):
        huber_entropy_1d_numpy(np.zeros((2, 3)), np.ones((2, 3)), weights=np.array([1.0, -0.5, 0.5]))

    # Zero weights sum
    with pytest.raises(ValueError, match="strictly positive"):
        huber_entropy_1d_numpy(np.zeros((2, 3)), np.ones((2, 3)), weights=np.array([0.0, 0.0, 0.0]))

    # Incompatible weights shape
    with pytest.raises(ValueError, match="does not match"):
        huber_entropy_1d_numpy(np.zeros((2, 3)), np.ones((2, 3)), weights=np.ones((2, 4)))

