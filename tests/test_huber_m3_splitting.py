"""
Unit tests for Milestone 3: Huber Table I Component Splitting (Option B).

Reference:
- Huber, Bailey, Durrant-Whyte, Hanebeck (IEEE MFI 2008):
  "On Entropy Approximation for Gaussian Mixture Random Vectors" (Section IV-B & Remark 1)
- Specification: detailed_huber_implementation_plan.md (Milestone 3)
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy.integrate import quad

from dyrf_bo.entropy import (
    TABLE1_MEANS,
    TABLE1_STDS,
    TABLE1_WEIGHTS,
    huber_entropy_1d_numpy,
    huber_table1_split_1d,
)


def test_table1_moment_conservation():
    """
    Asserts moment conservation for standard normal N(0, 1) split into 4 components:
    - 0th moment (total probability mass) == 1.0 within 1e-10.
    - 1st moment (mean) == 0.0 within 1e-10.
    - 2nd central moment (variance) == 1.0 within 1e-10 under moment-preserving mode.
    Also verifies raw Table I constants and their immutability.
    """
    # 1. Moment-preserving split of N(0, 1)
    means_split, vars_split, weights_split = huber_table1_split_1d(
        np.array([0.0]), np.array([1.0])
    )

    assert means_split.shape == (1, 4), f"Expected shape (1, 4), got {means_split.shape}"
    assert vars_split.shape == (1, 4), f"Expected shape (1, 4), got {vars_split.shape}"
    assert weights_split.shape == (1, 4), f"Expected shape (1, 4), got {weights_split.shape}"

    w = weights_split[0]
    mu = means_split[0]
    var = vars_split[0]

    # 0th moment: sum(w_k) == 1.0
    m0 = np.sum(w)
    assert np.isclose(m0, 1.0, atol=1e-10), f"0th moment {m0} != 1.0"

    # 1st moment: sum(w_k * mu_k) == 0.0
    m1 = np.sum(w * mu)
    assert np.isclose(m1, 0.0, atol=1e-10), f"1st moment {m1} != 0.0"

    # 2nd central moment: sum(w_k * ((mu_k - m1)^2 + var_k)) == 1.0
    m2 = np.sum(w * ((mu - m1) ** 2 + var))
    assert np.isclose(m2, 1.0, atol=1e-10), f"2nd central moment {m2} != 1.0"

    # 2. Raw Table I parameters check (preserve_moments=False)
    means_raw, vars_raw, weights_raw = huber_table1_split_1d(
        np.array([0.0]), np.array([1.0]), preserve_moments=False
    )
    w_raw = weights_raw[0]
    mu_raw = means_raw[0]
    var_raw = vars_raw[0]

    np.testing.assert_allclose(w_raw, TABLE1_WEIGHTS / np.sum(TABLE1_WEIGHTS), atol=1e-12)
    np.testing.assert_allclose(mu_raw, TABLE1_MEANS, atol=1e-12)
    np.testing.assert_allclose(var_raw, TABLE1_STDS ** 2, atol=1e-12)

    # Raw 2nd moment from Hanebeck/Huber L2 fitting is ~0.9272854677
    m2_raw = np.sum(w_raw * ((mu_raw - np.sum(w_raw * mu_raw)) ** 2 + var_raw))
    assert np.isclose(m2_raw, 0.9272854677110389, atol=1e-10)

    # 3. Immutability verification of Table I constants
    with pytest.raises(ValueError):
        TABLE1_WEIGHTS[0] = 999.0
    with pytest.raises(ValueError):
        TABLE1_MEANS[0] = 999.0
    with pytest.raises(ValueError):
        TABLE1_STDS[0] = 999.0


def test_gmm_split_moment_conservation():
    """
    Asserts that for an arbitrary GMM (B=10, T=5) with non-uniform weights,
    the ensemble mean and total mixture variance of the 4T split mixture
    match the original GMM mean and total variance to within 1e-10.
    """
    rng = np.random.default_rng(2026)
    B, T = 10, 5

    means = rng.uniform(-5.0, 5.0, size=(B, T))
    variances = rng.uniform(0.1, 5.0, size=(B, T))
    raw_w = rng.uniform(0.1, 2.0, size=(B, T))
    weights = raw_w / np.sum(raw_w, axis=-1, keepdims=True)

    # Original mixture total mean and variance (Law of Total Variance)
    orig_mean = np.sum(weights * means, axis=-1)  # (B,)
    orig_var = (
        np.sum(weights * (variances + (means - orig_mean[:, np.newaxis]) ** 2), axis=-1)
    )  # (B,)

    # Split mixture into 4T components
    means_split, vars_split, weights_split = huber_table1_split_1d(
        means, variances, weights=weights
    )

    assert means_split.shape == (B, 4 * T)
    assert vars_split.shape == (B, 4 * T)
    assert weights_split.shape == (B, 4 * T)

    # Split mixture total mean and variance
    split_mean = np.sum(weights_split * means_split, axis=-1)  # (B,)
    split_var = np.sum(
        weights_split * (vars_split + (means_split - split_mean[:, np.newaxis]) ** 2),
        axis=-1,
    )  # (B,)

    np.testing.assert_allclose(
        split_mean,
        orig_mean,
        atol=1e-10,
        err_msg="Ensemble mean of split mixture deviates from original GMM mean",
    )
    np.testing.assert_allclose(
        split_var,
        orig_var,
        atol=1e-10,
        err_msg="Total variance of split mixture deviates from original GMM total variance",
    )


def test_splitting_accuracy_superiority():
    """
    Validates that Option B component splitting improves Taylor approximation accuracy
    over unsplit Option A on the Huber (2008) bimodal benchmark mixture:
    mu = [0, 2], sigma^2 = [1.0, 0.5], weights = [0.5, 0.5].
    Asserts |H_2_split - H_quad| < |H_2_unsplit - H_quad| (significant error reduction).
    """
    mu_orig = np.array([[0.0, 2.0]])
    var_orig = np.array([[1.0, 0.5]])
    weights_orig = np.array([[0.5, 0.5]])

    # High-precision numerical quadrature ground truth
    def gmm_pdf(y: float) -> float:
        c1 = 0.5 * np.exp(-0.5 * (y - 0.0) ** 2 / 1.0) / np.sqrt(2.0 * np.pi * 1.0)
        c2 = 0.5 * np.exp(-0.5 * (y - 2.0) ** 2 / 0.5) / np.sqrt(2.0 * np.pi * 0.5)
        return float(c1 + c2)

    def entropy_integrand(y: float) -> float:
        p = gmm_pdf(y)
        return -p * np.log2(p) if p > 1e-300 else 0.0

    h_quad, _ = quad(entropy_integrand, -15.0, 15.0, epsabs=1e-12, epsrel=1e-12)

    # 1. Unsplit Option A
    h2_unsplit = huber_entropy_1d_numpy(
        mu_orig, var_orig, weights=weights_orig, enable_splitting=False
    )
    err_unsplit = float(abs(h2_unsplit[0] - h_quad))

    # 2. Split Option B (default moment-preserving)
    h2_split = huber_entropy_1d_numpy(
        mu_orig, var_orig, weights=weights_orig, enable_splitting=True
    )
    err_split = float(abs(h2_split[0] - h_quad))

    # Assert strict superiority and substantial error reduction (> 65% error reduction)
    assert err_split < err_unsplit, (
        f"Splitting error {err_split} must be strictly less than unsplit error {err_unsplit}"
    )
    assert err_split <= 0.35 * err_unsplit, (
        f"Splitting error {err_split} did not achieve >=65% error reduction over {err_unsplit}"
    )

    # 3. Split Option B with raw Table I constants also beats unsplit
    h2_split_raw = huber_entropy_1d_numpy(
        mu_orig, var_orig, weights=weights_orig, enable_splitting=True, preserve_moments=False
    )
    err_split_raw = float(abs(h2_split_raw[0] - h_quad))
    assert err_split_raw < err_unsplit, (
        f"Raw Table I splitting error {err_split_raw} must be strictly less than unsplit error {err_unsplit}"
    )


def test_enable_splitting_flag_toggle():
    """
    Asserts behavior when toggling enable_splitting:
    - enable_splitting=False exactly matches default pure leaf Option A.
    - enable_splitting=True invokes the (B, 4T, T) Huber Remark 1 kernel.
    - return_bounds=True works seamlessly with enable_splitting=True.
    """
    rng = np.random.default_rng(42)
    B, T = 8, 4

    means = rng.uniform(-3.0, 3.0, size=(B, T))
    variances = rng.uniform(0.5, 3.0, size=(B, T))

    # Default call vs explicit enable_splitting=False
    h2_default = huber_entropy_1d_numpy(means, variances)
    h2_unsplit = huber_entropy_1d_numpy(means, variances, enable_splitting=False)
    np.testing.assert_allclose(h2_default, h2_unsplit, atol=1e-14)

    # Explicit enable_splitting=True
    h2_split = huber_entropy_1d_numpy(means, variances, enable_splitting=True)
    assert h2_split.shape == (B,)
    assert isinstance(h2_split, np.ndarray)
    # Different from unsplit on non-degenerate mixtures
    assert not np.allclose(h2_default, h2_split, atol=1e-6)

    # return_bounds=True with enable_splitting=True
    h_l, h_2_split_b, h_u = huber_entropy_1d_numpy(
        means, variances, enable_splitting=True, return_bounds=True
    )
    assert isinstance(h_l, np.ndarray) and h_l.shape == (B,)
    assert isinstance(h_2_split_b, np.ndarray) and h_2_split_b.shape == (B,)
    assert isinstance(h_u, np.ndarray) and h_u.shape == (B,)

    np.testing.assert_allclose(h_2_split_b, h2_split, atol=1e-14)
    # Theoretical bounds must enclose the entropy
    assert np.all(h_l <= h_u + 1e-12)


def test_splitting_with_1d_input_and_custom_weights():
    """
    Asserts that 1D inputs of shape (T,) and custom non-uniform weights work
    properly with enable_splitting=True, matching batch 2D operations.
    """
    T = 6
    rng = np.random.default_rng(123)

    means_1d = rng.uniform(-2.0, 2.0, size=(T,))
    variances_1d = rng.uniform(0.2, 2.0, size=(T,))
    raw_w = rng.uniform(0.1, 1.0, size=(T,))
    weights_1d = raw_w / np.sum(raw_w)

    # 1D input call
    h2_1d = huber_entropy_1d_numpy(
        means_1d, variances_1d, weights=weights_1d, enable_splitting=True
    )
    assert isinstance(h2_1d, np.ndarray)
    assert h2_1d.shape == (1,)

    # 2D batch call (B=1, T)
    means_2d = means_1d[np.newaxis, :]
    vars_2d = variances_1d[np.newaxis, :]
    weights_2d = weights_1d[np.newaxis, :]

    h2_2d = huber_entropy_1d_numpy(
        means_2d, vars_2d, weights=weights_2d, enable_splitting=True
    )
    np.testing.assert_allclose(h2_1d, h2_2d, atol=1e-14)

    # Verify split component extraction for 1D input
    means_sp, vars_sp, w_sp = huber_table1_split_1d(
        means_1d, variances_1d, weights=weights_1d
    )
    assert means_sp.shape == (1, 4 * T)
    assert vars_sp.shape == (1, 4 * T)
    assert w_sp.shape == (1, 4 * T)
    np.testing.assert_allclose(np.sum(w_sp, axis=-1), 1.0, atol=1e-12)


def test_splitting_empty_batch():
    """
    Asserts that empty batch (B=0) inputs return shape (0,) without error.
    """
    means_empty = np.empty((0, 5), dtype=np.float64)
    vars_empty = np.empty((0, 5), dtype=np.float64)

    # Single output
    res = huber_entropy_1d_numpy(means_empty, vars_empty, enable_splitting=True)
    assert isinstance(res, np.ndarray)
    assert res.shape == (0,)

    # Bounded output
    h_l, h_2, h_u = huber_entropy_1d_numpy(
        means_empty, vars_empty, enable_splitting=True, return_bounds=True
    )
    assert h_l.shape == (0,)
    assert h_2.shape == (0,)
    assert h_u.shape == (0,)

    # huber_table1_split_1d with empty batch
    m_sp, v_sp, w_sp = huber_table1_split_1d(means_empty, vars_empty)
    assert m_sp.shape == (0, 20)
    assert v_sp.shape == (0, 20)
    assert w_sp.shape == (0, 20)


def test_single_gaussian_limit_with_splitting():
    """
    Reviewer Feedback Test:
    Verifies that for T=1 with enable_splitting=True, the computed H_2 matches
    the exact analytical Shannon differential entropy of a single Gaussian:
        H(Y) = 0.5 * log2(2 * pi * e * sigma^2)
    within 1e-12 bits across various means and variances.
    """
    means_test = [-5.0, 0.0, 3.5, 10.0]
    vars_test = [1e-4, 0.1, 1.0, 2.5, 25.0]

    for mu in means_test:
        for var in vars_test:
            mu_arr = np.array([[mu]])
            var_arr = np.array([[var]])

            h2_split = huber_entropy_1d_numpy(mu_arr, var_arr, enable_splitting=True)
            analytic_shannon = 0.5 * np.log2(2.0 * np.pi * np.e * var)

            assert np.isclose(
                h2_split[0], analytic_shannon, atol=1e-12
            ), f"Failed for mu={mu}, var={var}: got {h2_split[0]}, expected {analytic_shannon}"

    # Also test across a batch with shape (B, 1)
    B = 20
    rng = np.random.default_rng(2026)
    b_means = rng.uniform(-10.0, 10.0, size=(B, 1))
    b_vars = rng.uniform(0.01, 10.0, size=(B, 1))

    b_h2_split = huber_entropy_1d_numpy(b_means, b_vars, enable_splitting=True)
    b_analytic = 0.5 * np.log2(2.0 * np.pi * np.e * b_vars[:, 0])

    np.testing.assert_allclose(
        b_h2_split,
        b_analytic,
        atol=1e-12,
        err_msg="Batch T=1 with enable_splitting=True does not match analytic Shannon entropy within 1e-12",
    )

