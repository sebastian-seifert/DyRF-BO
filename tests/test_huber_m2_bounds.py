"""
Unit tests for Milestone 2: Information-Theoretic Bounds (H_l & H_u) and Validation Guards.

Reference:
- Huber, Bailey, Durrant-Whyte, Hanebeck (IEEE MFI 2008):
  "On Entropy Approximation for Gaussian Mixture Random Vectors" (Theorems 2 & 3)
- Specification: detailed_huber_implementation_plan.md (Milestone 2)
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy.integrate import quad

from dyrf_bo.entropy import huber_entropy_1d_numpy, huber_entropy_bounds_1d_numpy


def test_strict_bounding_property():
    """
    Asserts Huber Theorems 2 & 3: For any valid GMM, the true differential entropy H_quad
    satisfies H_l <= H_quad <= H_u within numerical integration tolerance,
    and H_l <= H_u strictly for all points.
    Tests across T in {2, 5, 10} with diverse random means and variances.
    """
    rng = np.random.default_rng(2026)

    for T in [2, 5, 10]:
        num_configs = 15  # 15 configs per T -> 45 total configs
        for _ in range(num_configs):
            means = rng.uniform(low=-5.0, high=5.0, size=(1, T))
            variances = rng.uniform(low=0.1, high=5.0, size=(1, T))

            # Mix of uniform and random non-uniform weights
            if rng.random() > 0.5:
                raw_w = rng.uniform(0.1, 1.0, size=(1, T))
                weights = raw_w / np.sum(raw_w)
            else:
                weights = None

            # Calculate bounds and H_2
            h_l, h_2, h_u = huber_entropy_1d_numpy(
                means, variances, weights=weights, return_bounds=True
            )
            bounds_l, bounds_u = huber_entropy_bounds_1d_numpy(
                means, variances, weights=weights
            )

            # Contract consistency check
            np.testing.assert_allclose(h_l, bounds_l, atol=1e-14)
            np.testing.assert_allclose(h_u, bounds_u, atol=1e-14)

            # 1. Strict bound order: H_l <= H_u
            assert h_l[0] <= h_u[0] + 1e-12, (
                f"Lower bound {h_l[0]} strictly exceeded upper bound {h_u[0]} for T={T}"
            )

            # 2. Compute true differential entropy via scipy.integrate.quad
            w_vec = np.full(T, 1.0 / T) if weights is None else weights[0]
            m_vec = means[0]
            v_vec = variances[0]
            s_vec = np.sqrt(v_vec)

            def pdf(y):
                comp_vals = (
                    w_vec
                    * (1.0 / (np.sqrt(2.0 * np.pi) * s_vec))
                    * np.exp(-0.5 * ((y - m_vec) / s_vec) ** 2)
                )
                return np.sum(comp_vals)

            def integrand(y):
                p = pdf(y)
                if p <= 1e-300:
                    return 0.0
                return -p * np.log2(p)

            # Integration range covers +/- 7 standard deviations around extremes
            y_min = np.min(m_vec - 7.0 * s_vec)
            y_max = np.max(m_vec + 7.0 * s_vec)

            h_quad, _ = quad(integrand, y_min, y_max, epsabs=1e-8, epsrel=1e-8, limit=200)

            # Invariant: H_l <= H_quad <= H_u within numerical tolerance
            tol = 1e-6
            assert h_l[0] <= h_quad + tol, (
                f"H_l violation: H_l={h_l[0]:.6f} > H_quad={h_quad:.6f} for T={T}, diff={h_l[0] - h_quad}"
            )
            assert h_quad <= h_u[0] + tol, (
                f"H_u violation: H_quad={h_quad:.6f} > H_u={h_u[0]:.6f} for T={T}, diff={h_quad - h_u[0]}"
            )


def test_single_gaussian_bounds():
    """
    Asserts that for T=1 and sigma^2=2.0:
    - H_l == 0.5 * log2(4 * pi * sigma^2)
    - H_2 == H_u == 0.5 * log2(2 * pi * e * sigma^2)
    within 1e-12 bits tolerance.
    """
    sigma_sq = 2.0
    means = np.array([[3.5]], dtype=np.float64)
    variances = np.array([[sigma_sq]], dtype=np.float64)

    h_l_expected = 0.5 * np.log2(4.0 * np.pi * sigma_sq)
    h_exact_expected = 0.5 * np.log2(2.0 * np.pi * np.e * sigma_sq)

    h_l, h_2, h_u = huber_entropy_1d_numpy(means, variances, return_bounds=True)

    np.testing.assert_allclose(
        h_l[0],
        h_l_expected,
        atol=1e-12,
        rtol=0,
        err_msg="H_l did not match theoretical single-Gaussian limit",
    )
    np.testing.assert_allclose(
        h_2[0],
        h_exact_expected,
        atol=1e-12,
        rtol=0,
        err_msg="H_2 did not match theoretical single-Gaussian entropy",
    )
    np.testing.assert_allclose(
        h_u[0],
        h_exact_expected,
        atol=1e-12,
        rtol=0,
        err_msg="H_u did not match theoretical single-Gaussian upper bound",
    )

    # Also test via direct bounds function
    bounds_l, bounds_u = huber_entropy_bounds_1d_numpy(means, variances)
    np.testing.assert_allclose(bounds_l[0], h_l_expected, atol=1e-12)
    np.testing.assert_allclose(bounds_u[0], h_exact_expected, atol=1e-12)


def test_single_gaussian_bounds_sweep():
    """
    Asserts exact theoretical formulas for T=1 across a wide range of means and variances:
    sigma^2 in {1e-4, 0.1, 1.0, 2.0, 25.0}, mu in {-10.0, 0.0, 5.0}.
    """
    test_means = [-10.0, 0.0, 5.0]
    test_vars = [1e-4, 0.1, 1.0, 2.0, 25.0]

    for mu in test_means:
        for var in test_vars:
            h_l_expected = 0.5 * np.log2(4.0 * np.pi * var)
            h_exact_expected = 0.5 * np.log2(2.0 * np.pi * np.e * var)

            m_arr = np.array([[mu]], dtype=np.float64)
            v_arr = np.array([[var]], dtype=np.float64)

            h_l, h_2, h_u = huber_entropy_1d_numpy(m_arr, v_arr, return_bounds=True)
            np.testing.assert_allclose(h_l[0], h_l_expected, atol=1e-12)
            np.testing.assert_allclose(h_2[0], h_exact_expected, atol=1e-12)
            np.testing.assert_allclose(h_u[0], h_exact_expected, atol=1e-12)
            assert h_l[0] < h_u[0], "Lower bound must be strictly less than upper bound"


def test_asymptotic_separation_limit():
    """
    Asserts that for widely separated components:
    T=3, mu=[-1000, 0, 1000], sigma^2=[1, 1, 1],
    cross-terms vanish and H_2 approaches H_u = log2(3) + 0.5 * log2(2 * pi * e)
    within 1e-6 bits tolerance.
    """
    means = np.array([[-1000.0, 0.0, 1000.0]], dtype=np.float64)
    variances = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)

    h_l, h_2, h_u = huber_entropy_1d_numpy(means, variances, return_bounds=True)

    h_u_expected = np.log2(3.0) + 0.5 * np.log2(2.0 * np.pi * np.e * 1.0)
    np.testing.assert_allclose(h_u[0], h_u_expected, atol=1e-12)

    # Asymptotic convergence: |H_2 - H_u| < 1e-6
    abs_diff = abs(h_2[0] - h_u[0])
    assert abs_diff < 1e-6, (
        f"Asymptotic separation difference |H_2 - H_u| = {abs_diff} exceeded 1e-6"
    )

    # Also check with non-uniform weights
    weights = np.array([[0.2, 0.5, 0.3]], dtype=np.float64)
    h_l_w, h_2_w, h_u_w = huber_entropy_1d_numpy(
        means, variances, weights=weights, return_bounds=True
    )
    discrete_h = -(0.2 * np.log2(0.2) + 0.5 * np.log2(0.5) + 0.3 * np.log2(0.3))
    h_u_w_expected = discrete_h + 0.5 * np.log2(2.0 * np.pi * np.e * 1.0)
    np.testing.assert_allclose(h_u_w[0], h_u_w_expected, atol=1e-12)
    assert abs(h_2_w[0] - h_u_w[0]) < 1e-6, (
        f"Non-uniform asymptotic separation difference |H_2 - H_u| = {abs(h_2_w[0] - h_u_w[0])} exceeded 1e-6"
    )


def test_batch_bounds_broadcasting():
    """
    Asserts that vectorized batch calculation of bounds over B=64, T=10 matches
    point-by-point iterative evaluation within 1e-14 bits tolerance.
    """
    rng = np.random.default_rng(789)
    B, T = 64, 10
    means = rng.normal(size=(B, T))
    variances = rng.uniform(0.2, 4.0, size=(B, T))

    # Vectorized computation
    h_l_batch, h_u_batch = huber_entropy_bounds_1d_numpy(means, variances)
    assert h_l_batch.shape == (B,)
    assert h_u_batch.shape == (B,)

    # Iterative computation
    h_l_loop = np.empty(B, dtype=np.float64)
    h_u_loop = np.empty(B, dtype=np.float64)
    for i in range(B):
        l_i, u_i = huber_entropy_bounds_1d_numpy(means[i : i + 1], variances[i : i + 1])
        h_l_loop[i] = l_i[0]
        h_u_loop[i] = u_i[0]

    np.testing.assert_allclose(h_l_batch, h_l_loop, atol=1e-14, rtol=0)
    np.testing.assert_allclose(h_u_batch, h_u_loop, atol=1e-14, rtol=0)


def test_return_bounds_contract():
    """
    Asserts return type and shape contract:
    - return_bounds=False returns 1D ndarray of shape (B,)
    - return_bounds=True returns a 3-element tuple (H_l, H_2, H_u) each of shape (B,)
    - huber_entropy_bounds_1d_numpy returns a 2-element tuple (H_l, H_u) each of shape (B,)
    - 1D input (T,) returns shape (1,)
    - Empty batch (0, T) returns empty shape (0,)
    """
    B, T = 8, 4
    rng = np.random.default_rng(42)
    means = rng.normal(size=(B, T))
    variances = rng.uniform(0.5, 2.0, size=(B, T))

    # 1. Default return_bounds=False
    res_default = huber_entropy_1d_numpy(means, variances, return_bounds=False)
    assert isinstance(res_default, np.ndarray), "Default return must be np.ndarray"
    assert res_default.shape == (B,), f"Expected shape ({B},), got {res_default.shape}"

    # 2. return_bounds=True
    res_bounds = huber_entropy_1d_numpy(means, variances, return_bounds=True)
    assert isinstance(res_bounds, tuple), "return_bounds=True must return a tuple"
    assert len(res_bounds) == 3, f"Expected 3 elements, got {len(res_bounds)}"
    h_l, h_2, h_u = res_bounds
    for name, arr in [("H_l", h_l), ("H_2", h_2), ("H_u", h_u)]:
        assert isinstance(arr, np.ndarray), f"{name} must be np.ndarray"
        assert arr.shape == (B,), f"{name} shape must be ({B},), got {arr.shape}"

    # H_2 should be identical between both calls
    np.testing.assert_allclose(res_default, h_2, atol=1e-15)

    # 3. Direct bounds function
    bounds_pair = huber_entropy_bounds_1d_numpy(means, variances)
    assert isinstance(bounds_pair, tuple), "huber_entropy_bounds_1d_numpy must return tuple"
    assert len(bounds_pair) == 2, f"Expected 2 elements, got {len(bounds_pair)}"
    np.testing.assert_allclose(bounds_pair[0], h_l, atol=1e-15)
    np.testing.assert_allclose(bounds_pair[1], h_u, atol=1e-15)

    # 4. 1D input shape (T,)
    means_1d = np.array([0.0, 1.0, 2.0])
    vars_1d = np.array([1.0, 1.0, 1.0])
    h2_1d = huber_entropy_1d_numpy(means_1d, vars_1d, return_bounds=False)
    assert isinstance(h2_1d, np.ndarray)
    assert h2_1d.shape == (1,)

    triplet_1d = huber_entropy_1d_numpy(means_1d, vars_1d, return_bounds=True)
    assert isinstance(triplet_1d, tuple) and len(triplet_1d) == 3
    assert triplet_1d[0].shape == (1,)
    assert triplet_1d[1].shape == (1,)
    assert triplet_1d[2].shape == (1,)

    bounds_1d = huber_entropy_bounds_1d_numpy(means_1d, vars_1d)
    assert isinstance(bounds_1d, tuple) and len(bounds_1d) == 2
    assert bounds_1d[0].shape == (1,)
    assert bounds_1d[1].shape == (1,)

    # 5. Empty batch shape (0, T)
    empty_means = np.empty((0, T))
    empty_vars = np.empty((0, T))
    assert huber_entropy_1d_numpy(empty_means, empty_vars, return_bounds=False).shape == (0,)
    empty_triplet = huber_entropy_1d_numpy(empty_means, empty_vars, return_bounds=True)
    assert empty_triplet[0].shape == (0,)
    assert empty_triplet[1].shape == (0,)
    assert empty_triplet[2].shape == (0,)
    empty_bounds = huber_entropy_bounds_1d_numpy(empty_means, empty_vars)
    assert empty_bounds[0].shape == (0,)
    assert empty_bounds[1].shape == (0,)


def test_nan_inf_negative_variance_guards():
    """
    Asserts that ValueError is raised on invalid numerical inputs:
    - NaN in means
    - Inf in means
    - NaN in variances
    - Inf in variances
    - Negative variance (< 0)
    - NaN or Inf in weights
    Tests both huber_entropy_1d_numpy and huber_entropy_bounds_1d_numpy.
    """
    valid_means = np.array([[0.0, 1.0]], dtype=np.float64)
    valid_vars = np.array([[1.0, 2.0]], dtype=np.float64)

    functions_to_test = [
        lambda m, v, **kw: huber_entropy_1d_numpy(m, v, **kw),
        lambda m, v, **kw: huber_entropy_bounds_1d_numpy(m, v, **kw),
    ]

    for fn in functions_to_test:
        # 1. NaN in means
        nan_means = np.array([[0.0, np.nan]], dtype=np.float64)
        with pytest.raises(ValueError, match="(?i)nan|finite"):
            fn(nan_means, valid_vars)

        # 2. Inf in means
        inf_means = np.array([[0.0, np.inf]], dtype=np.float64)
        with pytest.raises(ValueError, match="(?i)inf|finite"):
            fn(inf_means, valid_vars)

        neg_inf_means = np.array([[-np.inf, 1.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="(?i)inf|finite"):
            fn(neg_inf_means, valid_vars)

        # 3. NaN in variances
        nan_vars = np.array([[1.0, np.nan]], dtype=np.float64)
        with pytest.raises(ValueError, match="(?i)nan|finite"):
            fn(valid_means, nan_vars)

        # 4. Inf in variances
        inf_vars = np.array([[1.0, np.inf]], dtype=np.float64)
        with pytest.raises(ValueError, match="(?i)inf|finite"):
            fn(valid_means, inf_vars)

        # 5. Negative variances (< 0)
        neg_vars = np.array([[1.0, -0.01]], dtype=np.float64)
        with pytest.raises(ValueError, match="(?i)negative"):
            fn(valid_means, neg_vars)

        # 6. NaN in weights
        with pytest.raises(ValueError, match="(?i)nan|finite"):
            fn(valid_means, valid_vars, weights=np.array([[0.5, np.nan]]))

        # 7. Inf in weights
        with pytest.raises(ValueError, match="(?i)inf|finite"):
            fn(valid_means, valid_vars, weights=np.array([[0.5, np.inf]]))


def test_zero_variance_clipping_bounds():
    """
    Asserts that variances of 0.0 are non-negative and properly clipped by min_var,
    yielding finite and consistent bounds without errors.
    """
    means = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
    zero_vars = np.zeros((1, 3), dtype=np.float64)

    h_l, h_2, h_u = huber_entropy_1d_numpy(means, zero_vars, min_var=1e-6, return_bounds=True)

    assert np.all(np.isfinite(h_l))
    assert np.all(np.isfinite(h_2))
    assert np.all(np.isfinite(h_u))
    assert h_l[0] <= h_u[0]
