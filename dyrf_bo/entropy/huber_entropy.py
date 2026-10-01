"""
Closed-Form Huber et al. (2008) Entropy Approximation for Gaussian Mixture Random Vectors.

Reference:
    Huber, Bailey, Durrant-Whyte, and Hanebeck.
    "On Entropy Approximation for Gaussian Mixture Random Vectors."
    2008 IEEE International Conference on Multisensor Fusion and Integration
    for Intelligent Systems (MFI 2008), pp. 181-188.
"""

from typing import Literal, Optional, Tuple, Union, overload
import warnings
import numpy as np

LN2: float = float(np.log(2.0))
INV_LN2: float = float(1.0 / np.log(2.0))
HALF_LOG_2PI: float = float(0.5 * np.log(2.0 * np.pi))
EPSILON_LOG_WEIGHT: float = 1e-300
MIN_SAFE_VAR: float = 1e-15

# ==============================================================================
# Huber (2008) Table I Splitting Library Constants for Standard Normal N(0, 1)
# ==============================================================================

# Raw weights from Table I: sum is ~0.99999999998
_RAW_TABLE1_WEIGHTS = np.array(
    [0.12738084098, 0.37261915901, 0.37261915901, 0.12738084098], dtype=np.float64
)
_RAW_TABLE1_WEIGHTS.flags.writeable = False

# Center locations from Table I: sum(w * mu) == 0.0
_RAW_TABLE1_MEANS = np.array(
    [-1.41312052330, -0.44973059608, 0.44973059608, 1.41312052330], dtype=np.float64
)
_RAW_TABLE1_MEANS.flags.writeable = False

# Standard deviations from Table I: 0.51751260421
_RAW_TABLE1_STDS = np.array(
    [0.51751260421, 0.51751260421, 0.51751260421, 0.51751260421], dtype=np.float64
)
_RAW_TABLE1_STDS.flags.writeable = False

# Normalized weights to strictly enforce sum == 1.0 within float64 precision
_NORM_TABLE1_WEIGHTS = _RAW_TABLE1_WEIGHTS / np.sum(_RAW_TABLE1_WEIGHTS)
_NORM_TABLE1_WEIGHTS.flags.writeable = False

# Variance-preserving standard deviation ensuring sum(w * (mu^2 + sigma^2)) == 1.0
_TARGET_SIGMA2 = (1.0 - np.sum(_NORM_TABLE1_WEIGHTS * _RAW_TABLE1_MEANS ** 2)) / np.sum(
    _NORM_TABLE1_WEIGHTS
)
_MOMENT_MATCHED_STDS = np.full(4, np.sqrt(_TARGET_SIGMA2), dtype=np.float64)
_MOMENT_MATCHED_STDS.flags.writeable = False

# Public immutable constants
TABLE1_WEIGHTS: np.ndarray = _RAW_TABLE1_WEIGHTS
TABLE1_MEANS: np.ndarray = _RAW_TABLE1_MEANS
TABLE1_STDS: np.ndarray = _RAW_TABLE1_STDS
TABLE1_STDS_MOMENT_MATCHED: np.ndarray = _MOMENT_MATCHED_STDS
 
# GPU acceleration hooks & flags
from dyrf_bo.entropy.huber_entropy_gpu import (
    HAS_CUPY,
    HAS_GPU,
    clear_gpu_memory_pool,
    get_dynamic_huber_batch_size_gpu,
    huber_entropy_1d_cupy,
)


def _fast_logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable vectorized LogSumExp along the specified axis.
    Guaranteed zero warnings for all-negative-infinity slices.

    Parameters
    ----------
    a : np.ndarray
        Input array.
    axis : int, default=-1
        Axis along which the log-sum-exp is computed.

    Returns
    -------
    out : np.ndarray
        Array with the specified axis reduced.
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    is_neg_inf = np.isneginf(a_max)
    safe_a_max = np.where(is_neg_inf, 0.0, a_max)
    tmp = np.exp(a - safe_a_max)
    sum_tmp = np.sum(tmp, axis=axis, keepdims=True)
    safe_sum = np.maximum(sum_tmp, 1e-300)
    out = np.where(is_neg_inf, -np.inf, np.log(safe_sum) + safe_a_max)
    return np.squeeze(out, axis=axis)


def _validate_and_preprocess_inputs(
    means: np.ndarray,
    variances: np.ndarray,
    weights: Optional[np.ndarray] = None,
    min_var: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Validates shapes, numerical sanity (no NaN/Inf, non-negative variances),
    and standardizes inputs to 2D float64 arrays of shape (B, T).

    Parameters
    ----------
    means : np.ndarray
        Component means of shape (B, T) or (T,).
    variances : np.ndarray
        Component variances of shape (B, T) or (T,).
    weights : Optional[np.ndarray], default=None
        Component weights of shape (B, T), (1, T), (T,), or None.
    min_var : float, default=1e-6
        Numerical floor for component variances.

    Returns
    -------
    means_arr : np.ndarray
        2D array of component means of shape (B, T).
    vars_clipped : np.ndarray
        2D array of component variances clipped to safe floor of shape (B, T).
    w_norm : np.ndarray
        2D normalized component weights of shape (B, T).
    B : int
        Batch size.
    T : int
        Number of mixture components.

    Raises
    ------
    ValueError
        If inputs contain NaN/Inf, negative variances (< 0), mismatched shapes,
        invalid dimensions, or invalid weights.
    """
    means_arr = np.asarray(means, dtype=np.float64)
    vars_arr = np.asarray(variances, dtype=np.float64)

    # 1. Finite numerical checks for means and variances
    if not np.all(np.isfinite(means_arr)):
        raise ValueError("means must contain only finite real values (found NaN or Inf)")
    if not np.all(np.isfinite(vars_arr)):
        raise ValueError("variances must contain only finite real values (found NaN or Inf)")

    # 2. Strict non-negative variance check
    if np.any(vars_arr < 0.0):
        raise ValueError("Component variances must be non-negative (found negative variance < 0)")

    # 3. Shape dimension checks and standardization to 2D (B, T)
    if means_arr.ndim == 1:
        means_arr = means_arr[np.newaxis, :]
    elif means_arr.ndim != 2:
        raise ValueError(f"means must be 1D or 2D array, got shape {means_arr.shape}")

    if vars_arr.ndim == 1:
        vars_arr = vars_arr[np.newaxis, :]
    elif vars_arr.ndim != 2:
        raise ValueError(f"variances must be 1D or 2D array, got shape {vars_arr.shape}")

    if means_arr.shape != vars_arr.shape:
        raise ValueError(
            f"Shape mismatch between means {means_arr.shape} and variances {vars_arr.shape}"
        )

    B, T = means_arr.shape
    if T < 1:
        raise ValueError(f"Number of mixture components T must be at least 1, got {T}")

    # 4. Variance floor clipping
    safe_min_var = max(float(min_var), MIN_SAFE_VAR)
    vars_clipped = np.maximum(vars_arr, safe_min_var)

    # 5. Process and validate mixture weights
    if weights is None:
        w_norm = np.full((B, T), 1.0 / T, dtype=np.float64)
    else:
        w_arr = np.asarray(weights, dtype=np.float64)
        if not np.all(np.isfinite(w_arr)):
            raise ValueError("weights must contain only finite real values (found NaN or Inf)")
        if np.any(w_arr < 0.0):
            raise ValueError("Component mixture weights must be non-negative")

        if w_arr.ndim == 1:
            if w_arr.shape[0] != T:
                raise ValueError(
                    f"weights 1D shape ({w_arr.shape[0]},) does not match number of components T={T}"
                )
            w_arr = np.tile(w_arr[np.newaxis, :], (B, 1))
        elif w_arr.ndim == 2:
            if w_arr.shape == (1, T) and B > 1:
                w_arr = np.tile(w_arr, (B, 1))
            elif w_arr.shape != (B, T):
                raise ValueError(
                    f"weights 2D shape {w_arr.shape} does not match (B, T)=({B}, {T})"
                )
        else:
            raise ValueError(f"weights must be 1D or 2D array, got shape {w_arr.shape}")

        w_sums = np.sum(w_arr, axis=-1, keepdims=True)
        if np.any(w_sums <= 0.0):
            raise ValueError("Component weights must sum to a strictly positive value")
        w_norm = w_arr / w_sums

    return means_arr, vars_clipped, w_norm, B, T


def _compute_bounds_bits(
    means_arr: np.ndarray,
    vars_clipped: np.ndarray,
    w_norm: np.ndarray,
    delta: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes lower bound H_l (Huber Theorem 2) and upper bound H_u (Huber Theorem 3)
    in bits for validated 2D inputs of shape (B, T).

    Parameters
    ----------
    means_arr : np.ndarray of shape (B, T)
        Component means.
    vars_clipped : np.ndarray of shape (B, T)
        Component variances with safety floor applied.
    w_norm : np.ndarray of shape (B, T)
        Normalized mixture weights.
    delta : Optional[np.ndarray] of shape (B, T, T), default=None
        Precomputed pairwise difference means_arr[:, :, None] - means_arr[:, None, :].

    Returns
    -------
    h_l_bits : np.ndarray of shape (B,)
        Lower bound differential entropy in bits.
    h_u_bits : np.ndarray of shape (B,)
        Upper bound differential entropy in bits.
    """
    if delta is None:
        delta = means_arr[:, :, np.newaxis] - means_arr[:, np.newaxis, :]

    # 1. Lower Bound H_l (Huber 2008, Theorem 2)
    s_matrix = vars_clipped[:, :, np.newaxis] + vars_clipped[:, np.newaxis, :]  # (B, T, T)
    log_z = -HALF_LOG_2PI - 0.5 * np.log(s_matrix) - 0.5 * (delta ** 2) / s_matrix

    log_w = np.where(w_norm > 0.0, np.log(np.maximum(w_norm, EPSILON_LOG_WEIGHT)), -np.inf)
    log_w_j = log_w[:, np.newaxis, :]  # (B, 1, T)

    ell_z = log_w_j + log_z  # (B, T, T)
    inner_lse = _fast_logsumexp(ell_z, axis=-1)  # (B, T)

    h_l_nats = -np.sum(w_norm * inner_lse, axis=-1)  # (B,)
    h_l_bits = h_l_nats * INV_LN2

    # 2. Upper Bound H_u (Huber 2008, Theorem 3)
    discrete_h_bits = -np.sum(
        np.where(w_norm > 0.0, w_norm * np.log2(np.maximum(w_norm, EPSILON_LOG_WEIGHT)), 0.0),
        axis=-1,
    )  # (B,)
    comp_entropy_bits = 0.5 * np.log2(2.0 * np.pi * np.e * vars_clipped)  # (B, T)
    weighted_comp_entropy = np.sum(w_norm * comp_entropy_bits, axis=-1)  # (B,)
    h_u_bits = discrete_h_bits + weighted_comp_entropy  # (B,)

    return h_l_bits, h_u_bits


def _compute_h2_bits(
    means_arr: np.ndarray,
    vars_clipped: np.ndarray,
    w_norm: np.ndarray,
    delta: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes second-order Taylor expansion entropy H_2 (Huber et al., 2008)
    in bits for validated 2D inputs of shape (B, T) without splitting.

    Parameters
    ----------
    means_arr : np.ndarray of shape (B, T)
        Component means.
    vars_clipped : np.ndarray of shape (B, T)
        Component variances with safety floor applied.
    w_norm : np.ndarray of shape (B, T)
        Normalized mixture weights.
    delta : Optional[np.ndarray] of shape (B, T, T), default=None
        Precomputed pairwise difference means_arr[:, :, None] - means_arr[:, None, :].

    Returns
    -------
    h2_bits : np.ndarray of shape (B,)
        Second-order Huber differential entropy in bits.
    """
    if delta is None:
        delta = means_arr[:, :, np.newaxis] - means_arr[:, np.newaxis, :]  # (B, T, T)

    vars_j = vars_clipped[:, np.newaxis, :]  # (B, 1, T)
    inv_vars_j = 1.0 / vars_j
    u = delta * inv_vars_j  # (mu_i - mu_j) / sigma_j^2

    log_w = np.where(w_norm > 0.0, np.log(np.maximum(w_norm, EPSILON_LOG_WEIGHT)), -np.inf)
    log_w_j = log_w[:, np.newaxis, :]  # (B, 1, T)

    log_norm = -HALF_LOG_2PI - 0.5 * np.log(vars_j) - 0.5 * (delta * u)
    ell = log_w_j + log_norm  # (B, T, T)

    # Numerically stable single-pass logsumexp and softmax responsibilities
    m = np.max(ell, axis=-1, keepdims=True)
    is_neg_inf = np.isneginf(m)
    safe_m = np.where(is_neg_inf, 0.0, m)
    tmp = np.exp(ell - safe_m)
    sum_tmp = np.sum(tmp, axis=-1, keepdims=True)
    safe_sum = np.maximum(sum_tmp, 1e-300)

    log_f = np.squeeze(np.where(is_neg_inf, -np.inf, np.log(safe_sum) + safe_m), axis=-1)  # (B, T)
    responsibilities = np.where(is_neg_inf, 0.0, tmp / safe_sum)  # (B, T, T)

    grad_ratio = -np.sum(responsibilities * u, axis=-1)  # (B, T)
    second_ratio = np.sum(responsibilities * (u * u - inv_vars_j), axis=-1)  # (B, T)
    f_hessian = second_ratio - (grad_ratio ** 2)  # (B, T)

    h0 = -np.sum(w_norm * log_f, axis=-1)  # (B,)
    h2_nats = h0 - 0.5 * np.sum(w_norm * f_hessian * vars_clipped, axis=-1)  # (B,)
    h2_bits = h2_nats * INV_LN2

    return h2_bits


def _compute_h2_split_bits(
    means_orig: np.ndarray,
    vars_orig: np.ndarray,
    w_orig: np.ndarray,
    means_split: np.ndarray,
    vars_split: np.ndarray,
    w_split: np.ndarray,
) -> np.ndarray:
    """
    Computes split second-order Taylor expansion entropy H_2 (Huber et al., 2008)
    in bits enforcing Huber Remark 1:
    - Outer expectation evaluated over 4T split components (means_split, vars_split, w_split).
    - Inner mixture log density g(y) evaluated over the ORIGINAL T components
      (means_orig, vars_orig, w_orig).
    Cross-evaluation tensor has shape (B, 4T, T).

    Parameters
    ----------
    means_orig : np.ndarray of shape (B, T)
        Original component means.
    vars_orig : np.ndarray of shape (B, T)
        Original component variances.
    w_orig : np.ndarray of shape (B, T)
        Original normalized mixture weights.
    means_split : np.ndarray of shape (B, 4T)
        Split component evaluation centers.
    vars_split : np.ndarray of shape (B, 4T)
        Split component variances.
    w_split : np.ndarray of shape (B, 4T)
        Split component outer integration weights.

    Returns
    -------
    h2_bits : np.ndarray of shape (B,)
        Second-order Huber differential entropy in bits computed via splitting.
    """
    # delta: means_split (B, 4T, 1) - means_orig (B, 1, T) -> (B, 4T, T)
    delta = means_split[:, :, np.newaxis] - means_orig[:, np.newaxis, :]

    vars_j = vars_orig[:, np.newaxis, :]  # (B, 1, T)
    inv_vars_j = 1.0 / vars_j  # (B, 1, T)
    u = delta * inv_vars_j  # (B, 4T, T)

    log_w = np.where(w_orig > 0.0, np.log(np.maximum(w_orig, EPSILON_LOG_WEIGHT)), -np.inf)
    log_w_j = log_w[:, np.newaxis, :]  # (B, 1, T)

    log_norm = -HALF_LOG_2PI - 0.5 * np.log(vars_j) - 0.5 * (delta * u)
    ell = log_w_j + log_norm  # (B, 4T, T)

    # Numerically stable single-pass logsumexp and softmax responsibilities across T
    m = np.max(ell, axis=-1, keepdims=True)  # (B, 4T, 1)
    is_neg_inf = np.isneginf(m)
    safe_m = np.where(is_neg_inf, 0.0, m)
    tmp = np.exp(ell - safe_m)  # (B, 4T, T)
    sum_tmp = np.sum(tmp, axis=-1, keepdims=True)  # (B, 4T, 1)
    safe_sum = np.maximum(sum_tmp, 1e-300)

    log_g = np.squeeze(np.where(is_neg_inf, -np.inf, np.log(safe_sum) + safe_m), axis=-1)  # (B, 4T)
    responsibilities = np.where(is_neg_inf, 0.0, tmp / safe_sum)  # (B, 4T, T)

    # Derivative ratios of g at mu_split
    grad_ratio = -np.sum(responsibilities * u, axis=-1)  # (B, 4T)
    second_ratio = np.sum(responsibilities * (u * u - inv_vars_j), axis=-1)  # (B, 4T)
    f_hessian = second_ratio - (grad_ratio ** 2)  # (B, 4T)

    # Contraction over the 4T split components
    h0 = -np.sum(w_split * log_g, axis=-1)  # (B,)
    h2_nats = h0 - 0.5 * np.sum(w_split * f_hessian * vars_split, axis=-1)  # (B,)
    h2_bits = h2_nats * INV_LN2

    return h2_bits


def huber_table1_split_1d(
    means: np.ndarray,
    variances: np.ndarray,
    weights: Optional[np.ndarray] = None,
    min_var: float = 1e-6,
    preserve_moments: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits 1D GMM components using the Huber (2008) Table I library.
    Each of the T components is expanded into 4 subcomponents, producing
    an expanded mixture of size 4T.

    Parameters
    ----------
    means : np.ndarray
        Component means of shape (B, T) or (T,).
    variances : np.ndarray
        Component variances of shape (B, T) or (T,).
    weights : Optional[np.ndarray], default=None
        Component weights of shape (B, T), (1, T), (T,), or None (uniform 1/T).
    min_var : float, default=1e-6
        Numerical floor for component variances.
    preserve_moments : bool, default=True
        If True, scales standard deviations to enforce exact unit variance preservation
        sum(w_k * (mu_k^2 + sigma_k^2)) == 1.0 (to within float precision).
        If False, uses raw Table I constants (sigma = 0.51751260421).

    Returns
    -------
    means_split : np.ndarray of shape (B, 4T)
        Expanded component means.
    variances_split : np.ndarray of shape (B, 4T)
        Expanded component variances.
    weights_split : np.ndarray of shape (B, 4T)
        Expanded normalized component weights summing to 1 across 4T.
    """
    means_arr, vars_clipped, w_norm, B, T = _validate_and_preprocess_inputs(
        means, variances, weights=weights, min_var=min_var
    )

    if B == 0:
        return (
            np.empty((0, 4 * T), dtype=np.float64),
            np.empty((0, 4 * T), dtype=np.float64),
            np.empty((0, 4 * T), dtype=np.float64),
        )

    # Select parameters
    w_lib = _NORM_TABLE1_WEIGHTS
    mu_lib = _RAW_TABLE1_MEANS
    std_lib = _MOMENT_MATCHED_STDS if preserve_moments else _RAW_TABLE1_STDS

    # Vectorized expansion:
    # mu_split_{b, t, k} = mu_{b, t} + sigma_{b, t} * mu_lib_k
    # var_split_{b, t, k} = var_{b, t} * std_lib_k^2
    # w_split_{b, t, k} = w_{b, t} * w_lib_k
    std_orig = np.sqrt(vars_clipped)  # (B, T)
    means_expanded = (
        means_arr[:, :, np.newaxis] + std_orig[:, :, np.newaxis] * mu_lib[np.newaxis, np.newaxis, :]
    )  # (B, T, 4)
    vars_expanded = (
        vars_clipped[:, :, np.newaxis] * (std_lib ** 2)[np.newaxis, np.newaxis, :]
    )  # (B, T, 4)
    weights_expanded = (
        w_norm[:, :, np.newaxis] * w_lib[np.newaxis, np.newaxis, :]
    )  # (B, T, 4)

    # Flatten (B, T, 4) -> (B, 4T)
    means_split = means_expanded.reshape(B, 4 * T)
    vars_split = vars_expanded.reshape(B, 4 * T)
    weights_split = weights_expanded.reshape(B, 4 * T)

    return means_split, vars_split, weights_split


def huber_entropy_bounds_1d_numpy(
    means: np.ndarray,
    variances: np.ndarray,
    weights: Optional[np.ndarray] = None,
    min_var: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes closed-form information-theoretic lower and upper bounds (H_l, H_u)
    for 1D Gaussian Mixture Models (Huber et al., 2008, Theorems 2 & 3) in bits.

    Parameters
    ----------
    means : np.ndarray
        Component means of shape (B, T) or (T,).
    variances : np.ndarray
        Component variances of shape (B, T) or (T,).
    weights : Optional[np.ndarray], default=None
        Component mixture weights summing to 1 across T.
        Can be shape (T,), (1, T), or (B, T). If None, uniform weights 1/T are used.
    min_var : float, default=1e-6
        Numerical floor for component variances to ensure strict positivity.

    Returns
    -------
    h_l : np.ndarray
        1D array of shape (B,) containing the lower bound H_l in bits.
    h_u : np.ndarray
        1D array of shape (B,) containing the upper bound H_u in bits.

    Raises
    ------
    ValueError
        If inputs contain NaN or Inf values, negative variances (< 0),
        mismatched dimensions, non-positive weights, or T < 1.
    """
    means_arr, vars_clipped, w_norm, B, _ = _validate_and_preprocess_inputs(
        means, variances, weights=weights, min_var=min_var
    )

    if B == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    return _compute_bounds_bits(means_arr, vars_clipped, w_norm)


@overload
def huber_entropy_1d_numpy(
    means: np.ndarray,
    variances: np.ndarray,
    weights: Optional[np.ndarray] = None,
    min_var: float = 1e-6,
    enable_splitting: bool = False,
    return_bounds: Literal[False] = False,
    preserve_moments: bool = True,
) -> np.ndarray:
    ...


@overload
def huber_entropy_1d_numpy(
    means: np.ndarray,
    variances: np.ndarray,
    weights: Optional[np.ndarray] = None,
    min_var: float = 1e-6,
    enable_splitting: bool = False,
    return_bounds: Literal[True] = ...,
    preserve_moments: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@overload
def huber_entropy_1d_numpy(
    means: np.ndarray,
    variances: np.ndarray,
    weights: Optional[np.ndarray] = None,
    min_var: float = 1e-6,
    enable_splitting: bool = False,
    return_bounds: bool = False,
    preserve_moments: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ...


def huber_entropy_1d_numpy(
    means: np.ndarray,
    variances: np.ndarray,
    weights: Optional[np.ndarray] = None,
    min_var: float = 1e-6,
    enable_splitting: bool = False,
    return_bounds: bool = False,
    preserve_moments: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Computes closed-form second-order Huber (2008) GMM differential entropy H_2
    for 1D scalar mixture random variables in bits.

    Parameters
    ----------
    means : np.ndarray
        Component means of shape (B, T) or (T,).
    variances : np.ndarray
        Component variances of shape (B, T) or (T,).
    weights : Optional[np.ndarray], default=None
        Component mixture weights summing to 1 across T.
        Can be shape (T,), (1, T), or (B, T). If None, uniform weights 1/T are used.
    min_var : float, default=1e-6
        Numerical floor for component variances to ensure strict positivity
        and prevent division by zero or negative logarithms.
    enable_splitting : bool, default=False
        If False (default), evaluates pure leaf Option A (pairwise T x T evaluation).
        If True, splits each component into 4 subcomponents using Table I library
        and evaluates via asymmetric cross-tensor kernel (B, 4T, T) enforcing Huber Remark 1.
    return_bounds : bool, default=False
        If False, returns H_2 array of shape (B,).
        If True, returns bounding triplet tuple (H_l, H_2, H_u) of 1D arrays of shape (B,).
    preserve_moments : bool, default=True
        Used only when enable_splitting=True. If True, uses variance-preserving standard
        deviation to maintain exact mean and variance conservation. If False, uses raw
        Huber Table I constants.

    Returns
    -------
    result : Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        If return_bounds is False (default):
            1D array of shape (B,) containing second-order Huber differential entropy H_2 in bits.
        If return_bounds is True:
            Tuple (H_l, H_2, H_u) each of shape (B,) in bits, where H_l is the lower bound,
            H_2 is the second-order Taylor approximation, and H_u is the upper bound.

    Raises
    ------
    ValueError
        If inputs contain NaN or Inf values, negative variances (< 0),
        mismatched shapes, non-positive weights, or zero mixture components (T < 1).
    """
    means_arr, vars_clipped, w_norm, B, T = _validate_and_preprocess_inputs(
        means, variances, weights=weights, min_var=min_var
    )

    if B == 0:
        empty = np.empty((0,), dtype=np.float64)
        if return_bounds:
            return empty, empty.copy(), empty.copy()
        return empty

    if enable_splitting:
        means_sp, vars_sp, w_sp = huber_table1_split_1d(
            means_arr, vars_clipped, weights=w_norm, min_var=min_var, preserve_moments=preserve_moments
        )
        h2_bits = _compute_h2_split_bits(
            means_arr, vars_clipped, w_norm, means_sp, vars_sp, w_sp
        )
    else:
        delta = means_arr[:, :, np.newaxis] - means_arr[:, np.newaxis, :]
        h2_bits = _compute_h2_bits(means_arr, vars_clipped, w_norm, delta=delta)

    if return_bounds:
        delta_orig = means_arr[:, :, np.newaxis] - means_arr[:, np.newaxis, :]
        h_l_bits, h_u_bits = _compute_bounds_bits(
            means_arr, vars_clipped, w_norm, delta=delta_orig
        )
        return h_l_bits, h2_bits, h_u_bits

    return h2_bits


def _execute_chunk_gpu(
    b_means: np.ndarray,
    b_vars: np.ndarray,
    b_weights: np.ndarray,
    min_var: float = 1e-6,
    enable_splitting: bool = False,
    return_bounds: bool = False,
    preserve_moments: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Executes a single batch chunk on GPU with automatic OOM retry and batch halving.
    """
    from dyrf_bo.entropy import huber_entropy_gpu

    current_cp = getattr(huber_entropy_gpu, "cp", None)
    if current_cp is None:
        warnings.warn(
            "GPU backend requested but CuPy module not found. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return huber_entropy_1d_numpy(
            b_means,
            b_vars,
            weights=b_weights,
            min_var=min_var,
            enable_splitting=enable_splitting,
            return_bounds=return_bounds,
            preserve_moments=preserve_moments,
        )

    chunk_len = b_means.shape[0]
    if chunk_len == 0:
        out_dtype = (
            b_means.dtype
            if b_means.dtype in (np.float32, np.float64)
            else np.float64
        )
        empty = np.empty((0,), dtype=out_dtype)
        if return_bounds:
            return empty, empty.copy(), empty.copy()
        return empty

    sub_chunk_size = chunk_len

    while True:
        try:
            if sub_chunk_size >= chunk_len:
                cp_means = current_cp.asarray(b_means)
                cp_vars = current_cp.asarray(b_vars)
                cp_weights = current_cp.asarray(b_weights)

                res = huber_entropy_1d_cupy(
                    cp_means,
                    cp_vars,
                    weights_cp=cp_weights,
                    min_var=min_var,
                    enable_splitting=enable_splitting,
                    return_bounds=return_bounds,
                    preserve_moments=preserve_moments,
                )
                if return_bounds:
                    return (
                        current_cp.asnumpy(res[0]),
                        current_cp.asnumpy(res[1]),
                        current_cp.asnumpy(res[2]),
                    )
                return current_cp.asnumpy(res)
            else:
                out_dtype = (
                    b_means.dtype
                    if b_means.dtype in (np.float32, np.float64)
                    else np.float64
                )
                if return_bounds:
                    sub_hl = np.empty((chunk_len,), dtype=out_dtype)
                    sub_h2 = np.empty((chunk_len,), dtype=out_dtype)
                    sub_hu = np.empty((chunk_len,), dtype=out_dtype)
                else:
                    sub_h2 = np.empty((chunk_len,), dtype=out_dtype)

                for sub_start in range(0, chunk_len, sub_chunk_size):
                    sub_end = min(sub_start + sub_chunk_size, chunk_len)
                    cp_means = current_cp.asarray(b_means[sub_start:sub_end])
                    cp_vars = current_cp.asarray(b_vars[sub_start:sub_end])
                    cp_weights = current_cp.asarray(b_weights[sub_start:sub_end])

                    sub_res = huber_entropy_1d_cupy(
                        cp_means,
                        cp_vars,
                        weights_cp=cp_weights,
                        min_var=min_var,
                        enable_splitting=enable_splitting,
                        return_bounds=return_bounds,
                        preserve_moments=preserve_moments,
                    )
                    if return_bounds:
                        sub_hl[sub_start:sub_end] = current_cp.asnumpy(sub_res[0])
                        sub_h2[sub_start:sub_end] = current_cp.asnumpy(sub_res[1])
                        sub_hu[sub_start:sub_end] = current_cp.asnumpy(sub_res[2])
                    else:
                        sub_h2[sub_start:sub_end] = current_cp.asnumpy(sub_res)

                if return_bounds:
                    return sub_hl, sub_h2, sub_hu
                return sub_h2

        except Exception as e:
            is_oom = "out of memory" in str(e).lower() or isinstance(e, MemoryError)
            if hasattr(current_cp, "cuda") and hasattr(current_cp.cuda, "memory"):
                if isinstance(e, getattr(current_cp.cuda.memory, "OutOfMemoryError", ())):
                    is_oom = True

            if is_oom:
                clear_gpu_memory_pool()
                sub_chunk_size = sub_chunk_size // 2
                if sub_chunk_size < 1:
                    warnings.warn(
                        "GPU out of memory during Huber entropy computation. Falling back to CPU for this batch.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return huber_entropy_1d_numpy(
                        b_means,
                        b_vars,
                        weights=b_weights,
                        min_var=min_var,
                        enable_splitting=enable_splitting,
                        return_bounds=return_bounds,
                        preserve_moments=preserve_moments,
                    )
            else:
                raise


@overload
def huber_entropy_1d(
    means: np.ndarray,
    variances: np.ndarray,
    weights: Optional[np.ndarray] = None,
    min_var: float = 1e-6,
    enable_splitting: bool = False,
    return_bounds: Literal[False] = False,
    backend: str = "auto",
    batch_size: Union[str, int] = "auto",
    preserve_moments: bool = True,
) -> np.ndarray:
    ...


@overload
def huber_entropy_1d(
    means: np.ndarray,
    variances: np.ndarray,
    weights: Optional[np.ndarray] = None,
    min_var: float = 1e-6,
    enable_splitting: bool = False,
    return_bounds: Literal[True] = ...,
    backend: str = "auto",
    batch_size: Union[str, int] = "auto",
    preserve_moments: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@overload
def huber_entropy_1d(
    means: np.ndarray,
    variances: np.ndarray,
    weights: Optional[np.ndarray] = None,
    min_var: float = 1e-6,
    enable_splitting: bool = False,
    return_bounds: bool = False,
    backend: str = "auto",
    batch_size: Union[str, int] = "auto",
    preserve_moments: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ...


def huber_entropy_1d(
    means: np.ndarray,
    variances: np.ndarray,
    weights: Optional[np.ndarray] = None,
    min_var: float = 1e-6,
    enable_splitting: bool = False,
    return_bounds: bool = False,
    backend: str = "auto",
    batch_size: Union[str, int] = "auto",
    preserve_moments: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Unified entrypoint for closed-form Huber (2008) GMM differential entropy approximation
    with automatic CPU/GPU backend dispatching and dynamic batch chunking.

    Parameters
    ----------
    means : np.ndarray
        Component means of shape (B, T) or (T,).
    variances : np.ndarray
        Component variances of shape (B, T) or (T,).
    weights : Optional[np.ndarray], default=None
        Component mixture weights summing to 1 across T.
        Can be shape (T,), (1, T), or (B, T). If None, uniform weights 1/T are used.
    min_var : float, default=1e-6
        Numerical floor for component variances to ensure strict positivity.
    enable_splitting : bool, default=False
        If False (default), evaluates pure leaf Option A (pairwise T x T evaluation).
        If True, splits each component into 4 subcomponents using Table I library
        and evaluates via asymmetric cross-tensor kernel (B, 4T, T) enforcing Huber Remark 1.
    return_bounds : bool, default=False
        If False (default), returns H_2 array of shape (B,).
        If True, returns bounding triplet tuple (H_l, H_2, H_u) of 1D arrays of shape (B,).
    backend : str, default="auto"
        Execution backend: 'auto', 'cpu', or 'gpu'.
        'auto' will use GPU if CuPy and a CUDA device are available, else CPU.
        'gpu' forces GPU execution; if CUDA/CuPy is unavailable, it issues a warning and falls back to CPU.
    batch_size : Union[str, int], default="auto"
        Batch chunk size for memory management:
        'auto' dynamically resolves optimal batch size based on available VRAM / RAM.
        An integer specifies explicit chunk size (must be > 0).
    preserve_moments : bool, default=True
        Used only when enable_splitting=True. If True, uses variance-preserving standard
        deviation to maintain exact mean and variance conservation. If False, uses raw
        Huber Table I constants.

    Returns
    -------
    result : Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        If return_bounds is False (default):
            1D array of shape (B,) containing second-order Huber differential entropy H_2 in bits.
        If return_bounds is True:
            Tuple (H_l, H_2, H_u) each of shape (B,) in bits, where H_l is the lower bound,
            H_2 is the second-order Taylor approximation, and H_u is the upper bound.

    Raises
    ------
    ValueError
        If backend is not one of 'auto', 'cpu', 'gpu', or batch_size is invalid,
        or inputs contain invalid values or mismatched shapes.
    """
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"backend must be one of: 'auto', 'cpu', 'gpu', got {backend!r}")

    if isinstance(batch_size, str):
        if batch_size != "auto":
            raise ValueError(
                f"batch_size must be 'auto' or a positive integer, got {batch_size!r}"
            )
    elif isinstance(batch_size, (int, np.integer)):
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be 'auto' or a positive integer, got {batch_size!r}"
            )
    else:
        raise ValueError(
            f"batch_size must be 'auto' or a positive integer, got {type(batch_size).__name__}"
        )

    # Resolve backend
    current_has_gpu = globals().get("HAS_GPU", False)
    resolved_backend = backend
    if backend == "auto":
        resolved_backend = "gpu" if current_has_gpu else "cpu"
    elif backend == "gpu" and not current_has_gpu:
        warnings.warn(
            "GPU backend requested but CUDA/CuPy is unavailable. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        resolved_backend = "cpu"

    means_arr, vars_clipped, w_norm, B, T = _validate_and_preprocess_inputs(
        means, variances, weights=weights, min_var=min_var
    )

    if B == 0:
        empty = np.empty((0,), dtype=np.float64)
        if return_bounds:
            return empty, empty.copy(), empty.copy()
        return empty

    # Resolve chunk size
    if isinstance(batch_size, (int, np.integer)):
        chunk_size = int(batch_size)
    else:
        if resolved_backend == "gpu":
            chunk_size = get_dynamic_huber_batch_size_gpu(
                T, enable_splitting=enable_splitting, dtype=means_arr.dtype
            )
        else:
            chunk_size = min(B, 5000)

    # Monolithic execution if batch fits in single chunk
    if B <= chunk_size:
        if resolved_backend == "gpu":
            return _execute_chunk_gpu(
                means_arr,
                vars_clipped,
                w_norm,
                min_var=min_var,
                enable_splitting=enable_splitting,
                return_bounds=return_bounds,
                preserve_moments=preserve_moments,
            )
        else:
            return huber_entropy_1d_numpy(
                means_arr,
                vars_clipped,
                weights=w_norm,
                min_var=min_var,
                enable_splitting=enable_splitting,
                return_bounds=return_bounds,
                preserve_moments=preserve_moments,
            )

    # Chunked execution along batch dimension B
    out_dtype = (
        means_arr.dtype if means_arr.dtype in (np.float32, np.float64) else np.float64
    )
    if return_bounds:
        h_l_all = np.empty((B,), dtype=out_dtype)
        h_2_all = np.empty((B,), dtype=out_dtype)
        h_u_all = np.empty((B,), dtype=out_dtype)
    else:
        h_2_all = np.empty((B,), dtype=out_dtype)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        b_means = means_arr[start:end]
        b_vars = vars_clipped[start:end]
        b_weights = w_norm[start:end]

        if resolved_backend == "gpu":
            res = _execute_chunk_gpu(
                b_means,
                b_vars,
                b_weights,
                min_var=min_var,
                enable_splitting=enable_splitting,
                return_bounds=return_bounds,
                preserve_moments=preserve_moments,
            )
        else:
            res = huber_entropy_1d_numpy(
                b_means,
                b_vars,
                weights=b_weights,
                min_var=min_var,
                enable_splitting=enable_splitting,
                return_bounds=return_bounds,
                preserve_moments=preserve_moments,
            )

        if return_bounds:
            h_l_all[start:end], h_2_all[start:end], h_u_all[start:end] = res
        else:
            h_2_all[start:end] = res

    if return_bounds:
        return h_l_all, h_2_all, h_u_all
    return h_2_all

