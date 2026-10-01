"""
GPU-accelerated Huber (2008) GMM differential entropy approximation routines via CuPy.

Reference:
- Huber, Bailey, Durrant-Whyte, Hanebeck (IEEE MFI 2008):
  "On Entropy Approximation for Gaussian Mixture Random Vectors"
- Specification: detailed_huber_implementation_plan.md (Milestone 4)
"""

import warnings
from typing import Any, Optional, Tuple, Union

import numpy as np

# CuPy & CUDA availability detection conforming to DyRF-BO architecture
try:
    import cupy as cp
    import cupyx

    HAS_CUPY = True
    try:
        if cp.cuda.runtime.getDeviceCount() > 0:
            smoke_test = cp.asarray([1.0])
            smoke_test = smoke_test + 1.0
            cp.cuda.Stream.null.synchronize()
            HAS_GPU = bool(cp.asnumpy(smoke_test)[0] == 2.0)
        else:
            HAS_GPU = False
    except Exception:
        HAS_GPU = False
except ImportError:
    cp = None
    cupyx = None
    HAS_CUPY = False
    HAS_GPU = False

# Import Table I constants from CPU module
from dyrf_bo.entropy.huber_entropy import (
    _MOMENT_MATCHED_STDS,
    _NORM_TABLE1_WEIGHTS,
    _RAW_TABLE1_MEANS,
    _RAW_TABLE1_STDS,
    TABLE1_MEANS,
    TABLE1_STDS,
    TABLE1_STDS_MOMENT_MATCHED,
    TABLE1_WEIGHTS,
)

_LN2 = 0.693147180559945309417232121458


def clear_gpu_memory_pool() -> None:
    """Frees all unused memory blocks in the CuPy default memory pool."""
    if cp is not None:
        try:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            if hasattr(cp, "get_default_pinned_memory_pool"):
                pinned_mempool = cp.get_default_pinned_memory_pool()
                pinned_mempool.free_all_blocks()
        except Exception:
            pass


def _fast_logsumexp_cupy(a: Any, axis: int = -1) -> Any:
    """
    Numerically stable vectorized LogSumExp along the specified axis for CuPy arrays.
    Guaranteed zero warnings for all-negative-infinity slices.
    """
    xp = cp if cp is not None else np
    a_max = xp.max(a, axis=axis, keepdims=True)
    is_neg_inf = xp.isneginf(a_max)
    safe_a_max = xp.where(is_neg_inf, 0.0, a_max)
    tmp = xp.exp(a - safe_a_max)
    sum_tmp = xp.sum(tmp, axis=axis, keepdims=True)
    safe_sum = xp.maximum(sum_tmp, 1e-300)
    out = xp.where(is_neg_inf, -xp.inf, xp.log(safe_sum) + safe_a_max)
    return xp.squeeze(out, axis=axis)


def _compute_h2_cupy(
    means_cp: Any,
    vars_clipped_cp: Any,
    w_norm_cp: Any,
    delta_cp: Optional[Any] = None,
) -> Any:
    """
    Computes closed-form second-order Huber differential entropy H_2 in bits on GPU.
    Option A: Pairwise T x T tensor evaluation.
    """
    xp = cp if cp is not None else np

    if delta_cp is None:
        delta_cp = means_cp[:, :, xp.newaxis] - means_cp[:, xp.newaxis, :]  # (B, T, T)

    inv_2var = 0.5 / vars_clipped_cp[:, xp.newaxis, :]  # (B, 1, T)
    half_log_2pi_var = 0.5 * xp.log(2.0 * xp.pi * vars_clipped_cp[:, xp.newaxis, :])
    log_joint = (
        xp.log(w_norm_cp[:, xp.newaxis, :])
        - half_log_2pi_var
        - (delta_cp ** 2) * inv_2var
    )  # (B, T, T)

    log_f = _fast_logsumexp_cupy(log_joint, axis=-1)  # (B, T)
    log_w = log_joint - log_f[:, :, xp.newaxis]
    w_ij = xp.exp(log_w)  # (B, T, T)

    inv_var = 1.0 / vars_clipped_cp[:, xp.newaxis, :]
    inv_var2 = inv_var ** 2

    grad_ratio = -xp.sum(w_ij * delta_cp * inv_var, axis=-1)  # (B, T)
    second_ratio = xp.sum(w_ij * ((delta_cp ** 2) * inv_var2 - inv_var), axis=-1)  # (B, T)
    F = second_ratio - (grad_ratio ** 2)  # (B, T)

    h0_nats = -xp.sum(w_norm_cp * log_f, axis=-1)  # (B,)
    h2_nats = h0_nats - 0.5 * xp.sum(w_norm_cp * F * vars_clipped_cp, axis=-1)  # (B,)

    return h2_nats / _LN2


def _compute_bounds_cupy(
    means_cp: Any,
    vars_clipped_cp: Any,
    w_norm_cp: Any,
    delta_cp: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """
    Computes closed-form information-theoretic lower and upper bounds (H_l, H_u) in bits on GPU.
    """
    xp = cp if cp is not None else np

    # Lower Bound H_l (Theorem 2)
    sum_vars = (
        vars_clipped_cp[:, :, xp.newaxis] + vars_clipped_cp[:, xp.newaxis, :]
    )  # (B, T, T)
    if delta_cp is None:
        delta_cp = means_cp[:, :, xp.newaxis] - means_cp[:, xp.newaxis, :]

    inv_2S = 0.5 / sum_vars
    half_log_2pi_S = 0.5 * xp.log(2.0 * xp.pi * sum_vars)
    log_z = -half_log_2pi_S - (delta_cp ** 2) * inv_2S
    log_omega_z = xp.log(w_norm_cp[:, xp.newaxis, :]) + log_z
    log_sum_omega_z = _fast_logsumexp_cupy(log_omega_z, axis=-1)  # (B, T)
    h_l_nats = -xp.sum(w_norm_cp * log_sum_omega_z, axis=-1)  # (B,)
    h_l_bits = h_l_nats / _LN2

    # Upper Bound H_u (Theorem 3)
    safe_w = xp.maximum(w_norm_cp, 1e-300)
    h_omega_bits = -xp.sum(w_norm_cp * (xp.log(safe_w) / _LN2), axis=-1)  # (B,)
    h_aleatoric_bits = xp.sum(
        w_norm_cp * 0.5 * (xp.log(2.0 * xp.pi * xp.e * vars_clipped_cp) / _LN2),
        axis=-1,
    )  # (B,)
    h_u_bits = h_omega_bits + h_aleatoric_bits

    return h_l_bits, h_u_bits


def huber_table1_split_1d_cupy(
    means_cp: Any,
    vars_clipped_cp: Any,
    weights_cp: Optional[Any] = None,
    min_var: float = 1e-6,
    preserve_moments: bool = True,
) -> Tuple[Any, Any, Any]:
    """
    Splits 1D GMM components using Huber Table I library on GPU.
    Returns:
        means_split: (B, 4T)
        vars_split: (B, 4T)
        weights_split: (B, 4T)
    """
    if cp is None:
        raise RuntimeError("CuPy / GPU acceleration is not available in the current environment.")
    xp = cp

    if means_cp.ndim == 1:
        means_cp = means_cp[xp.newaxis, :]
        vars_clipped_cp = vars_clipped_cp[xp.newaxis, :]
        if weights_cp is not None and weights_cp.ndim == 1:
            weights_cp = weights_cp[xp.newaxis, :]

    B, T = means_cp.shape
    if B == 0:
        empty = xp.empty((0, 4 * T), dtype=means_cp.dtype)
        return empty, empty.copy(), empty.copy()

    vars_safe = xp.maximum(vars_clipped_cp, min_var)
    if vars_safe.shape[0] == 1 and B > 1:
        vars_safe = xp.tile(vars_safe, (B, 1))

    if weights_cp is None:
        w_norm = xp.full_like(means_cp, 1.0 / T)
    else:
        if weights_cp.ndim == 1:
            weights_cp = weights_cp[xp.newaxis, :]
        w_sum = xp.sum(weights_cp, axis=-1, keepdims=True)
        w_norm = weights_cp / xp.maximum(w_sum, 1e-300)

    if w_norm.shape[0] == 1 and B > 1:
        w_norm = xp.tile(w_norm, (B, 1))

    dtype = means_cp.dtype
    w_lib = xp.asarray(_NORM_TABLE1_WEIGHTS, dtype=dtype)
    mu_lib = xp.asarray(_RAW_TABLE1_MEANS, dtype=dtype)
    std_lib = xp.asarray(
        _MOMENT_MATCHED_STDS if preserve_moments else _RAW_TABLE1_STDS,
        dtype=dtype,
    )

    std_orig = xp.sqrt(vars_safe)  # (B, T)
    means_expanded = (
        means_cp[:, :, xp.newaxis]
        + std_orig[:, :, xp.newaxis] * mu_lib[xp.newaxis, xp.newaxis, :]
    )  # (B, T, 4)
    vars_expanded = (
        vars_safe[:, :, xp.newaxis] * (std_lib ** 2)[xp.newaxis, xp.newaxis, :]
    )  # (B, T, 4)
    weights_expanded = (
        w_norm[:, :, xp.newaxis] * w_lib[xp.newaxis, xp.newaxis, :]
    )  # (B, T, 4)

    means_split = means_expanded.reshape(B, 4 * T)
    vars_split = vars_expanded.reshape(B, 4 * T)
    weights_split = weights_expanded.reshape(B, 4 * T)

    return means_split, vars_split, weights_split


def _compute_h2_split_cupy(
    means_orig: Any,
    vars_orig: Any,
    w_orig: Any,
    means_sp: Any,
    vars_sp: Any,
    w_sp: Any,
) -> Any:
    """
    Computes second-order Huber entropy H_2 in bits for split components on GPU.
    Strictly enforces Huber Remark 1: outer expectation over 4T split points,
    inner density g(y) over the original T components.
    """
    xp = cp if cp is not None else np

    # Asymmetric cross-evaluation tensor (B, 4T, T)
    delta = means_sp[:, :, xp.newaxis] - means_orig[:, xp.newaxis, :]
    inv_2var = 0.5 / vars_orig[:, xp.newaxis, :]
    half_log_2pi_var = 0.5 * xp.log(2.0 * xp.pi * vars_orig[:, xp.newaxis, :])
    log_joint = (
        xp.log(w_orig[:, xp.newaxis, :])
        - half_log_2pi_var
        - (delta ** 2) * inv_2var
    )  # (B, 4T, T)

    log_g = _fast_logsumexp_cupy(log_joint, axis=-1)  # (B, 4T)
    log_w = log_joint - log_g[:, :, xp.newaxis]
    w_mj = xp.exp(log_w)  # (B, 4T, T)

    inv_var = 1.0 / vars_orig[:, xp.newaxis, :]
    inv_var2 = inv_var ** 2

    grad_ratio = -xp.sum(w_mj * delta * inv_var, axis=-1)  # (B, 4T)
    second_ratio = xp.sum(w_mj * ((delta ** 2) * inv_var2 - inv_var), axis=-1)  # (B, 4T)
    F = second_ratio - (grad_ratio ** 2)  # (B, 4T)

    h0_nats = -xp.sum(w_sp * log_g, axis=-1)  # (B,)
    h2_nats = h0_nats - 0.5 * xp.sum(w_sp * F * vars_sp, axis=-1)  # (B,)

    return h2_nats / _LN2


def huber_entropy_1d_cupy(
    means_cp: Any,
    variances_cp: Any,
    weights_cp: Optional[Any] = None,
    min_var: float = 1e-6,
    enable_splitting: bool = False,
    return_bounds: bool = False,
    preserve_moments: bool = True,
) -> Union[Any, Tuple[Any, Any, Any]]:
    """
    Computes closed-form second-order Huber (2008) GMM differential entropy H_2
    (and optional bounds H_l, H_u) in bits on GPU via CuPy.

    Parameters
    ----------
    means_cp : cp.ndarray
        Component means of shape (B, T) or (T,).
    variances_cp : cp.ndarray
        Component variances of shape (B, T) or (T,).
    weights_cp : Optional[cp.ndarray], default=None
        Component mixture weights summing to 1 across T.
    min_var : float, default=1e-6
        Numerical floor for component variances.
    enable_splitting : bool, default=False
        If True, splits components via Table I library (Huber Remark 1).
    return_bounds : bool, default=False
        If True, returns (H_l, H_2, H_u) bounding triplet.
    preserve_moments : bool, default=True
        If True, enforces exact moment conservation in splitting.

    Returns
    -------
    result : Union[cp.ndarray, Tuple[cp.ndarray, cp.ndarray, cp.ndarray]]
        H_2 array of shape (B,) or bounding triplet (H_l, H_2, H_u).
    """
    if cp is None:
        raise RuntimeError("CuPy / GPU acceleration is not available in the current environment.")
    xp = cp

    is_1d = means_cp.ndim == 1
    if is_1d:
        means_cp = means_cp[xp.newaxis, :]
        variances_cp = variances_cp[xp.newaxis, :]
        if weights_cp is not None and weights_cp.ndim == 1:
            weights_cp = weights_cp[xp.newaxis, :]

    B, T = means_cp.shape
    if B == 0:
        empty = xp.empty((0,), dtype=means_cp.dtype)
        if return_bounds:
            return empty, empty.copy(), empty.copy()
        return empty

    vars_clipped = xp.maximum(variances_cp, min_var)
    if vars_clipped.shape[0] == 1 and B > 1:
        vars_clipped = xp.tile(vars_clipped, (B, 1))

    if weights_cp is None:
        w_norm = xp.full_like(means_cp, 1.0 / T)
    else:
        if weights_cp.ndim == 1:
            weights_cp = weights_cp[xp.newaxis, :]
        w_sum = xp.sum(weights_cp, axis=-1, keepdims=True)
        w_norm = weights_cp / xp.maximum(w_sum, 1e-300)

    if w_norm.shape[0] == 1 and B > 1:
        w_norm = xp.tile(w_norm, (B, 1))

    if enable_splitting:
        m_sp, v_sp, w_sp = huber_table1_split_1d_cupy(
            means_cp,
            vars_clipped,
            weights_cp=w_norm,
            min_var=min_var,
            preserve_moments=preserve_moments,
        )
        h2_bits = _compute_h2_split_cupy(
            means_cp, vars_clipped, w_norm, m_sp, v_sp, w_sp
        )
    else:
        delta = means_cp[:, :, xp.newaxis] - means_cp[:, xp.newaxis, :]
        h2_bits = _compute_h2_cupy(means_cp, vars_clipped, w_norm, delta_cp=delta)

    if return_bounds:
        delta_orig = means_cp[:, :, xp.newaxis] - means_cp[:, xp.newaxis, :]
        h_l_bits, h_u_bits = _compute_bounds_cupy(
            means_cp, vars_clipped, w_norm, delta_cp=delta_orig
        )
        return h_l_bits, h2_bits, h_u_bits

    return h2_bits


def get_dynamic_huber_batch_size_gpu(
    n_components: int,
    enable_splitting: bool = False,
    dtype: Any = np.float64,
    free_vram_bytes: Optional[int] = None,
    device_id: int = 0,
) -> int:
    """
    Computes a memory-safe batch size for GPU execution based on available VRAM.

    Formula (Specification Section 4.2):
        batch_size = clip(floor((free_vram * 0.35) / (bytes_per_sample * 4)), 100, 25000)
    where:
        bytes_per_sample = (4 * T * T * itemsize) if enable_splitting else (T * T * itemsize)

    Parameters
    ----------
    n_components : int
        Number of mixture components T.
    enable_splitting : bool, default=False
        Whether Huber Table I splitting is active.
    dtype : Any, default=np.float64
        Data type of the tensors.
    free_vram_bytes : Optional[int], default=None
        Free VRAM in bytes. If None, queries the GPU device info.
    device_id : int, default=0
        GPU device ID.

    Returns
    -------
    batch_size : int
        Recommended batch size in range [100, 25000].
    """
    itemsize = np.dtype(dtype).itemsize
    T = max(int(n_components), 1)

    if free_vram_bytes is None:
        if HAS_GPU and cp is not None:
            try:
                free_mem, _ = cp.cuda.Device(device_id).mem_info
                free_vram_bytes = free_mem
            except Exception:
                free_vram_bytes = 2 * 1024**3
        else:
            free_vram_bytes = 2 * 1024**3

    bytes_per_sample = (4 * T * T * itemsize) if enable_splitting else (T * T * itemsize)
    total_bytes_per_sample = max(bytes_per_sample * 4, 1)
    target_vram = free_vram_bytes * 0.35
    batch_size = int(target_vram // total_bytes_per_sample)
    return int(np.clip(batch_size, 100, 25000))
