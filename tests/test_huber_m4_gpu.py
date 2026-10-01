"""
Unit tests for Milestone 4: GPU Acceleration via CuPy & Dynamic Chunking.

Reference:
- Huber, Bailey, Durrant-Whyte, Hanebeck (IEEE MFI 2008):
  "On Entropy Approximation for Gaussian Mixture Random Vectors"
- Specification: detailed_huber_implementation_plan.md (Milestone 4)
"""

import os
import sys
import warnings
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from dyrf_bo.entropy import (
    huber_entropy_1d,
    huber_entropy_1d_numpy,
)
from dyrf_bo.entropy.huber_entropy_gpu import (
    HAS_CUPY,
    HAS_GPU,
    get_dynamic_huber_batch_size_gpu,
    huber_entropy_1d_cupy,
)


class MockCuPyModule:
    """NumPy-backed drop-in replacement for CuPy to test GPU kernels in non-CUDA environments."""

    ndarray = np.ndarray
    float64 = np.float64
    float32 = np.float32
    newaxis = np.newaxis
    pi = np.pi
    e = np.e
    inf = np.inf

    @staticmethod
    def asarray(a: Any, dtype: Any = None) -> np.ndarray:
        return np.asarray(a, dtype=dtype)

    @staticmethod
    def asnumpy(a: Any) -> np.ndarray:
        return np.asarray(a)

    @staticmethod
    def empty(shape: Any, dtype: Any = np.float64) -> np.ndarray:
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def full_like(a: Any, fill_value: Any, dtype: Any = None) -> np.ndarray:
        return np.full_like(a, fill_value, dtype=dtype)

    @staticmethod
    def empty_like(a: Any, dtype: Any = None) -> np.ndarray:
        return np.empty_like(a, dtype=dtype)

    @staticmethod
    def max(a: Any, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.max(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def isneginf(a: Any) -> np.ndarray:
        return np.isneginf(a)

    @staticmethod
    def where(cond: Any, x: Any, y: Any) -> np.ndarray:
        return np.where(cond, x, y)

    @staticmethod
    def exp(a: Any) -> np.ndarray:
        return np.exp(a)

    @staticmethod
    def sum(a: Any, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def maximum(x: Any, y: Any) -> np.ndarray:
        return np.maximum(x, y)

    @staticmethod
    def log(a: Any) -> np.ndarray:
        return np.log(a)

    @staticmethod
    def squeeze(a: Any, axis: Any = None) -> np.ndarray:
        return np.squeeze(a, axis=axis)

    @staticmethod
    def sqrt(a: Any) -> np.ndarray:
        return np.sqrt(a)

    @staticmethod
    def tile(a: Any, reps: Any) -> np.ndarray:
        return np.tile(a, reps)

    @staticmethod
    def get_default_memory_pool() -> MagicMock:
        return MagicMock()

    @staticmethod
    def get_default_pinned_memory_pool() -> MagicMock:
        return MagicMock()

    class cuda:
        class Device:
            def __init__(self, device_id: int = 0):
                self.device_id = device_id
                # 4 GB free out of 8 GB
                self.mem_info = (4 * 1024**3, 8 * 1024**3)

        class Stream:
            null = MagicMock()

        class memory:
            class OutOfMemoryError(Exception):
                pass

        @staticmethod
        def get_device_id() -> int:
            return 0


def test_unified_dispatcher_api():
    """
    Tests the unified huber_entropy_1d dispatcher API across input shapes,
    backend parameters, splitting options, and bounds flags.
    """
    rng = np.random.default_rng(42)
    B, T = 16, 5
    means = rng.uniform(-3.0, 3.0, size=(B, T))
    variances = rng.uniform(0.2, 2.0, size=(B, T))
    raw_w = rng.uniform(0.5, 1.5, size=(B, T))
    weights = raw_w / np.sum(raw_w, axis=-1, keepdims=True)

    # 1. Default arguments: backend="auto", batch_size="auto", return_bounds=False
    h2 = huber_entropy_1d(means, variances)
    assert isinstance(h2, np.ndarray)
    assert h2.shape == (B,)
    assert not np.isnan(h2).any()

    # 2. return_bounds=True: returns (H_l, H_2, H_u) tuple of 1D arrays
    res_bounds = huber_entropy_1d(means, variances, return_bounds=True)
    assert isinstance(res_bounds, tuple)
    assert len(res_bounds) == 3
    h_l, h_2_b, h_u = res_bounds
    assert h_l.shape == (B,)
    assert h_2_b.shape == (B,)
    assert h_u.shape == (B,)
    np.testing.assert_allclose(h_2_b, h2, atol=1e-14)
    assert np.all(h_l <= h_u + 1e-12)

    # 3. enable_splitting=True
    h2_split = huber_entropy_1d(means, variances, enable_splitting=True)
    assert isinstance(h2_split, np.ndarray)
    assert h2_split.shape == (B,)

    # 4. 1D input shape (T,) -> returns 1D array of shape (1,)
    means_1d = means[0]
    vars_1d = variances[0]
    h2_1d = huber_entropy_1d(means_1d, vars_1d)
    assert h2_1d.shape == (1,)
    np.testing.assert_allclose(h2_1d[0], h2[0], atol=1e-14)

    # 5. Empty batch (0, T) -> returns shape (0,)
    empty_means = np.empty((0, T), dtype=np.float64)
    empty_vars = np.empty((0, T), dtype=np.float64)
    empty_out = huber_entropy_1d(empty_means, empty_vars)
    assert empty_out.shape == (0,)

    empty_l, empty_2, empty_u = huber_entropy_1d(empty_means, empty_vars, return_bounds=True)
    assert empty_l.shape == (0,) and empty_2.shape == (0,) and empty_u.shape == (0,)

    # 6. Explicit backend="cpu"
    h2_cpu = huber_entropy_1d(means, variances, backend="cpu")
    np.testing.assert_allclose(h2_cpu, h2, atol=1e-14)

    # 7. Custom weights
    h2_weighted = huber_entropy_1d(means, variances, weights=weights)
    assert h2_weighted.shape == (B,)

    # 8. Invalid backend raises ValueError
    with pytest.raises(ValueError, match="backend must be one of: 'auto', 'cpu', 'gpu'"):
        huber_entropy_1d(means, variances, backend="tpu")

    # 9. Invalid batch_size raises ValueError
    with pytest.raises(ValueError, match="batch_size must be 'auto' or a positive integer"):
        huber_entropy_1d(means, variances, batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be 'auto' or a positive integer"):
        huber_entropy_1d(means, variances, batch_size=-10)
    with pytest.raises(ValueError, match="batch_size must be 'auto' or a positive integer"):
        huber_entropy_1d(means, variances, batch_size="fast")


def test_dynamic_chunking_stress():
    """
    Stress-tests dynamic batch chunking with B=50000, T=20 synthetic batch size
    and batch_size=500. Verifies chunked results match monolithic computation within 1e-12.
    """
    rng = np.random.default_rng(2026)
    B, T = 50000, 20
    chunk_size = 500

    means = rng.uniform(-4.0, 4.0, size=(B, T))
    variances = rng.uniform(0.1, 3.0, size=(B, T))

    # Compute with chunking
    h2_chunked = huber_entropy_1d(means, variances, backend="cpu", batch_size=chunk_size)
    assert h2_chunked.shape == (B,)

    # Verify against ground truth computed in monolithic slices
    # To check exact parity, verify on selected representative slices:
    sample_indices = [0, 1, 499, 500, 501, 24999, 25000, 49999]
    sample_means = means[sample_indices]
    sample_vars = variances[sample_indices]
    h2_ref_samples = huber_entropy_1d_numpy(sample_means, sample_vars)

    np.testing.assert_allclose(
        h2_chunked[sample_indices],
        h2_ref_samples,
        atol=1e-12,
        err_msg="Chunked calculation diverged from direct numpy calculation on boundary slices",
    )

    # Verify return_bounds=True with chunking
    h_l, h_2_b, h_u = huber_entropy_1d(
        means[:2000], variances[:2000], backend="cpu", batch_size=300, return_bounds=True
    )
    assert h_l.shape == (2000,)
    assert h_2_b.shape == (2000,)
    assert h_u.shape == (2000,)
    np.testing.assert_allclose(h_2_b, h2_chunked[:2000], atol=1e-12)
    assert np.all(h_l <= h_u + 1e-12)

    # Verify enable_splitting=True with chunking
    h2_split_chunked = huber_entropy_1d(
        means[:1500], variances[:1500], backend="cpu", batch_size=400, enable_splitting=True
    )
    h2_split_monolithic = huber_entropy_1d_numpy(
        means[:1500], variances[:1500], enable_splitting=True
    )
    np.testing.assert_allclose(h2_split_chunked, h2_split_monolithic, atol=1e-12)


def test_gpu_fallback_when_no_cuda():
    """
    Forces backend="gpu" when CuPy/CUDA is unavailable or mocked as unavailable.
    Verifies that it logs/warns with a warning and cleanly falls back to CPU
    without crashing, producing results matching CPU execution.
    """
    rng = np.random.default_rng(999)
    means = rng.uniform(-2.0, 2.0, size=(10, 4))
    variances = rng.uniform(0.5, 2.0, size=(10, 4))

    expected_cpu = huber_entropy_1d(means, variances, backend="cpu")

    # If real GPU is not available, calling backend="gpu" should warn and match CPU
    if not HAS_GPU:
        with pytest.warns(RuntimeWarning, match="GPU backend requested but CUDA/CuPy is unavailable"):
            out_fallback = huber_entropy_1d(means, variances, backend="gpu")
        np.testing.assert_allclose(out_fallback, expected_cpu, atol=1e-14)
    else:
        # If GPU is available, mock HAS_GPU=False to test the fallback branch
        with patch("dyrf_bo.entropy.huber_entropy.HAS_GPU", False):
            with pytest.warns(RuntimeWarning, match="GPU backend requested but CUDA/CuPy is unavailable"):
                out_fallback = huber_entropy_1d(means, variances, backend="gpu")
            np.testing.assert_allclose(out_fallback, expected_cpu, atol=1e-14)


def test_cpu_gpu_numerical_parity():
    """
    Asserts numerical parity between CPU (numpy) and GPU (cupy / mocked cupy) kernels:
    - Pure leaf Option A: |H_2_gpu - H_2_cpu| < 1e-6 float32 / 1e-12 float64
    - Split Option B: |H_2_gpu_split - H_2_cpu_split| < 1e-12 float64
    - Bounds: |H_l_gpu - H_l_cpu| < 1e-12, |H_u_gpu - H_u_cpu| < 1e-12 float64
    """
    rng = np.random.default_rng(777)
    B, T = 100, 20
    means = rng.uniform(-5.0, 5.0, size=(B, T))
    variances = rng.uniform(0.1, 5.0, size=(B, T))

    # Reference CPU outputs
    h2_cpu = huber_entropy_1d_numpy(means, variances)
    h_l_cpu, _, h_u_cpu = huber_entropy_1d_numpy(means, variances, return_bounds=True)
    h2_split_cpu = huber_entropy_1d_numpy(means, variances, enable_splitting=True)

    if HAS_GPU:
        # Real CuPy execution
        import cupy as cp

        means_cp = cp.asarray(means)
        vars_cp = cp.asarray(variances)

        h2_gpu = cp.asnumpy(huber_entropy_1d_cupy(means_cp, vars_cp))
        np.testing.assert_allclose(h2_gpu, h2_cpu, atol=1e-12)

        h_l_gpu, h2_b_gpu, h_u_gpu = huber_entropy_1d_cupy(means_cp, vars_cp, return_bounds=True)
        np.testing.assert_allclose(cp.asnumpy(h_l_gpu), h_l_cpu, atol=1e-12)
        np.testing.assert_allclose(cp.asnumpy(h_u_gpu), h_u_cpu, atol=1e-12)

        h2_split_gpu = cp.asnumpy(
            huber_entropy_1d_cupy(means_cp, vars_cp, enable_splitting=True)
        )
        np.testing.assert_allclose(h2_split_gpu, h2_split_cpu, atol=1e-12)

        # Float32 test
        means_f32 = means.astype(np.float32)
        vars_f32 = variances.astype(np.float32)
        h2_cpu_f32 = huber_entropy_1d_numpy(means_f32, vars_f32)
        h2_gpu_f32 = cp.asnumpy(huber_entropy_1d_cupy(cp.asarray(means_f32), cp.asarray(vars_f32)))
        np.testing.assert_allclose(h2_gpu_f32, h2_cpu_f32, atol=1e-6)
    else:
        # Execute using MockCuPyModule to verify GPU kernel logic directly
        with patch("dyrf_bo.entropy.huber_entropy_gpu.cp", MockCuPyModule):
            # Option A
            h2_mock_gpu = huber_entropy_1d_cupy(means, variances)
            np.testing.assert_allclose(h2_mock_gpu, h2_cpu, atol=1e-12)

            # Bounds
            h_l_mock, h2_b_mock, h_u_mock = huber_entropy_1d_cupy(
                means, variances, return_bounds=True
            )
            np.testing.assert_allclose(h_l_mock, h_l_cpu, atol=1e-12)
            np.testing.assert_allclose(h2_b_mock, h2_cpu, atol=1e-12)
            np.testing.assert_allclose(h_u_mock, h_u_cpu, atol=1e-12)

            # Option B (Split)
            h2_split_mock = huber_entropy_1d_cupy(means, variances, enable_splitting=True)
            np.testing.assert_allclose(h2_split_mock, h2_split_cpu, atol=1e-12)

            # Float32 parity
            means_f32 = means.astype(np.float32)
            vars_f32 = variances.astype(np.float32)
            h2_cpu_f32 = huber_entropy_1d_numpy(means_f32, vars_f32)
            h2_mock_f32 = huber_entropy_1d_cupy(means_f32, vars_f32)
            np.testing.assert_allclose(h2_mock_f32, h2_cpu_f32, atol=1e-6)


def test_dynamic_vram_resolution():
    """
    Tests dynamic VRAM resolution logic for GPU batch sizing.
    Verifies that batch size respects memory constraints and clipping range [100, 25000].
    """
    # 1. 4 GB free memory, T=50, float64
    batch_size_opt_a = get_dynamic_huber_batch_size_gpu(
        n_components=50,
        enable_splitting=False,
        free_vram_bytes=4 * 1024**3,
    )
    assert 100 <= batch_size_opt_a <= 25000

    # 2. Splitting requires 4x more memory per sample, so batch size should be smaller
    batch_size_opt_b = get_dynamic_huber_batch_size_gpu(
        n_components=50,
        enable_splitting=True,
        free_vram_bytes=4 * 1024**3,
    )
    assert 100 <= batch_size_opt_b <= 25000
    assert batch_size_opt_b < batch_size_opt_a

    # 3. Extremely large VRAM (e.g. 80 GB) -> clipped to 25000
    batch_size_huge = get_dynamic_huber_batch_size_gpu(
        n_components=10,
        enable_splitting=False,
        free_vram_bytes=80 * 1024**3,
    )
    assert batch_size_huge == 25000

    # 4. Tiny VRAM (e.g. 1 MB) -> clipped to lower bound 100
    batch_size_tiny = get_dynamic_huber_batch_size_gpu(
        n_components=100,
        enable_splitting=True,
        free_vram_bytes=1 * 1024**2,
    )
    assert batch_size_tiny == 100


def test_gpu_memory_halving_on_oom():
    """
    Genuine OOM simulation:
    Mocks huber_entropy_1d_cupy to raise MemoryError on first call with full chunk,
    then succeed on halved chunks. Verifies clear_gpu_memory_pool is called and
    halved execution returns correct results matching CPU reference.
    """
    rng = np.random.default_rng(101)
    B, T = 20, 5
    means = rng.uniform(-2.0, 2.0, size=(B, T))
    variances = rng.uniform(0.5, 2.0, size=(B, T))

    cpu_ref = huber_entropy_1d_numpy(means, variances)
    cpu_bounds = huber_entropy_1d_numpy(means, variances, return_bounds=True)

    from dyrf_bo.entropy.huber_entropy import _execute_chunk_gpu
    import dyrf_bo.entropy.huber_entropy as he_mod
    import dyrf_bo.entropy.huber_entropy_gpu as heg_mod

    real_cupy_fn = huber_entropy_1d_cupy

    def mock_cupy_fn(*args, **kwargs):
        input_means = args[0]
        if input_means.shape[0] >= 20:
            raise MemoryError("Simulated GPU Out of Memory on full chunk")
        return real_cupy_fn(*args, **kwargs)

    mock_clear_pool = MagicMock()

    with patch.object(he_mod, "HAS_GPU", True), \
         patch.object(heg_mod, "HAS_GPU", True), \
         patch.object(heg_mod, "cp", MockCuPyModule), \
         patch.object(he_mod, "clear_gpu_memory_pool", mock_clear_pool), \
         patch.object(he_mod, "huber_entropy_1d_cupy", side_effect=mock_cupy_fn):

        # 1. Test return_bounds=False
        res_halved = _execute_chunk_gpu(means, variances, np.full_like(means, 1.0 / T))
        assert mock_clear_pool.call_count >= 1
        np.testing.assert_allclose(res_halved, cpu_ref, atol=1e-12)

        # 2. Test return_bounds=True
        mock_clear_pool.reset_mock()
        res_bounds_halved = _execute_chunk_gpu(
            means, variances, np.full_like(means, 1.0 / T), return_bounds=True
        )
        assert mock_clear_pool.call_count >= 1
        np.testing.assert_allclose(res_bounds_halved[0], cpu_bounds[0], atol=1e-12)
        np.testing.assert_allclose(res_bounds_halved[1], cpu_bounds[1], atol=1e-12)
        np.testing.assert_allclose(res_bounds_halved[2], cpu_bounds[2], atol=1e-12)


def test_persistent_oom_fallback_to_cpu():
    """
    Simulates persistent GPU OOM where memory errors persist across all batch halvings
    down to sub_chunk_size < 1. Verifies that RuntimeWarning is emitted and execution
    falls back cleanly to CPU, returning valid mathematical results.
    """
    rng = np.random.default_rng(202)
    B, T = 16, 4
    means = rng.uniform(-2.0, 2.0, size=(B, T))
    variances = rng.uniform(0.5, 2.0, size=(B, T))

    cpu_ref = huber_entropy_1d_numpy(means, variances)
    cpu_bounds = huber_entropy_1d_numpy(means, variances, return_bounds=True)

    import dyrf_bo.entropy.huber_entropy as he_mod
    import dyrf_bo.entropy.huber_entropy_gpu as heg_mod
    from dyrf_bo.entropy.huber_entropy import _execute_chunk_gpu

    def persistent_oom(*args, **kwargs):
        raise MemoryError("Persistent GPU OOM")

    mock_clear_pool = MagicMock()

    with patch.object(he_mod, "HAS_GPU", True), \
         patch.object(heg_mod, "HAS_GPU", True), \
         patch.object(heg_mod, "cp", MockCuPyModule), \
         patch.object(he_mod, "clear_gpu_memory_pool", mock_clear_pool), \
         patch.object(he_mod, "huber_entropy_1d_cupy", side_effect=persistent_oom):

        # 1. return_bounds=False
        with pytest.warns(RuntimeWarning, match="GPU out of memory during Huber entropy computation. Falling back to CPU"):
            out = _execute_chunk_gpu(means, variances, np.full_like(means, 1.0 / T))
        np.testing.assert_allclose(out, cpu_ref, atol=1e-12)
        assert mock_clear_pool.call_count >= 1

        # 2. return_bounds=True
        mock_clear_pool.reset_mock()
        with pytest.warns(RuntimeWarning, match="GPU out of memory during Huber entropy computation. Falling back to CPU"):
            out_bounds = _execute_chunk_gpu(
                means, variances, np.full_like(means, 1.0 / T), return_bounds=True
            )
        np.testing.assert_allclose(out_bounds[0], cpu_bounds[0], atol=1e-12)
        np.testing.assert_allclose(out_bounds[1], cpu_bounds[1], atol=1e-12)
        np.testing.assert_allclose(out_bounds[2], cpu_bounds[2], atol=1e-12)


@pytest.mark.parametrize("B", [1, 2, 5, 9])
def test_small_batch_sizes_gpu(B: int):
    """
    Verifies that small batch sizes B in {1, 2, 5, 9} execute cleanly on GPU
    without crashing or returning None, matching CPU reference outputs exactly.
    """
    rng = np.random.default_rng(303 + B)
    T = 4
    means = rng.uniform(-2.0, 2.0, size=(B, T))
    variances = rng.uniform(0.5, 2.0, size=(B, T))

    cpu_ref = huber_entropy_1d(means, variances, backend="cpu")
    cpu_bounds = huber_entropy_1d(means, variances, backend="cpu", return_bounds=True)
    cpu_split = huber_entropy_1d(means, variances, backend="cpu", enable_splitting=True)

    import dyrf_bo.entropy.huber_entropy as he_mod
    import dyrf_bo.entropy.huber_entropy_gpu as heg_mod

    with patch.object(he_mod, "HAS_GPU", True), \
         patch.object(heg_mod, "HAS_GPU", True), \
         patch.object(heg_mod, "cp", MockCuPyModule):

        # Standard H2
        gpu_res = huber_entropy_1d(means, variances, backend="gpu")
        assert gpu_res is not None
        assert isinstance(gpu_res, np.ndarray)
        assert gpu_res.shape == (B,)
        np.testing.assert_allclose(gpu_res, cpu_ref, atol=1e-12)

        # Bounds (H_l, H_2, H_u)
        gpu_bounds = huber_entropy_1d(means, variances, backend="gpu", return_bounds=True)
        assert isinstance(gpu_bounds, tuple)
        assert len(gpu_bounds) == 3
        for arr in gpu_bounds:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (B,)
        np.testing.assert_allclose(gpu_bounds[0], cpu_bounds[0], atol=1e-12)
        np.testing.assert_allclose(gpu_bounds[1], cpu_bounds[1], atol=1e-12)
        np.testing.assert_allclose(gpu_bounds[2], cpu_bounds[2], atol=1e-12)

        # Splitting
        gpu_split = huber_entropy_1d(means, variances, backend="gpu", enable_splitting=True)
        assert isinstance(gpu_split, np.ndarray)
        assert gpu_split.shape == (B,)
        np.testing.assert_allclose(gpu_split, cpu_split, atol=1e-12)


def test_remainder_batch_gpu():
    """
    Tests remainder batch handling on GPU (B=505 with batch_size=500), where the last
    sub-chunk has length 5 (< 10). Verifies clean execution and exact match with CPU.
    """
    rng = np.random.default_rng(404)
    B, T = 505, 4
    means = rng.uniform(-3.0, 3.0, size=(B, T))
    variances = rng.uniform(0.2, 2.0, size=(B, T))

    cpu_ref = huber_entropy_1d(means, variances, backend="cpu", batch_size=500)
    cpu_bounds = huber_entropy_1d(means, variances, backend="cpu", batch_size=500, return_bounds=True)
    cpu_split = huber_entropy_1d(means, variances, backend="cpu", batch_size=500, enable_splitting=True)

    import dyrf_bo.entropy.huber_entropy as he_mod
    import dyrf_bo.entropy.huber_entropy_gpu as heg_mod

    with patch.object(he_mod, "HAS_GPU", True), \
         patch.object(heg_mod, "HAS_GPU", True), \
         patch.object(heg_mod, "cp", MockCuPyModule):

        # Standard H2
        gpu_res = huber_entropy_1d(means, variances, backend="gpu", batch_size=500)
        assert gpu_res is not None
        assert isinstance(gpu_res, np.ndarray)
        assert gpu_res.shape == (505,)
        np.testing.assert_allclose(gpu_res, cpu_ref, atol=1e-12)

        # Bounds
        gpu_bounds = huber_entropy_1d(means, variances, backend="gpu", batch_size=500, return_bounds=True)
        assert isinstance(gpu_bounds, tuple)
        assert len(gpu_bounds) == 3
        for arr in gpu_bounds:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (505,)
        np.testing.assert_allclose(gpu_bounds[0], cpu_bounds[0], atol=1e-12)
        np.testing.assert_allclose(gpu_bounds[1], cpu_bounds[1], atol=1e-12)
        np.testing.assert_allclose(gpu_bounds[2], cpu_bounds[2], atol=1e-12)

        # Splitting
        gpu_split = huber_entropy_1d(means, variances, backend="gpu", batch_size=500, enable_splitting=True)
        assert isinstance(gpu_split, np.ndarray)
        assert gpu_split.shape == (505,)
        np.testing.assert_allclose(gpu_split, cpu_split, atol=1e-12)


def test_weight_broadcasting_gpu():
    """
    Tests weight broadcasting in huber_table1_split_1d_cupy and huber_entropy_1d_cupy
    when weights have shape (T,) or (1, T) and batch size B > 1.
    """
    from dyrf_bo.entropy.huber_entropy_gpu import (
        huber_entropy_1d_cupy,
        huber_table1_split_1d_cupy,
    )
    import dyrf_bo.entropy.huber_entropy_gpu as heg_mod

    rng = np.random.default_rng(505)
    B, T = 8, 3
    means = rng.uniform(-2.0, 2.0, size=(B, T))
    variances = rng.uniform(0.5, 2.0, size=(B, T))
    weights_1d = np.array([0.2, 0.5, 0.3])
    weights_2d = weights_1d[np.newaxis, :]  # (1, T)
    weights_tiled = np.tile(weights_1d, (B, 1))  # (B, T)

    with patch.object(heg_mod, "cp", MockCuPyModule):
        # 1. Test huber_table1_split_1d_cupy with (1, T) and (T,)
        m_sp_tiled, v_sp_tiled, w_sp_tiled = huber_table1_split_1d_cupy(
            means, variances, weights_cp=weights_tiled
        )
        m_sp_1d, v_sp_1d, w_sp_1d = huber_table1_split_1d_cupy(
            means, variances, weights_cp=weights_1d
        )
        m_sp_2d, v_sp_2d, w_sp_2d = huber_table1_split_1d_cupy(
            means, variances, weights_cp=weights_2d
        )

        assert m_sp_1d.shape == (B, 4 * T)
        assert w_sp_1d.shape == (B, 4 * T)
        np.testing.assert_allclose(m_sp_1d, m_sp_tiled, atol=1e-12)
        np.testing.assert_allclose(w_sp_1d, w_sp_tiled, atol=1e-12)
        np.testing.assert_allclose(m_sp_2d, m_sp_tiled, atol=1e-12)
        np.testing.assert_allclose(w_sp_2d, w_sp_tiled, atol=1e-12)

        # 2. Test huber_entropy_1d_cupy with (1, T) and (T,)
        h2_tiled = huber_entropy_1d_cupy(means, variances, weights_cp=weights_tiled, enable_splitting=True)
        h2_1d = huber_entropy_1d_cupy(means, variances, weights_cp=weights_1d, enable_splitting=True)
        h2_2d = huber_entropy_1d_cupy(means, variances, weights_cp=weights_2d, enable_splitting=True)

        assert h2_1d.shape == (B,)
        np.testing.assert_allclose(h2_1d, h2_tiled, atol=1e-12)
        np.testing.assert_allclose(h2_2d, h2_tiled, atol=1e-12)


def test_cupy_unavailable_raises_runtime_error():
    """
    Verifies that calling huber_entropy_1d_cupy or huber_table1_split_1d_cupy
    when cp is None raises a RuntimeError.
    """
    from dyrf_bo.entropy.huber_entropy_gpu import (
        huber_entropy_1d_cupy,
        huber_table1_split_1d_cupy,
    )
    import dyrf_bo.entropy.huber_entropy_gpu as heg_mod

    means = np.zeros((2, 3))
    variances = np.ones((2, 3))

    with patch.object(heg_mod, "cp", None):
        with pytest.raises(RuntimeError, match="CuPy / GPU acceleration is not available"):
            huber_entropy_1d_cupy(means, variances)

        with pytest.raises(RuntimeError, match="CuPy / GPU acceleration is not available"):
            huber_table1_split_1d_cupy(means, variances)
