"""Entropy calculation and approximation routines for DyRF-BO."""

from dyrf_bo.entropy.huber_entropy import (
    TABLE1_MEANS,
    TABLE1_STDS,
    TABLE1_STDS_MOMENT_MATCHED,
    TABLE1_WEIGHTS,
    huber_entropy_1d,
    huber_entropy_1d_numpy,
    huber_entropy_bounds_1d_numpy,
    huber_table1_split_1d,
)
from dyrf_bo.entropy.huber_entropy_gpu import (
    HAS_CUPY,
    HAS_GPU,
    clear_gpu_memory_pool,
    get_dynamic_huber_batch_size_gpu,
    huber_entropy_1d_cupy,
    huber_table1_split_1d_cupy,
)

__all__ = [
    "huber_entropy_1d",
    "huber_entropy_1d_numpy",
    "huber_entropy_bounds_1d_numpy",
    "huber_table1_split_1d",
    "huber_entropy_1d_cupy",
    "huber_table1_split_1d_cupy",
    "get_dynamic_huber_batch_size_gpu",
    "clear_gpu_memory_pool",
    "HAS_CUPY",
    "HAS_GPU",
    "TABLE1_WEIGHTS",
    "TABLE1_MEANS",
    "TABLE1_STDS",
    "TABLE1_STDS_MOMENT_MATCHED",
]
