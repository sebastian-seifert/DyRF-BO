import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import List
import pytest

from scripts.carps_bbob_full_registry import (
    CarpsBBOBFullRegistry,
    BBOBTaskInfo,
    BBOBFunctionCategory,
)


def test_registry_total_task_count():
    """Verify exactly 360 tasks are generated across 5 dimensions, 24 functions, 3 instances."""
    tasks = CarpsBBOBFullRegistry.get_all_tasks()
    assert len(tasks) == 360
    assert len(set(tasks)) == 360  # All unique


def test_registry_dimensions():
    """Verify supported dimensions: 2, 4, 8, 16, 32."""
    assert CarpsBBOBFullRegistry.SUPPORTED_DIMENSIONS == (2, 4, 8, 16, 32)
    for dim in (2, 4, 8, 16, 32):
        dim_tasks = CarpsBBOBFullRegistry.get_tasks_by_dimension(dim)
        assert len(dim_tasks) == 72  # 24 * 3 = 72 per dimension


def test_registry_task_info_metadata():
    """Verify metadata extraction and categorical attribution for individual tasks."""
    info = CarpsBBOBFullRegistry.get_task_info("+task=BBOB/cfg_16_12_0")
    assert isinstance(info, BBOBTaskInfo)
    assert info.dimension == 16
    assert info.fid == 12
    assert info.iid == 0
    assert info.task_arg == "+task=BBOB/cfg_16_12_0"
    assert info.function_name == "Bent Cigar"
    assert info.category == BBOBFunctionCategory.ILL_CONDITIONED

    info_f1 = CarpsBBOBFullRegistry.get_task_info("+task=BBOB/cfg_2_1_2")
    assert info_f1.dimension == 2
    assert info_f1.fid == 1
    assert info_f1.iid == 2
    assert info_f1.category == BBOBFunctionCategory.SEPARABLE

    info_f24 = CarpsBBOBFullRegistry.get_task_info("+task=BBOB/cfg_32_24_1")
    assert info_f24.dimension == 32
    assert info_f24.fid == 24
    assert info_f24.iid == 1
    assert info_f24.category == BBOBFunctionCategory.MULTIMODAL_WEAK_GLOBAL


def test_registry_dimension_tier_filtering():
    """Verify grouping by low, mid, and high dimensional tiers."""
    low_d = CarpsBBOBFullRegistry.get_tasks_by_tier("low_d")   # <= 4D (2, 4): 144
    mid_d = CarpsBBOBFullRegistry.get_tasks_by_tier("mid_d")   # 8D: 72
    high_d = CarpsBBOBFullRegistry.get_tasks_by_tier("high_d") # >= 16D (16, 32): 144

    assert len(low_d) == 144
    assert len(mid_d) == 72
    assert len(high_d) == 144
    assert len(low_d) + len(mid_d) + len(high_d) == 360


def test_registry_invalid_inputs():
    """Verify validation guards on invalid dimension or task string."""
    with pytest.raises(ValueError):
        CarpsBBOBFullRegistry.get_tasks_by_dimension(10)

    with pytest.raises(ValueError):
        CarpsBBOBFullRegistry.get_task_info("+task=BBOB/cfg_invalid_task")
