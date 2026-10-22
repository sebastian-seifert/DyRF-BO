#!/usr/bin/env python3
"""CARP-S BBOB Full Benchmark Suite Registry with Dimensionality and Function Metadata.

Covers all 360 standard BBOB tasks in CARP-S:
- Dimensions: D in {2, 4, 8, 16, 32} (5 dimensions)
- Functions: f1 to f24 (24 functions)
- Instances: iid in {0, 1, 2} (3 instances)
- Total: 5 * 24 * 3 = 360 tasks
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Tuple


class BBOBFunctionCategory(str, Enum):
    SEPARABLE = "separable"
    MODERATE_CONDITIONING = "moderate_conditioning"
    ILL_CONDITIONED = "ill_conditioned"
    MULTIMODAL_ADEQUATE_GLOBAL = "multimodal_adequate_global"
    MULTIMODAL_WEAK_GLOBAL = "multimodal_weak_global"


BBOB_FUNCTION_METADATA: Dict[int, Tuple[str, BBOBFunctionCategory]] = {
    1: ("Sphere", BBOBFunctionCategory.SEPARABLE),
    2: ("Ellipsoidal", BBOBFunctionCategory.SEPARABLE),
    3: ("Rastrigin", BBOBFunctionCategory.SEPARABLE),
    4: ("Buche-Rastrigin", BBOBFunctionCategory.SEPARABLE),
    5: ("Linear Slope", BBOBFunctionCategory.SEPARABLE),
    6: ("Attractive Sector", BBOBFunctionCategory.MODERATE_CONDITIONING),
    7: ("Step Ellipsoidal", BBOBFunctionCategory.MODERATE_CONDITIONING),
    8: ("Rosenbrock", BBOBFunctionCategory.MODERATE_CONDITIONING),
    9: ("Rosenbrock Rotated", BBOBFunctionCategory.MODERATE_CONDITIONING),
    10: ("Ellipsoidal High Cond", BBOBFunctionCategory.ILL_CONDITIONED),
    11: ("Discus", BBOBFunctionCategory.ILL_CONDITIONED),
    12: ("Bent Cigar", BBOBFunctionCategory.ILL_CONDITIONED),
    13: ("Sharp Ridge", BBOBFunctionCategory.ILL_CONDITIONED),
    14: ("Different Powers", BBOBFunctionCategory.ILL_CONDITIONED),
    15: ("Rastrigin Multi", BBOBFunctionCategory.MULTIMODAL_ADEQUATE_GLOBAL),
    16: ("Weierstrass", BBOBFunctionCategory.MULTIMODAL_ADEQUATE_GLOBAL),
    17: ("Schaffer F7 Cond 10", BBOBFunctionCategory.MULTIMODAL_ADEQUATE_GLOBAL),
    18: ("Schaffer F7 Cond 1000", BBOBFunctionCategory.MULTIMODAL_ADEQUATE_GLOBAL),
    19: ("Griewank-Rosenbrock", BBOBFunctionCategory.MULTIMODAL_ADEQUATE_GLOBAL),
    20: ("Schwefel", BBOBFunctionCategory.MULTIMODAL_WEAK_GLOBAL),
    21: ("Gallagher 101 Peaks", BBOBFunctionCategory.MULTIMODAL_WEAK_GLOBAL),
    22: ("Gallagher 21 Peaks", BBOBFunctionCategory.MULTIMODAL_WEAK_GLOBAL),
    23: ("Katsuura", BBOBFunctionCategory.MULTIMODAL_WEAK_GLOBAL),
    24: ("Lunacek bi-Rastrigin", BBOBFunctionCategory.MULTIMODAL_WEAK_GLOBAL),
}


@dataclass(frozen=True)
class BBOBTaskInfo:
    task_arg: str
    dimension: int
    fid: int
    iid: int
    function_name: str
    category: BBOBFunctionCategory


class CarpsBBOBFullRegistry:
    SUPPORTED_DIMENSIONS: Tuple[int, ...] = (2, 4, 8, 16, 32)
    EXTENDED_DIMENSIONS: Tuple[int, ...] = (2, 4, 8, 16, 32, 128)
    NUM_FUNCTIONS: int = 24
    NUM_INSTANCES: int = 3

    @classmethod
    def get_all_tasks(cls) -> List[str]:
        """Returns all 360 standard BBOB tasks formatted as Hydra arguments."""
        tasks: List[str] = []
        for dim in cls.SUPPORTED_DIMENSIONS:
            tasks.extend(cls.get_tasks_by_dimension(dim))
        return tasks

    @classmethod
    def get_tasks_by_dimension(cls, dim: int) -> List[str]:
        """Returns all 72 tasks for a specific dimension."""
        if dim not in cls.EXTENDED_DIMENSIONS:
            raise ValueError(
                f"Unsupported dimension {dim}. Supported dimensions: {cls.EXTENDED_DIMENSIONS}"
            )
        return [
            f"+task=BBOB/cfg_{dim}_{fid}_{iid}"
            for fid in range(1, cls.NUM_FUNCTIONS + 1)
            for iid in range(cls.NUM_INSTANCES)
        ]

    @classmethod
    def get_tasks_by_tier(cls, tier: str) -> List[str]:
        """Returns tasks grouped by dimensional tier: 'low_d', 'mid_d', 'high_d', or 'ultra_high_d'."""
        tier_lower = tier.lower()
        if tier_lower in ("low", "low_d", "low-d"):
            dims = (2, 4)
        elif tier_lower in ("mid", "mid_d", "mid-d"):
            dims = (8,)
        elif tier_lower in ("high", "high_d", "high-d"):
            dims = (16, 32)
        elif tier_lower in ("ultra", "ultra_high_d", "ultra-high-d", "ultra_high"):
            dims = (128,)
        else:
            raise ValueError(f"Unknown tier: {tier}. Expected 'low_d', 'mid_d', 'high_d', or 'ultra_high_d'.")

        tasks: List[str] = []
        for d in dims:
            tasks.extend(cls.get_tasks_by_dimension(d))
        return tasks

    @classmethod
    def get_task_info(cls, task_str: str) -> BBOBTaskInfo:
        """Parses a task argument string and extracts its structured metadata."""
        match = re.search(r"cfg_(\d+)_(\d+)_(\d+)", task_str)
        if not match:
            raise ValueError(f"Invalid BBOB task string format: '{task_str}'. Expected cfg_<dim>_<fid>_<iid>.")

        dim = int(match.group(1))
        fid = int(match.group(2))
        iid = int(match.group(3))

        if dim not in cls.EXTENDED_DIMENSIONS:
            raise ValueError(f"Dimension {dim} not supported.")
        if fid not in BBOB_FUNCTION_METADATA:
            raise ValueError(f"Function ID {fid} invalid. Must be 1..24.")
        if not (0 <= iid < cls.NUM_INSTANCES):
            raise ValueError(f"Instance ID {iid} invalid. Must be 0..2.")

        func_name, category = BBOB_FUNCTION_METADATA[fid]
        normalized_task_arg = f"+task=BBOB/cfg_{dim}_{fid}_{iid}"

        return BBOBTaskInfo(
            task_arg=normalized_task_arg,
            dimension=dim,
            fid=fid,
            iid=iid,
            function_name=func_name,
            category=category,
        )
