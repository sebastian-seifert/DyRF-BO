#!/usr/bin/env python3
"""CARP-S BBOB High-Dimensional and Extreme Benchmark Registry.

Provides structured access to BBOB tasks for:
- D=16 (High-D): 24 functions * 3 instances = 72 tasks
- D=32 (Extreme): 24 functions * 3 instances = 72 tasks
- Total: 144 tasks
"""

from __future__ import annotations

from typing import List, Sequence


class CarpsBBOBHighDimRegistry:
    DIM_16_TASKS: List[str] = [
        f"+task=BBOB/cfg_16_{fid}_{iid}"
        for fid in range(1, 25)
        for iid in range(3)
    ]

    DIM_32_TASKS: List[str] = [
        f"+task=BBOB/cfg_32_{fid}_{iid}"
        for fid in range(1, 25)
        for iid in range(3)
    ]

    @classmethod
    def get_dim16_tasks(cls) -> List[str]:
        """Returns all 72 BBOB tasks in dimension 16."""
        return list(cls.DIM_16_TASKS)

    @classmethod
    def get_dim32_tasks(cls) -> List[str]:
        """Returns all 72 BBOB tasks in dimension 32."""
        return list(cls.DIM_32_TASKS)

    @classmethod
    def get_all_tasks(cls) -> List[str]:
        """Returns all 144 BBOB tasks across dimensions 16 and 32."""
        return list(cls.DIM_16_TASKS) + list(cls.DIM_32_TASKS)

    @classmethod
    def get_tasks_by_dimensions(cls, dimensions: Sequence[int]) -> List[str]:
        """Returns tasks for the specified dimensions (subset of [16, 32])."""
        tasks: List[str] = []
        for dim in dimensions:
            if dim == 16:
                tasks.extend(cls.DIM_16_TASKS)
            elif dim == 32:
                tasks.extend(cls.DIM_32_TASKS)
            else:
                raise ValueError(f"Unsupported dimension {dim} for BBOB High-D registry. Expected 16 or 32.")
        return tasks
