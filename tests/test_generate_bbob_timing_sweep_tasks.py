#!/usr/bin/env python3
"""Tests for dedicated BBOB Dimensionality Timing Sweep task generator."""

import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_bbob_timing_sweep_tasks import (
    generate_bbob_timing_sweep_tasks,
    TIMING_SWEEP_OUTPUT_DIR,
    TIMING_SWEEP_RUNS_DIR,
    TIMING_DIMENSIONS,
    TIMING_FUNCTION_ID,
    TIMING_INSTANCE_ID,
)
from scripts.generate_bbob_full_sweep_tasks import ALL_OPTIMIZERS


def test_timing_sweep_task_count_and_targets(tmp_path):
    """Verify task generator creates exactly 1,100 tasks with correct dimensions and optimizers."""
    out_dir = str(tmp_path / "results_timing")
    runs_dir = str(tmp_path / "runs_timing")

    # 5 dimensions * 1 function * 11 optimizers * 20 seeds = 1,100 tasks
    tasks = generate_bbob_timing_sweep_tasks(
        seeds=20,
        trials=100,
        output_dir=out_dir,
        runs_dir=out_dir,
        baserundir=runs_dir,
        shuffle=True,
        shuffle_seed=42,
    )
    assert len(tasks) == 1100

    # Ensure all target dimensions are represented equally
    for dim in [2, 4, 8, 16, 32]:
        dim_tasks = [t for t in tasks if f"+task=BBOB/cfg_{dim}_8_0" in t]
        assert len(dim_tasks) == 11 * 20

    # Ensure all 11 optimizers are represented equally
    for opt in ALL_OPTIMIZERS:
        opt_tasks = [t for t in tasks if f"optimizer_id={opt['id']} " in t]
        assert len(opt_tasks) == 5 * 20

    # Ensure manifest file exists and matches
    manifest_file = Path(out_dir) / "tasks.txt"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 1100


def test_timing_sweep_deterministic_shuffling(tmp_path):
    """Verify that shuffling is deterministic given a seed and different when disabled."""
    out1 = str(tmp_path / "out1")
    out2 = str(tmp_path / "out2")
    out3 = str(tmp_path / "out3")

    tasks1 = generate_bbob_timing_sweep_tasks(seeds=2, output_dir=out1, shuffle=True, shuffle_seed=123)
    tasks2 = generate_bbob_timing_sweep_tasks(seeds=2, output_dir=out2, shuffle=True, shuffle_seed=123)
    tasks_unshuffled = generate_bbob_timing_sweep_tasks(seeds=2, output_dir=out3, shuffle=False)

    assert tasks1 == tasks2
    assert tasks1 != tasks_unshuffled
    assert set(tasks1) == set(tasks_unshuffled)


def test_timing_sweep_constants():
    """Verify constants match the benchmark specifications."""
    assert TIMING_DIMENSIONS == (2, 4, 8, 16, 32)
    assert TIMING_FUNCTION_ID == 8
    assert TIMING_INSTANCE_ID == 0
    assert "sweep_bbob_timing" in TIMING_SWEEP_OUTPUT_DIR
    assert "sweep_bbob_timing" in TIMING_SWEEP_RUNS_DIR
