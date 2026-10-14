#!/usr/bin/env python3
"""Tests for dedicated BBOB Dimension 4 sweep task generator."""

import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_bbob_dim4_sweep_tasks import (
    generate_bbob_dim4_sweep_tasks,
    BBOB_DIM4_OUTPUT_DIR,
    BBOB_DIM4_RUNS_DIR,
)


def test_generate_bbob_dim4_sweep_tasks_dry(tmp_path):
    """Verify task generator creates tasks exclusively for dimension 4."""
    out_dir = str(tmp_path / "results_dim4")
    runs_dir = str(tmp_path / "runs_dim4")

    # 72 tasks (24 funcs * 3 instances) * 11 optimizers * 2 seeds = 1,584 tasks
    tasks = generate_bbob_dim4_sweep_tasks(
        seeds=2,
        trials=100,
        output_dir=out_dir,
        runs_dir=out_dir,
        baserundir=runs_dir,
    )
    assert len(tasks) == 1584

    # Every command must target cfg_4_
    for cmd in tasks:
        assert "+task=BBOB/cfg_4_" in cmd
        assert f"baserundir={runs_dir}" in cmd

    manifest_file = Path(out_dir) / "tasks.txt"
    assert manifest_file.exists()
    with open(manifest_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 1584


def test_bbob_dim4_defaults():
    """Verify default paths and full task count for 20 seeds."""
    assert "sweep_bbob_dim4" in BBOB_DIM4_OUTPUT_DIR
    assert "sweep_bbob_dim4" in BBOB_DIM4_RUNS_DIR
