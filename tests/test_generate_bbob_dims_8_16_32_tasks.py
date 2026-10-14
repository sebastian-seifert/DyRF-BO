#!/usr/bin/env python3
"""Tests for dedicated BBOB Dimensions 8, 16, and 32 sweep task generators."""

import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_bbob_dim8_sweep_tasks import generate_bbob_dim8_sweep_tasks, BBOB_DIM8_OUTPUT_DIR
from scripts.generate_bbob_dim16_sweep_tasks import generate_bbob_dim16_sweep_tasks, BBOB_DIM16_OUTPUT_DIR
from scripts.generate_bbob_dim32_sweep_tasks import generate_bbob_dim32_sweep_tasks, BBOB_DIM32_OUTPUT_DIR


@pytest.mark.parametrize("dim,gen_func,out_dir_const", [
    (8, generate_bbob_dim8_sweep_tasks, BBOB_DIM8_OUTPUT_DIR),
    (16, generate_bbob_dim16_sweep_tasks, BBOB_DIM16_OUTPUT_DIR),
    (32, generate_bbob_dim32_sweep_tasks, BBOB_DIM32_OUTPUT_DIR),
])
def test_generate_bbob_highdim_sweep_tasks_dry(tmp_path, dim, gen_func, out_dir_const):
    """Verify task generators create tasks exclusively for the specified dimension."""
    out_dir = str(tmp_path / f"results_dim{dim}")
    runs_dir = str(tmp_path / f"runs_dim{dim}")

    # 72 tasks (24 funcs * 3 instances) * 11 optimizers * 2 seeds = 1,584 tasks
    tasks = gen_func(
        seeds=2,
        trials=100,
        output_dir=out_dir,
        runs_dir=out_dir,
        baserundir=runs_dir,
    )
    assert len(tasks) == 1584

    # Every command must target cfg_{dim}_
    for cmd in tasks:
        assert f"+task=BBOB/cfg_{dim}_" in cmd
        assert f"baserundir={runs_dir}" in cmd

    manifest_file = Path(out_dir) / "tasks.txt"
    assert manifest_file.exists()
    with open(manifest_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 1584
    assert f"sweep_bbob_dim{dim}" in out_dir_const
