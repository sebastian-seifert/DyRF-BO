#!/usr/bin/env python3
"""Tests for dedicated YAHPO Gym rbv2_super sweep task generator and SLURM submission scripts."""

import os
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_yahpo_rbv2_super_sweep_tasks import (
    generate_yahpo_rbv2_super_sweep_tasks,
    YAHPO_SUPER_OUTPUT_DIR,
    YAHPO_SUPER_RUNS_DIR,
)
from scripts.generate_bbob_full_sweep_tasks import ALL_OPTIMIZERS
from scripts.carps_realworld_registry import CarpsRealworldRegistry


def test_generate_yahpo_rbv2_super_sweep_tasks_dry(tmp_path):
    """Verify task generator creates tasks across 103 datasets, 11 optimizers, and paired seeds."""
    out_dir = str(tmp_path / "results_super")
    runs_dir = str(tmp_path / "runs_super")

    super_tasks = CarpsRealworldRegistry.get_super_tasks()
    num_tasks = len(super_tasks)
    assert num_tasks == 103

    # 103 tasks * 11 optimizers * 2 seeds = 2,266 tasks
    tasks = generate_yahpo_rbv2_super_sweep_tasks(
        seeds=2,
        trials=100,
        output_dir=out_dir,
        runs_dir=out_dir,
        baserundir=runs_dir,
    )
    assert len(tasks) == num_tasks * 11 * 2

    # Check task contents
    for cmd in tasks:
        assert "+task=YAHPO/blackbox/cfg_rbv2_super_" in cmd
        assert f"baserundir={runs_dir}" in cmd
        assert "task.optimization_resources.n_trials=100" in cmd

    manifest_file = Path(out_dir) / "tasks.txt"
    assert manifest_file.exists()
    with open(manifest_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == num_tasks * 11 * 2


def test_sbatch_scripts_contain_super_paths():
    """Verify sbatch scripts reference sweep_yahpo_rbv2_super logs and manifest."""
    array_path = Path("scripts/submit_yahpo_rbv2_super_sweep_array.sbatch")
    master_path = Path("scripts/submit_yahpo_rbv2_super_sweep_master.sbatch")
    runner_path = Path("scripts/submit_yahpo_rbv2_super_sweep_all.sh")

    for p in [array_path, master_path, runner_path]:
        assert p.exists()
        assert os.access(p, os.X_OK)

    array_content = array_path.read_text(encoding="utf-8")
    assert "results/sweep_yahpo_rbv2_super/logs" in array_content
    assert "results/sweep_yahpo_rbv2_super/tasks.txt" in array_content
    assert "scripts/run_carps_patched.py" in array_content

    master_content = master_path.read_text(encoding="utf-8")
    assert "supr_master" in master_content
    assert "results/sweep_yahpo_rbv2_super/logs" in master_content
    assert "--chunk-size 250" in master_content

    runner_content = runner_path.read_text(encoding="utf-8")
    assert "master_slurm_scheduler.py" in runner_content
    assert "results/sweep_yahpo_rbv2_super" in runner_content
