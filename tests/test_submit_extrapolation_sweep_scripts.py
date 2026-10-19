#!/usr/bin/env python3
"""Tests for Extrapolation UQ Sweep SLURM submission scripts."""

import os
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_extrapolation_sbatch_array_script():
    """Verify array sbatch script references logs, tasks, and single-threading."""
    sbatch_path = Path("scripts/submit_extrapolation_sweep_array.sbatch")
    assert sbatch_path.exists()
    assert os.access(sbatch_path, os.X_OK)

    content = sbatch_path.read_text(encoding="utf-8")
    assert "results/extrapolation_uq/logs" in content
    assert "results/extrapolation_sweep_tasks.txt" in content
    assert "export OMP_NUM_THREADS=1" in content
    assert "export MKL_NUM_THREADS=1" in content
    assert "export OPENBLAS_NUM_THREADS=1" in content
    assert "export NUMEXPR_NUM_THREADS=1" in content
    assert "TASK_FILE=" in content


def test_extrapolation_sbatch_master_script():
    """Verify master sbatch script references master_slurm_scheduler.py and array script."""
    master_path = Path("scripts/submit_extrapolation_sweep_master.sbatch")
    assert master_path.exists()
    assert os.access(master_path, os.X_OK)

    content = master_path.read_text(encoding="utf-8")
    assert "results/extrapolation_uq/logs" in content
    assert "master_slurm_scheduler.py" in content
    assert "results/extrapolation_sweep_tasks.txt" in content
    assert "scripts/submit_extrapolation_sweep_array.sbatch" in content
    assert "results/extrapolation_uq/scheduler_state.json" in content


def test_extrapolation_submit_all_script_dry_run():
    """Verify submit_all runner supports --dry-run and invokes master_slurm_scheduler."""
    runner_path = Path("scripts/submit_extrapolation_sweep_all.sh")
    assert runner_path.exists()
    assert os.access(runner_path, os.X_OK)

    content = runner_path.read_text(encoding="utf-8")
    assert "results/extrapolation_sweep_tasks.txt" in content
    assert "scripts/master_slurm_scheduler.py" in content
    assert "scripts/submit_extrapolation_sweep_array.sbatch" in content

    state_file = Path("results/extrapolation_uq/scheduler_state.json")
    if state_file.exists():
        state_file.unlink()

    # Test dry run execution via bash
    try:
        res = subprocess.run(
            ["bash", str(runner_path), "--dry-run", "--once"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert res.returncode == 0, f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        assert "[DRY-RUN]" in res.stdout
        assert "results/extrapolation_sweep_tasks.txt" in res.stdout
    finally:
        if state_file.exists():
            state_file.unlink()

