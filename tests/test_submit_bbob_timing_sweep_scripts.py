#!/usr/bin/env python3
"""Tests for BBOB Timing Sweep SLURM submission scripts."""

import os
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_timing_sbatch_array_script():
    """Verify array sbatch script references timing paths, run_carps_timed, and single-threading."""
    sbatch_path = Path("scripts/submit_bbob_timing_sweep_array.sbatch")
    assert sbatch_path.exists()
    assert os.access(sbatch_path, os.X_OK)

    content = sbatch_path.read_text(encoding="utf-8")
    assert "results/sweep_bbob_timing/logs" in content
    assert "results/sweep_bbob_timing/tasks.txt" in content
    assert "scripts/run_carps_timed.py" in content
    assert "#SBATCH --nodelist=ai-n005" in content
    assert "export OMP_NUM_THREADS=1" in content
    assert "export MKL_NUM_THREADS=1" in content
    assert "export OPENBLAS_NUM_THREADS=1" in content
    assert "export NUMEXPR_NUM_THREADS=1" in content


def test_timing_sbatch_master_script():
    """Verify master sbatch script references scheduler, timing logs, and array script."""
    master_path = Path("scripts/submit_bbob_timing_sweep_master.sbatch")
    assert master_path.exists()
    assert os.access(master_path, os.X_OK)

    content = master_path.read_text(encoding="utf-8")
    assert "results/sweep_bbob_timing/logs" in content
    assert "master_slurm_scheduler.py" in content
    assert "results/sweep_bbob_timing/tasks.txt" in content
    assert "scripts/submit_bbob_timing_sweep_array.sbatch" in content


def test_timing_submit_all_script_dry_run():
    """Verify submit_all runner generates manifest if missing and supports --dry-run."""
    runner_path = Path("scripts/submit_bbob_timing_sweep_all.sh")
    assert runner_path.exists()
    assert os.access(runner_path, os.X_OK)

    content = runner_path.read_text(encoding="utf-8")
    assert "results/sweep_bbob_timing" in content
    assert "scripts/generate_bbob_timing_sweep_tasks.py" in content
    assert "scripts/submit_bbob_timing_sweep_array.sbatch" in content

    # Clean up existing state file if present to test fresh dispatch
    state_file = Path("results/sweep_bbob_timing/scheduler_state.json")
    if state_file.exists():
        state_file.unlink()

    # Test dry run execution via bash
    res = subprocess.run(
        ["bash", str(runner_path), "--dry-run", "--once"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert res.returncode == 0
    assert "[DRY-RUN]" in res.stdout
    assert "results/sweep_bbob_timing/tasks.txt" in res.stdout
