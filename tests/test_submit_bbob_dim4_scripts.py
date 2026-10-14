#!/usr/bin/env python3
"""Tests for BBOB Dim 4 SLURM submission scripts."""

import os
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_sbatch_script_contains_dim4_paths():
    """Verify sbatch script references sweep_bbob_dim4 logs and default manifest."""
    sbatch_path = Path("scripts/submit_bbob_dim4_sweep_array.sbatch")
    assert sbatch_path.exists()
    assert os.access(sbatch_path, os.X_OK)

    content = sbatch_path.read_text(encoding="utf-8")
    assert "results/sweep_bbob_dim4/logs" in content
    assert "results/sweep_bbob_dim4/tasks.txt" in content
    assert "scripts/run_carps_patched.py" in content


def test_master_sbatch_script_contains_dim4_config():
    """Verify master sbatch script references scheduler, dim4 logs, and 250 chunk size."""
    master_path = Path("scripts/submit_bbob_dim4_sweep_master.sbatch")
    assert master_path.exists()
    assert os.access(master_path, os.X_OK)

    content = master_path.read_text(encoding="utf-8")
    assert "bbob_d4_master" in content
    assert "results/sweep_bbob_dim4/logs" in content
    assert "master_slurm_scheduler.py" in content
    assert "--chunk-size 250" in content
    assert "results/sweep_bbob_dim4/tasks.txt" in content


def test_submit_all_script_uses_master_scheduler_with_chunk250():
    """Verify submit_all runner invokes master scheduler with chunk size 250."""
    runner_path = Path("scripts/submit_bbob_dim4_sweep_all.sh")
    assert runner_path.exists()
    assert os.access(runner_path, os.X_OK)

    content = runner_path.read_text(encoding="utf-8")
    assert "master_slurm_scheduler.py" in content
    assert "--chunk-size 250" in content or "CHUNK_SIZE=250" in content

    # Test dry run execution via bash
    res = subprocess.run(
        ["bash", str(runner_path), "--dry-run", "--once"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert res.returncode == 0
    assert "[DRY-RUN]" in res.stdout
    assert "results/sweep_bbob_dim4/tasks.txt" in res.stdout
