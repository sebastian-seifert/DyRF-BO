#!/usr/bin/env python3
"""Tests for BBOB Dim 2 SLURM submission scripts."""

import os
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_sbatch_script_contains_dim2_paths():
    """Verify sbatch script references sweep_bbob_dim2 logs and default manifest."""
    sbatch_path = Path("scripts/submit_bbob_dim2_sweep_array.sbatch")
    assert sbatch_path.exists()
    assert os.access(sbatch_path, os.X_OK)

    content = sbatch_path.read_text(encoding="utf-8")
    assert "results/sweep_bbob_dim2/logs" in content
    assert "results/sweep_bbob_dim2/tasks.txt" in content
    assert "scripts/run_carps_patched.py" in content


def test_submit_all_script_dry_run():
    """Verify submit_all runner supports --dry-run and partitions tasks."""
    runner_path = Path("scripts/submit_bbob_dim2_sweep_all.sh")
    assert runner_path.exists()
    assert os.access(runner_path, os.X_OK)

    # Test dry run execution via bash
    res = subprocess.run(
        ["bash", str(runner_path), "--dry-run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert res.returncode == 0
    assert "[DRY-RUN]" in res.stdout
    assert "results/sweep_bbob_dim2/tasks.txt" in res.stdout
