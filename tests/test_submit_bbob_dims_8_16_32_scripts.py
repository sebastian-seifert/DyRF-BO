#!/usr/bin/env python3
"""Tests for BBOB Dims 8, 16, and 32 SLURM submission scripts."""

import os
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.mark.parametrize("dim", [8, 16, 32])
def test_sbatch_scripts_exist_and_configured(dim):
    """Verify sbatch worker scripts exist, have execute bits, and configure log/manifest paths."""
    sbatch_path = Path(f"scripts/submit_bbob_dim{dim}_sweep_array.sbatch")
    assert sbatch_path.exists()
    assert os.access(sbatch_path, os.X_OK)

    content = sbatch_path.read_text(encoding="utf-8")
    assert f"results/sweep_bbob_dim{dim}/logs" in content
    assert f"results/sweep_bbob_dim{dim}/tasks.txt" in content
    assert "scripts/run_carps_patched.py" in content


@pytest.mark.parametrize("dim", [8, 16, 32])
def test_master_sbatch_scripts_exist_and_configured(dim):
    """Verify master sbatch scripts reference scheduler, dim logs, and 250 chunk size."""
    master_path = Path(f"scripts/submit_bbob_dim{dim}_sweep_master.sbatch")
    assert master_path.exists()
    assert os.access(master_path, os.X_OK)

    content = master_path.read_text(encoding="utf-8")
    assert f"bbob_d{dim}_master" in content
    assert f"results/sweep_bbob_dim{dim}/logs" in content
    assert "master_slurm_scheduler.py" in content
    assert "--chunk-size 250" in content
    assert f"results/sweep_bbob_dim{dim}/tasks.txt" in content


@pytest.mark.parametrize("dim", [8, 16, 32])
def test_submit_all_scripts_run_dry_run(dim):
    """Verify submit_all runner invokes master scheduler with chunk size 250 and dry-run succeeds."""
    runner_path = Path(f"scripts/submit_bbob_dim{dim}_sweep_all.sh")
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
    assert f"results/sweep_bbob_dim{dim}/tasks.txt" in res.stdout
