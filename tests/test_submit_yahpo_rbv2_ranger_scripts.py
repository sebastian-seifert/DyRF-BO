#!/usr/bin/env python3
"""Tests for YAHPO Gym rbv2_ranger SLURM submission scripts and orchestrator."""

import os
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_sbatch_script_contains_ranger_paths():
    """Verify sbatch script references sweep_yahpo_rbv2_ranger logs and manifest."""
    sbatch_path = Path("scripts/submit_yahpo_rbv2_ranger_sweep_array.sbatch")
    assert sbatch_path.exists()
    assert os.access(sbatch_path, os.X_OK)

    content = sbatch_path.read_text(encoding="utf-8")
    assert "results/sweep_yahpo_rbv2_ranger/logs" in content
    assert "results/sweep_yahpo_rbv2_ranger/tasks.txt" in content
    assert "scripts/run_carps_patched.py" in content
    assert "OPENBLAS_NUM_THREADS=1" in content


def test_master_sbatch_script_contains_ranger_config():
    """Verify master sbatch script references scheduler, ranger logs, and chunk size 250."""
    master_path = Path("scripts/submit_yahpo_rbv2_ranger_sweep_master.sbatch")
    assert master_path.exists()
    assert os.access(master_path, os.X_OK)

    content = master_path.read_text(encoding="utf-8")
    assert "rngr_master" in content
    assert "results/sweep_yahpo_rbv2_ranger/logs" in content
    assert "master_slurm_scheduler.py" in content
    assert "--chunk-size 250" in content
    assert "results/sweep_yahpo_rbv2_ranger/tasks.txt" in content


def test_submit_all_script_uses_master_scheduler_with_chunk250():
    """Verify submit_all runner invokes master scheduler with chunk size 250 and supports dry-run."""
    runner_path = Path("scripts/submit_yahpo_rbv2_ranger_sweep_all.sh")
    assert runner_path.exists()
    assert os.access(runner_path, os.X_OK)

    content = runner_path.read_text(encoding="utf-8")
    assert "master_slurm_scheduler.py" in content
    assert "--chunk-size" in content or "CHUNK_SIZE=250" in content

    # Test dry run execution via bash
    res = subprocess.run(
        ["bash", str(runner_path), "--dry-run", "--once"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert res.returncode == 0
    assert "[DRY-RUN]" in res.stdout
    assert "results/sweep_yahpo_rbv2_ranger/tasks.txt" in res.stdout
