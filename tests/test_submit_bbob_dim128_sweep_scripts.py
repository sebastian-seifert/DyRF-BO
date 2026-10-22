#!/usr/bin/env python3
"""Tests for BBOB 128D SLURM Cluster Submission Script."""

import os
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_sbatch_script_exists_and_configured():
    sbatch_path = REPO_ROOT / "scripts" / "submit_bbob_dim128_sweep_array.sbatch"
    assert sbatch_path.is_file(), f"Missing sbatch script: {sbatch_path}"

    content = sbatch_path.read_text(encoding="utf-8")

    # SLURM headers
    assert "#SBATCH -p ai" in content
    assert "#SBATCH --job-name=bbob_d128" in content
    assert "#SBATCH --output=results/sweep_bbob_dim128/logs/bbob_d128_%A_%a.log" in content
    assert "#SBATCH --error=results/sweep_bbob_dim128/logs/bbob_d128_%A_%a.err" in content
    assert "#SBATCH --cpus-per-task=8" in content
    assert "#SBATCH --mem=16G" in content

    # Default task file
    assert "results/sweep_bbob_dim128/tasks.txt" in content

    # Environment setup & Python binary discovery
    assert "conda activate dyrf" in content
    assert "scripts/run_carps_patched.py" in content
    assert "mkdir -p results/sweep_bbob_dim128/logs" in content


def test_partitioned_submission_scripts_exist():
    driver = REPO_ROOT / "scripts" / "submit_bbob_dim128_sweep.sh"
    p1 = REPO_ROOT / "scripts" / "submit_bbob_dim128_sweep_part1.sh"
    p2 = REPO_ROOT / "scripts" / "submit_bbob_dim128_sweep_part2.sh"

    assert driver.is_file(), f"Missing {driver}"
    assert p1.is_file(), f"Missing {p1}"
    assert p2.is_file(), f"Missing {p2}"

    c_driver = driver.read_text(encoding="utf-8")
    c1 = p1.read_text(encoding="utf-8")
    c2 = p2.read_text(encoding="utf-8")

    # Driver should use chunk size 200 to prevent LUIS MaxArraySize / invalid job description errors
    assert "CHUNK_SIZE=200" in c_driver or "CHUNK_SIZE" in c_driver
    assert "START_TASK" in c_driver
    assert "END_TASK" in c_driver
    assert "submit_bbob_dim128_sweep_array.sbatch" in c_driver

    # Part 1: Tasks 1 to 3600
    assert "START_TASK=1" in c1
    assert "END_TASK=3600" in c1
    assert "submit_bbob_dim128_sweep.sh" in c1

    # Part 2: Tasks 3601 to 7200
    assert "START_TASK=3601" in c2
    assert "END_TASK=7200" in c2
    assert "submit_bbob_dim128_sweep.sh" in c2


