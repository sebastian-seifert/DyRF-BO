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
