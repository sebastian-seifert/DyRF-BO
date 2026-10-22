#!/usr/bin/env python3
"""Tests for BBOB Dimension 128 Sweep Task Generator."""

import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_bbob_dim128_sweep_tasks import (
    DIM128_OPTIMIZERS,
    generate_bbob_dim128_sweep_tasks,
)


def test_dim128_optimizers_are_strictly_five():
    expected_ids = [
        "SMAC3_HPOFacade_lcb",
        "SMAC3_HPOFacade_ei",
        "SMAC20_ProximityAC_LCB",
        "SMAC20_ProximityBC_LCB",
        "SMAC20_Entropy_LCB",
    ]
    actual_ids = [opt["id"] for opt in DIM128_OPTIMIZERS]
    assert sorted(actual_ids) == sorted(expected_ids)
    assert len(DIM128_OPTIMIZERS) == 5


def test_generate_bbob_dim128_sweep_tasks_counts(tmp_path):
    # Test with 2 seeds and 10 trials for speed
    cmds = generate_bbob_dim128_sweep_tasks(
        seeds=[1, 2],
        trials=10,
        output_dir=str(tmp_path),
        runs_dir=str(tmp_path / "runs"),
        baserundir=str(tmp_path / "baseruns"),
    )

    # 72 tasks * 5 optimizers * 2 seeds = 720 commands
    assert len(cmds) == 720

    # Verify command contents
    sample = cmds[0]
    assert "--config-dir carps_integration/configs" in sample
    assert "cfg_128_" in sample
    assert "task.optimization_resources.n_trials=10" in sample
    assert "seed=" in sample

    # Verify task manifest files written to disk
    tasks_all = tmp_path / "tasks_all.txt"
    tasks_txt = tmp_path / "tasks.txt"
    assert tasks_all.is_file()
    assert tasks_txt.is_file()
    assert len(tasks_all.read_text().strip().split("\n")) == 720
    assert len(tasks_txt.read_text().strip().split("\n")) == 720


def test_generate_bbob_dim128_default_total_count(monkeypatch, tmp_path):
    # Verify default parameter contract: 20 seeds * 5 optimizers * 72 tasks = 7,200 runs
    cmds = generate_bbob_dim128_sweep_tasks(
        seeds=20,
        trials=100,
        output_dir=str(tmp_path),
        runs_dir=str(tmp_path / "runs"),
        baserundir=str(tmp_path / "baseruns"),
    )
    assert len(cmds) == 7200
