#!/usr/bin/env python3
"""Tests for BBOB Full Sweep Task Generator."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from scripts.generate_bbob_full_sweep_tasks import (
    ALL_OPTIMIZERS,
    build_bbob_command_line,
    generate_bbob_full_sweep_tasks,
)


def test_optimizer_definitions():
    """Verify all 11 target optimizers from BBSubset big comparison are present."""
    assert len(ALL_OPTIMIZERS) == 11
    opt_ids = [opt["id"] for opt in ALL_OPTIMIZERS]
    expected = [
        "SMAC3_HPOFacade_lcb",
        "SMAC3_HPOFacade_ei",
        "SMAC20_Entropy_LCB",
        "SMAC20_ProximityA_LCB",
        "SMAC20_ProximityA_LCB_CV",
        "SMAC20_ProximityAC_LCB",
        "SMAC20_ProximityAC_LCB_CV",
        "SMAC20_ProximityB_LCB",
        "SMAC20_ProximityB_LCB_CV",
        "SMAC20_ProximityBC_LCB",
        "SMAC20_ProximityBC_LCB_CV",
    ]
    assert opt_ids == expected


def test_build_bbob_command_line():
    """Verify CLI generation formats arguments, telemetry, and paths correctly."""
    opt_info = ALL_OPTIMIZERS[0]
    cmd = build_bbob_command_line(
        opt_info=opt_info,
        task_arg="+task=BBOB/cfg_16_12_0",
        seed=5,
        trials=100,
        runs_dir="results/sweep_bbob_full_comparison",
        baserundir="runs/sweep_bbob_full_comparison",
    )
    assert "--config-dir carps_integration/configs" in cmd
    assert "+optimizer=smac3_hpo_facade_lcb" in cmd
    assert "+task=BBOB/cfg_16_12_0" in cmd
    assert "task.optimization_resources.n_trials=100" in cmd
    assert "seed=5" in cmd
    assert "telemetry_SMAC3_HPOFacade_lcb_cfg_16_12_0_seed5.json" in cmd
    assert "baserundir=runs/sweep_bbob_full_comparison" in cmd
    assert "optimizer_id=SMAC3_HPOFacade_lcb" in cmd


def test_generate_bbob_full_sweep_tasks_dry(tmp_path):
    """Verify task count generation: 360 tasks * 11 optimizers * 2 seeds = 7,920 tasks for testing."""
    out_dir = str(tmp_path / "tasks_output")
    tasks = generate_bbob_full_sweep_tasks(
        seeds=2,
        trials=100,
        output_dir=out_dir,
    )
    # 360 tasks * 11 optimizers * 2 seeds = 7920
    assert len(tasks) == 7920

    manifest_file = os.path.join(out_dir, "tasks_all.txt")
    assert os.path.exists(manifest_file)
    with open(manifest_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 7920


def test_generate_bbob_tasks_by_dimension_filter(tmp_path):
    """Verify generating tasks filtered by specific dimension."""
    out_dir = str(tmp_path / "tasks_dim16")
    tasks = generate_bbob_full_sweep_tasks(
        seeds=1,
        trials=100,
        dimensions=[16],
        output_dir=out_dir,
    )
    # 72 tasks * 11 optimizers * 1 seed = 792
    assert len(tasks) == 792
