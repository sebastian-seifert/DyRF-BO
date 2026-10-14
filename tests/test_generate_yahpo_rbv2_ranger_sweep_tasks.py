#!/usr/bin/env python3
"""Tests for dedicated YAHPO Gym rbv2_ranger sweep task generator."""

import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_yahpo_rbv2_ranger_sweep_tasks import (
    generate_yahpo_rbv2_ranger_sweep_tasks,
    YAHPO_RANGER_OUTPUT_DIR,
    YAHPO_RANGER_RUNS_DIR,
)
from scripts.generate_bbob_full_sweep_tasks import ALL_OPTIMIZERS
from scripts.carps_realworld_registry import CarpsRealworldRegistry


def test_generate_yahpo_rbv2_ranger_sweep_tasks_dry(tmp_path):
    """Verify task generator creates tasks across 119 datasets, 11 optimizers, and paired seeds."""
    out_dir = str(tmp_path / "results_ranger")
    runs_dir = str(tmp_path / "runs_ranger")

    # 119 tasks * 11 optimizers * 2 seeds = 2,618 tasks
    tasks = generate_yahpo_rbv2_ranger_sweep_tasks(
        seeds=2,
        trials=100,
        output_dir=out_dir,
        runs_dir=out_dir,
        baserundir=runs_dir,
    )
    assert len(tasks) == 119 * 11 * 2

    # Check task contents
    for cmd in tasks:
        assert "+task=YAHPO/blackbox/cfg_rbv2_ranger_" in cmd
        assert f"baserundir={runs_dir}" in cmd
        assert "task.optimization_resources.n_trials=100" in cmd

    manifest_file = Path(out_dir) / "tasks.txt"
    assert manifest_file.exists()
    with open(manifest_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 119 * 11 * 2


def test_yahpo_rbv2_ranger_all_optimizers_match_bbob():
    """Verify that the optimizers used exactly match the 11 BBOB optimizers."""
    assert len(ALL_OPTIMIZERS) == 11
    opt_ids = {opt["id"] for opt in ALL_OPTIMIZERS}
    assert "SMAC3_HPOFacade_lcb" in opt_ids
    assert "SMAC3_HPOFacade_ei" in opt_ids
    assert "SMAC20_Entropy_LCB" in opt_ids
    assert "SMAC20_ProximityA_LCB" in opt_ids
    assert "SMAC20_ProximityA_LCB_CV" in opt_ids
    assert "SMAC20_ProximityAC_LCB" in opt_ids
    assert "SMAC20_ProximityAC_LCB_CV" in opt_ids
    assert "SMAC20_ProximityB_LCB" in opt_ids
    assert "SMAC20_ProximityB_LCB_CV" in opt_ids
    assert "SMAC20_ProximityBC_LCB" in opt_ids
    assert "SMAC20_ProximityBC_LCB_CV" in opt_ids


def test_yahpo_rbv2_ranger_canonical_count():
    """Verify that canonical task list contains exactly 119 tasks."""
    ranger_tasks = CarpsRealworldRegistry.get_ranger_tasks()
    assert len(ranger_tasks) == 119
    assert all("cfg_rbv2_ranger_" in t for t in ranger_tasks)


def test_yahpo_rbv2_ranger_defaults():
    """Verify default output and run directory constants."""
    assert "sweep_yahpo_rbv2_ranger" in YAHPO_RANGER_OUTPUT_DIR
    assert "sweep_yahpo_rbv2_ranger" in YAHPO_RANGER_RUNS_DIR
