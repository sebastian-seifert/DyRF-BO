"""Unit tests for the BBSubset Held-Out Test task generator."""

import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_bbsubset_test_tasks import (
    ALL_OPTIMIZERS,
    OPTIMIZERS_PART1,
    OPTIMIZERS_PART2,
    generate_bbsubset_test_tasks,
)
from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


def test_optimizer_definitions():
    """Verify that all 11 optimizers are specified with correct IDs."""
    assert len(ALL_OPTIMIZERS) == 11
    assert len(OPTIMIZERS_PART1) == 6
    assert len(OPTIMIZERS_PART2) == 5

    opt_ids = [o["id"] for o in ALL_OPTIMIZERS]
    expected_ids = [
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
    assert opt_ids == expected_ids


def test_generate_bbsubset_test_tasks_smoke(tmp_path: Path):
    """Test generating a small subset of tasks into a temporary directory."""
    test_tasks = CarpsBBSubsetRegistry.get_test_tasks()
    assert len(test_tasks) == 20

    out_dir = tmp_path / "results" / "sweep_bbsubset_test"
    runs_dir = tmp_path / "results" / "sweep_bbsubset_test"
    base_run_dir = tmp_path / "runs" / "sweep_bbsubset_test"

    p1, p2 = generate_bbsubset_test_tasks(
        seeds=[1, 2],
        trials=100,
        output_dir=str(out_dir),
        runs_dir=str(runs_dir),
        baserundir=str(base_run_dir),
    )

    # 20 tasks * 6 opts * 2 seeds = 240
    assert len(p1) == 240
    # 20 tasks * 5 opts * 2 seeds = 200
    assert len(p2) == 200

    assert (out_dir / "tasks_part1.txt").exists()
    assert (out_dir / "tasks_part2.txt").exists()
    assert (out_dir / "tasks_all.txt").exists()

    all_lines = (out_dir / "tasks_all.txt").read_text().strip().split("\n")
    assert len(all_lines) == 440

    first_line = all_lines[0]
    assert f"baserundir={base_run_dir}" in first_line
    assert f"++optimizer.telemetry_path={runs_dir}/telemetry_" in first_line
