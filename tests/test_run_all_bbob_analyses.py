#!/usr/bin/env python3
"""Tests for BBOB unified analysis orchestrator (scripts/run_all_bbob_analyses.py)."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_all_bbob_analyses import (
    list_available_tasks,
    resolve_tasks,
    run_bbob_analysis_task,
    run_orchestrator,
    BBOB_DIMENSIONS,
)


def test_bbob_dimensions_constant():
    assert BBOB_DIMENSIONS == [2, 4, 8, 16, 32]


def test_list_available_tasks():
    tasks = list_available_tasks()
    for d in [2, 4, 8, 16, 32]:
        assert f"carps_dim{d}" in tasks
        assert f"custom_dim{d}" in tasks
    assert "carps_full" in tasks
    assert "custom_full" in tasks


def test_resolve_tasks_aliases():
    # 'all'
    all_resolved = resolve_tasks(["all"])
    assert len(all_resolved) == len(list_available_tasks())

    # 'carps_all'
    carps_all = resolve_tasks(["carps_all"])
    assert all(t.startswith("carps_") for t in carps_all)
    assert len(carps_all) == 6  # 5 dims + full

    # 'custom_all'
    custom_all = resolve_tasks(["custom_all"])
    assert all(t.startswith("custom_") for t in custom_all)
    assert len(custom_all) == 6

    # 'dim4'
    dim4_tasks = resolve_tasks(["dim4"])
    assert dim4_tasks == ["carps_dim4", "custom_dim4"]

    # explicit list
    explicit = resolve_tasks(["custom_dim2", "carps_dim8"])
    assert explicit == ["custom_dim2", "carps_dim8"]

    # invalid task raises ValueError
    with pytest.raises(ValueError, match="Unknown task"):
        resolve_tasks(["non_existent_task"])


def test_dry_run_task():
    res = run_bbob_analysis_task(
        task_name="custom_dim2",
        input_path="dummy/path.parquet",
        output_dir="dummy/out",
        dry_run=True,
    )
    assert res["status"] == "DRY_RUN"
    assert res["task"] == "custom_dim2"


def test_run_custom_task_execution(tmp_path):
    # Create valid synthetic logs
    records = []
    for opt in ["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei", "SMAC20_ProximityA_LCB"]:
        for seed in [1, 2]:
            records.append({
                "task_id": "cfg_2_1_0",
                "optimizer_id": opt,
                "seed": seed,
                "trial": 100,
                "cost": 1.0 if opt == "SMAC20_ProximityA_LCB" else 2.0,
                "trial_value__cost_inc": 1.0 if opt == "SMAC20_ProximityA_LCB" else 2.0,
            })
    input_file = tmp_path / "logs.parquet"
    pd.DataFrame(records).to_parquet(input_file)

    out_dir = tmp_path / "custom_out"
    res = run_bbob_analysis_task(
        task_name="custom_dim2",
        input_path=str(input_file),
        output_dir=str(out_dir),
        dry_run=False,
    )
    assert res["status"] == "SUCCESS"
    assert (out_dir / "bbob_dim2_scorecard.csv").exists()
    assert (out_dir / "bbob_dim2_analysis_report.md").exists()


@patch("carps.analysis.generate_report.generate_report")
def test_run_carps_task_execution(mock_gen_report, tmp_path):
    input_file = tmp_path / "logs.parquet"
    input_file.touch()
    out_dir = tmp_path / "carps_out"

    res = run_bbob_analysis_task(
        task_name="carps_dim4",
        input_path=str(input_file),
        output_dir=str(out_dir),
        dry_run=False,
    )
    assert res["status"] == "SUCCESS"
    mock_gen_report.assert_called_once()


def test_bash_launcher_syntax_and_exec():
    launcher = Path(PROJECT_ROOT) / "scripts" / "run_all_bbob_analyses.sh"
    assert launcher.exists(), f"Launcher script {launcher} must exist."
    assert os.access(launcher, os.X_OK), f"Launcher script {launcher} must be executable."

    res = subprocess.run(["bash", "-n", str(launcher)], capture_output=True, text=True)
    assert res.returncode == 0, f"Syntax error in launcher: {res.stderr}"
