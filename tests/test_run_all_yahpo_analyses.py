#!/usr/bin/env python3
"""Tests for YAHPO unified analysis orchestrator (scripts/run_all_yahpo_analyses.py)."""

from __future__ import annotations

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

from scripts.run_all_yahpo_analyses import (
    list_available_tasks,
    resolve_tasks,
    get_default_paths,
    run_yahpo_analysis_task,
    run_orchestrator,
    main,
    YAHPO_SUITES,
)


@pytest.fixture
def synthetic_yahpo_df():
    """Generates synthetic multi-task logs for YAHPO analysis."""
    records = []
    tasks = [f"cfg_task_{i}" for i in range(1, 6)]
    opts = ["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei", "SMAC20_ProximityA_LCB"]
    for task_id in tasks:
        for opt in opts:
            for seed in [1, 2]:
                for trial in [1, 10]:
                    cost = 0.2 if opt == "SMAC20_ProximityA_LCB" else 0.5
                    records.append({
                        "task_id": task_id,
                        "optimizer_id": opt,
                        "seed": seed,
                        "trial": trial,
                        "trial_value__cost_inc": cost,
                    })
    return pd.DataFrame(records)


def test_yahpo_suites_constant():
    assert "ranger" in YAHPO_SUITES
    assert "super" in YAHPO_SUITES


def test_list_available_tasks():
    tasks = list_available_tasks()
    assert tasks == ["ranger_carps", "ranger_custom", "super_carps", "super_custom"]


def test_resolve_tasks_aliases():
    # 'all'
    all_resolved = resolve_tasks(["all"])
    assert all_resolved == ["ranger_carps", "ranger_custom", "super_carps", "super_custom"]

    # 'ranger_all'
    ranger_all = resolve_tasks(["ranger_all"])
    assert ranger_all == ["ranger_carps", "ranger_custom"]

    # 'super_all'
    super_all = resolve_tasks(["super_all"])
    assert super_all == ["super_carps", "super_custom"]

    # 'carps_all'
    carps_all = resolve_tasks(["carps_all"])
    assert carps_all == ["ranger_carps", "super_carps"]

    # 'custom_all'
    custom_all = resolve_tasks(["custom_all"])
    assert custom_all == ["ranger_custom", "super_custom"]

    # explicit individual tasks
    assert resolve_tasks(["ranger_carps"]) == ["ranger_carps"]
    assert resolve_tasks(["ranger_custom"]) == ["ranger_custom"]
    assert resolve_tasks(["super_carps"]) == ["super_carps"]
    assert resolve_tasks(["super_custom"]) == ["super_custom"]

    # combined & duplicate deduplication
    assert resolve_tasks(["ranger_all", "ranger_carps", "super_custom"]) == [
        "ranger_carps",
        "ranger_custom",
        "super_custom",
    ]

    # invalid task raises ValueError
    with pytest.raises(ValueError, match="Unknown task"):
        resolve_tasks(["invalid_task_name"])


def test_get_default_paths(tmp_path):
    in_r_carps, out_r_carps = get_default_paths("ranger_carps", base_dir=tmp_path)
    assert in_r_carps == tmp_path / "results" / "sweep_yahpo_rbv2_ranger" / "logs.parquet"
    assert out_r_carps == tmp_path / "results" / "sweep_yahpo_rbv2_ranger" / "carps_report"

    in_r_custom, out_r_custom = get_default_paths("ranger_custom", base_dir=tmp_path)
    assert in_r_custom == tmp_path / "results" / "sweep_yahpo_rbv2_ranger" / "logs.parquet"
    assert out_r_custom == tmp_path / "results" / "sweep_yahpo_rbv2_ranger" / "analysis"

    in_s_carps, out_s_carps = get_default_paths("super_carps", base_dir=tmp_path)
    assert in_s_carps == tmp_path / "results" / "sweep_yahpo_rbv2_super" / "logs.parquet"
    assert out_s_carps == tmp_path / "results" / "sweep_yahpo_rbv2_super" / "carps_report"

    in_s_custom, out_s_custom = get_default_paths("super_custom", base_dir=tmp_path)
    assert in_s_custom == tmp_path / "results" / "sweep_yahpo_rbv2_super" / "logs.parquet"
    assert out_s_custom == tmp_path / "results" / "sweep_yahpo_rbv2_super" / "analysis"


def test_dry_run_task():
    res = run_yahpo_analysis_task(
        task_name="ranger_custom",
        input_path="dummy/path.parquet",
        output_dir="dummy/out",
        dry_run=True,
    )
    assert res["status"] == "DRY_RUN"
    assert res["task"] == "ranger_custom"
    assert res["input"] == "dummy/path.parquet"
    assert res["output"] == "dummy/out"


def test_dry_run_orchestrator():
    results = run_orchestrator(tasks=["all"], dry_run=True)
    assert len(results) == 4
    for r in results:
        assert r["status"] == "DRY_RUN"
        assert r["task"] in list_available_tasks()


def test_run_custom_task_ranger_execution(synthetic_yahpo_df, tmp_path):
    input_file = tmp_path / "logs.parquet"
    synthetic_yahpo_df.to_parquet(input_file)
    out_dir = tmp_path / "ranger_analysis"

    res = run_yahpo_analysis_task(
        task_name="ranger_custom",
        input_path=str(input_file),
        output_dir=str(out_dir),
        dry_run=False,
    )
    assert res["status"] == "SUCCESS"
    assert (out_dir / "yahpo_rbv2_ranger_scorecard.csv").exists()
    assert (out_dir / "global_scorecard.csv").exists()
    assert (out_dir / "yahpo_rbv2_ranger_analysis_report.md").exists()


def test_run_custom_task_super_execution(synthetic_yahpo_df, tmp_path):
    input_file = tmp_path / "logs.parquet"
    synthetic_yahpo_df.to_parquet(input_file)
    out_dir = tmp_path / "super_analysis"

    res = run_yahpo_analysis_task(
        task_name="super_custom",
        input_path=str(input_file),
        output_dir=str(out_dir),
        dry_run=False,
    )
    assert res["status"] == "SUCCESS"
    assert (out_dir / "yahpo_rbv2_super_scorecard.csv").exists()
    assert (out_dir / "global_scorecard.csv").exists()
    assert (out_dir / "yahpo_rbv2_super_analysis_report.md").exists()


@patch("carps.analysis.generate_report.generate_report")
def test_run_carps_task_execution(mock_gen_report, tmp_path):
    input_file = tmp_path / "logs.parquet"
    input_file.touch()
    out_dir = tmp_path / "carps_out"

    res = run_yahpo_analysis_task(
        task_name="ranger_carps",
        input_path=str(input_file),
        output_dir=str(out_dir),
        dry_run=False,
    )
    assert res["status"] == "SUCCESS"
    mock_gen_report.assert_called_once()


def test_fail_fast_behavior():
    with patch("scripts.run_all_yahpo_analyses.run_yahpo_analysis_task") as mock_task:
        # First task fails, second would succeed
        mock_task.side_effect = [
            {"task": "ranger_custom", "status": "FAILED", "error": "mock error"},
            {"task": "super_custom", "status": "SUCCESS"},
        ]

        # With fail_fast=True: stops after first failure
        results_fail_fast = run_orchestrator(
            tasks=["ranger_custom", "super_custom"],
            fail_fast=True,
        )
        assert len(results_fail_fast) == 1
        assert results_fail_fast[0]["status"] == "FAILED"

        # Reset mock
        mock_task.side_effect = [
            {"task": "ranger_custom", "status": "FAILED", "error": "mock error"},
            {"task": "super_custom", "status": "SUCCESS"},
        ]

        # With fail_fast=False: continues to second task
        results_continue = run_orchestrator(
            tasks=["ranger_custom", "super_custom"],
            fail_fast=False,
        )
        assert len(results_continue) == 2
        assert results_continue[0]["status"] == "FAILED"
        assert results_continue[1]["status"] == "SUCCESS"


def test_cli_execution_dry_run():
    ret = main(["--dry-run", "--tasks", "all"])
    assert ret == 0

    ret_alias = main(["--dry-run", "--tasks", "ranger_all", "--fail-fast"])
    assert ret_alias == 0


def test_bash_launcher_syntax_and_exec():
    launcher = Path(PROJECT_ROOT) / "scripts" / "run_all_yahpo_analyses.sh"
    assert launcher.exists(), f"Launcher script {launcher} must exist."
    assert os.access(launcher, os.X_OK), f"Launcher script {launcher} must be executable."

    res = subprocess.run(["bash", "-n", str(launcher)], capture_output=True, text=True)
    assert res.returncode == 0, f"Syntax error in launcher: {res.stderr}"

    help_res = subprocess.run([str(launcher), "--help"], capture_output=True, text=True)
    assert help_res.returncode == 0
    assert "YAHPO" in help_res.stdout or "yahpo" in help_res.stdout.lower()
