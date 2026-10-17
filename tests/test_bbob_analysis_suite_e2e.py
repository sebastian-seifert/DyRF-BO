#!/usr/bin/env python3
"""End-to-End Integration Verification for the Unified BBOB Analysis Pipeline."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_all_bbob_analyses import (
    BBOB_DIMENSIONS,
    list_available_tasks,
    resolve_tasks,
    run_orchestrator,
    run_bbob_analysis_task,
)


def test_e2e_all_tasks_cardinality():
    """Verify all 12 discrete tasks exist and resolve properly."""
    tasks = list_available_tasks()
    assert len(tasks) == 12
    for d in BBOB_DIMENSIONS:
        assert f"carps_dim{d}" in tasks
        assert f"custom_dim{d}" in tasks
    assert "carps_full" in tasks
    assert "custom_full" in tasks


def test_e2e_orchestrator_dry_run_all():
    """Verify complete dry-run execution over all 12 tasks."""
    results = run_orchestrator(tasks=["all"], dry_run=True)
    assert len(results) == 12
    for res in results:
        assert res["status"] == "DRY_RUN"
        assert res["task"] in list_available_tasks()
        assert Path(res["input"]).name in ["logs.parquet", "logs.csv"]


def test_e2e_custom_analyses_simulated_sweep(tmp_path):
    """Simulate complete custom statistical analysis across all 5 dimensions."""
    records = []
    for d in BBOB_DIMENSIONS:
        for opt in ["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei", "SMAC20_ProximityA_LCB"]:
            for seed in [1, 2]:
                records.append({
                    "task_id": f"cfg_{d}_1_0",
                    "optimizer_id": opt,
                    "seed": seed,
                    "trial": 100,
                    "cost": 1.0 if opt == "SMAC20_ProximityA_LCB" else 2.0,
                    "trial_value__cost_inc": 1.0 if opt == "SMAC20_ProximityA_LCB" else 2.0,
                })
    df_sweep = pd.DataFrame(records)

    base_dir = tmp_path / "mock_project"
    for d in BBOB_DIMENSIONS:
        dim_dir = base_dir / "results" / f"sweep_bbob_dim{d}"
        dim_dir.mkdir(parents=True, exist_ok=True)
        df_d = df_sweep[df_sweep["task_id"] == f"cfg_{d}_1_0"]
        df_d.to_parquet(dim_dir / "logs.parquet")

    full_dir = base_dir / "results" / "sweep_bbob_full_comparison"
    full_dir.mkdir(parents=True, exist_ok=True)
    df_sweep.to_parquet(full_dir / "logs.parquet")

    # Run all custom tasks
    results = run_orchestrator(
        tasks=["custom_all"],
        dry_run=False,
        base_dir=base_dir,
    )
    assert len(results) == 6
    for r in results:
        assert r["status"] == "SUCCESS", f"Task {r['task']} failed with {r.get('error')}"

    # Verify scorecards and reports generated for each dimension
    for d in BBOB_DIMENSIONS:
        out_d = base_dir / "results" / f"sweep_bbob_dim{d}" / "analysis"
        assert (out_d / f"bbob_dim{d}_scorecard.csv").exists()
        assert (out_d / f"bbob_dim{d}_analysis_report.md").exists()

    # Verify full sweep scorecard and report
    out_full = base_dir / "results" / "sweep_bbob_full_comparison" / "analysis"
    assert (out_full / "global_scorecard.csv").exists()
    assert (out_full / "bbob_analysis_report.md").exists()
    for d in BBOB_DIMENSIONS:
        assert (out_full / f"scorecard_dim_{d}.csv").exists()


def test_e2e_shell_script_help():
    """Verify bash script executes --help correctly."""
    launcher = Path(PROJECT_ROOT) / "scripts" / "run_all_bbob_analyses.sh"
    res = subprocess.run([str(launcher), "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Unified BBOB Analysis Runner" in res.stdout
