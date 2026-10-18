#!/usr/bin/env python3
"""Unit tests for HPOBench Tabular task and dataset availability verification suite."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_hpobench_tasks_available import (
    check_task_configs_available,
    check_dataset_files_available,
    check_sweep_completion_status,
    resolve_hpobench_data_dir,
    download_missing_datasets,
)


def test_check_task_configs_available_ml():
    """Verify all 88 ML tasks in CARP-S registry are detected."""
    res = check_task_configs_available(suite="ml")
    assert isinstance(res, dict)
    assert res["suite"] == "ml"
    assert res["total_expected"] == 88
    assert res["total_found"] == 88
    assert res["all_available"] is True
    assert len(res["missing"]) == 0
    assert len(res["available"]) == 88
    assert res["ml_tasks"]["found"] == 88


def test_check_task_configs_available_nas():
    """Verify all 10 NAS tasks in CARP-S registry are detected."""
    res = check_task_configs_available(suite="nas")
    assert isinstance(res, dict)
    assert res["suite"] == "nas"
    assert res["total_expected"] == 10
    assert res["total_found"] == 10
    assert res["all_available"] is True
    assert len(res["missing"]) == 0
    assert len(res["available"]) == 10
    assert res["nas_tasks"]["found"] == 10


def test_check_task_configs_available_all():
    """Verify combined 98 tasks (88 ML + 10 NAS) are detected."""
    res = check_task_configs_available(suite="all")
    assert isinstance(res, dict)
    assert res["suite"] == "all"
    assert res["total_expected"] == 98
    assert res["total_found"] == 98
    assert res["all_available"] is True
    assert len(res["missing"]) == 0
    assert len(res["available"]) == 98
    assert res["ml_tasks"]["found"] == 88
    assert res["nas_tasks"]["found"] == 10


def test_check_task_configs_available_invalid_suite():
    """Verify ValueError is raised for invalid suite names."""
    with pytest.raises(ValueError, match="Unknown suite"):
        check_task_configs_available(suite="unknown_suite")


def test_resolve_hpobench_data_dir(monkeypatch, tmp_path):
    """Verify data dir resolution with cluster env vars and fallbacks."""
    cluster_carps_dir = tmp_path / "benchmarks"
    cluster_hpobench_dir = cluster_carps_dir / "hpobench"
    cluster_hpobench_dir.mkdir(parents=True)

    # 1. HPOBENCH_DATA_DIR set
    monkeypatch.setenv("HPOBENCH_DATA_DIR", str(cluster_hpobench_dir))
    monkeypatch.delenv("CARPS_TASK_DATA_DIR", raising=False)
    resolved = resolve_hpobench_data_dir()
    assert resolved == cluster_hpobench_dir

    # 2. CARPS_TASK_DATA_DIR set pointing to parent dir containing hpobench
    monkeypatch.delenv("HPOBENCH_DATA_DIR", raising=False)
    monkeypatch.setenv("CARPS_TASK_DATA_DIR", str(cluster_carps_dir))
    resolved = resolve_hpobench_data_dir()
    assert resolved == cluster_hpobench_dir

    # 3. Explicit override parameter takes priority
    custom_dir = tmp_path / "custom_data"
    custom_dir.mkdir()
    resolved = resolve_hpobench_data_dir(data_dir=custom_dir)
    assert resolved == custom_dir


def test_check_dataset_files_available_mock(tmp_path):
    """Verify check_dataset_files_available correctly detects present and missing datasets."""
    # Create empty dir - all datasets should be missing
    res_empty = check_dataset_files_available(suite="all", data_dir=tmp_path)
    assert res_empty["all_available"] is False
    assert len(res_empty["missing"]) > 0

    # Populate Tabular ML datasets: lr, nn, rf, svm, xgb
    tabular_dir = tmp_path / "TabularData"
    tabular_dir.mkdir()
    for model in ["lr", "nn", "rf", "svm", "xgb"]:
        model_dir = tabular_dir / model
        model_dir.mkdir()
        # Create a dummy zip or task file
        (model_dir / f"{model}.zip").write_text("dummy zip content")

    # Now ML should be available
    res_ml = check_dataset_files_available(suite="ml", data_dir=tmp_path)
    assert res_ml["all_available"] is True
    assert len(res_ml["missing"]) == 0
    for model in ["lr", "nn", "rf", "svm", "xgb"]:
        assert res_ml["ml_datasets"][model]["available"] is True

    # Populate NAS datasets: nasbench_201, nasbench_101, fcnet
    nas201_dir = tmp_path / "nasbench_201"
    nas201_dir.mkdir()
    (nas201_dir / "NAS-Bench-201-v1_1-096897_cifar100.json").write_text("{}")
    (nas201_dir / "NAS-Bench-201-v1_1-096897_cifar10-valid.json").write_text("{}")
    (nas201_dir / "NAS-Bench-201-v1_1-096897_ImageNet16-120.json").write_text("{}")

    nas101_dir = tmp_path / "nasbench_101"
    nas101_dir.mkdir()
    (nas101_dir / "nasbench_full.tfrecord").write_text("dummy tfrecord")

    fcnet_dir = tmp_path / "fcnet_tabular_benchmarks"
    fcnet_dir.mkdir()
    for bench in ["naval_propulsion", "parkinsons_telemonitoring", "protein_structure", "slice_localization"]:
        (fcnet_dir / f"fcnet_{bench}_data.hdf5").write_text("dummy hdf5")

    # Now NAS and all should be available
    res_nas = check_dataset_files_available(suite="nas", data_dir=tmp_path)
    assert res_nas["all_available"] is True
    assert len(res_nas["missing"]) == 0

    res_all = check_dataset_files_available(suite="all", data_dir=tmp_path)
    assert res_all["all_available"] is True
    assert len(res_all["missing"]) == 0


def test_check_sweep_completion_status(tmp_path):
    """Verify check_sweep_completion_status compares manifests against run directories."""
    runs_dir = tmp_path / "runs" / "sweep_hpobench_ml"
    results_dir = tmp_path / "results" / "sweep_hpobench_ml"
    runs_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    # Create 3 tasks in manifest
    task_lines = [
        "--config-dir carps_integration/configs +optimizer=smac3_hpo_facade_lcb +task=HPOBench/blackbox/tabular/ml/cfg_ml_lr_10101 task.optimization_resources.n_trials=10 seed=1 baserundir="
        + str(runs_dir)
        + " optimizer_id=SMAC3_HPOFacade_lcb",
        "--config-dir carps_integration/configs +optimizer=smac3_hpo_facade_lcb +task=HPOBench/blackbox/tabular/ml/cfg_ml_lr_12 task.optimization_resources.n_trials=10 seed=1 baserundir="
        + str(runs_dir)
        + " optimizer_id=SMAC3_HPOFacade_lcb",
        "--config-dir carps_integration/configs +optimizer=smac3_hpo_facade_lcb +task=HPOBench/blackbox/tabular/ml/cfg_ml_rf_10101 task.optimization_resources.n_trials=10 seed=1 baserundir="
        + str(runs_dir)
        + " optimizer_id=SMAC3_HPOFacade_lcb",
    ]
    manifest_file = results_dir / "tasks.txt"
    manifest_file.write_text("\n".join(task_lines) + "\n")

    # Create run logs for first task (completed with 10 trials)
    run_1 = runs_dir / "SMAC3_HPOFacade_lcb" / "hpobench" / "blackbox" / "tabular" / "ml" / "lr" / "10101" / "1"
    run_1.mkdir(parents=True)
    with open(run_1 / "trial_logs.jsonl", "w") as f:
        for i in range(10):
            f.write(json.dumps({"trial": i + 1, "cost": 0.1 * i}) + "\n")

    # Create telemetry fallback for second task (completed with 10 trials)
    telem_file = results_dir / "telemetry_SMAC3_HPOFacade_lcb_cfg_ml_lr_12_seed1.json"
    telem_file.write_text(json.dumps({"history": [{"trial": i} for i in range(10)]}))
    # Update line 2 in manifest with telemetry path
    task_lines[1] += f" ++optimizer.telemetry_path={telem_file}"
    manifest_file.write_text("\n".join(task_lines) + "\n")

    # Third task has no run dir and no telemetry -> incomplete

    status = check_sweep_completion_status(
        suite="ml",
        base_dir=runs_dir,
        manifest_path=manifest_file,
    )

    assert status["total_tasks"] == 3
    assert status["completed_tasks"] == 2
    assert status["missing_tasks"] == 1
    assert pytest.approx(status["completion_rate"], 0.01) == 2 / 3
    assert len(status["missing_task_lines"]) == 1


def test_cli_execution_and_formatting(tmp_path):
    """Verify CLI interface runs without error and formats verification output."""
    python_bin = sys.executable
    script_path = PROJECT_ROOT / "scripts" / "verify_hpobench_tasks_available.py"

    cmd = [
        python_bin,
        str(script_path),
        "--suite",
        "ml",
        "--data-dir",
        str(tmp_path),
        "--format",
        "json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert "tasks" in data
    assert "datasets" in data
    assert data["tasks"]["total_found"] == 88


def test_bash_script_wrapper():
    """Verify scripts/verify_hpobench_tasks_available.sh exists and is executable."""
    sh_path = PROJECT_ROOT / "scripts" / "verify_hpobench_tasks_available.sh"
    assert sh_path.exists(), f"Shell script not found: {sh_path}"
    assert os.access(sh_path, os.X_OK), f"Shell script is not executable: {sh_path}"

    res = subprocess.run([str(sh_path), "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "verify_hpobench_tasks_available" in res.stdout or "usage:" in res.stdout.lower()
