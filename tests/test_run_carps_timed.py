#!/usr/bin/env python3
"""Tests for hardware-aware timed runner wrapper."""

import json
import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_carps_timed import (
    collect_hardware_metadata,
    compute_timing_metrics_from_trial_logs,
    parse_task_and_telemetry_info,
    execute_timed_run,
)


def test_collect_hardware_metadata():
    """Verify hardware metadata collection captures CPU, host, memory, and thread settings."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    hw = collect_hardware_metadata()

    assert "hostname" in hw
    assert "cpu_model" in hw
    assert "cpu_count" in hw
    assert "total_ram_gb" in hw
    assert "thread_env" in hw
    assert hw["thread_env"].get("OMP_NUM_THREADS") == "1"
    assert hw["thread_env"].get("MKL_NUM_THREADS") == "1"


def test_compute_timing_metrics_from_trial_logs(tmp_path):
    """Verify trial_logs parser calculates correct total target evaluation time and count."""
    log_file = tmp_path / "trial_logs.jsonl"
    trial_data = [
        {"n_trials": 1, "trial_value": {"time": 0.005, "cost": 1.2}},
        {"n_trials": 2, "trial_value": {"time": 0.010, "cost": 1.1}},
        {"n_trials": 3, "trial_value": {"time": 0.015, "cost": 0.9}},
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for t in trial_data:
            f.write(json.dumps(t) + "\n")

    metrics = compute_timing_metrics_from_trial_logs(str(log_file), total_wallclock_seconds=1.5)
    assert metrics["n_trials_logged"] == 3
    assert pytest.approx(metrics["target_eval_seconds"], rel=1e-3) == 0.030
    assert pytest.approx(metrics["bo_overhead_seconds"], rel=1e-3) == 1.470


def test_parse_task_and_telemetry_info():
    """Verify CLI argument extraction of telemetry path and identifiers."""
    args = [
        "--config-dir", "carps_integration/configs",
        "+optimizer=smac3_hpo_facade_lcb",
        "+task=BBOB/cfg_8_8_0",
        "task.optimization_resources.n_trials=100",
        "seed=7",
        "++optimizer.telemetry_path=results/sweep/telemetry_opt_cfg_8_8_0_seed7.json",
        "baserundir=runs/sweep",
        "optimizer_id=SMAC3_HPOFacade_lcb",
    ]
    info = parse_task_and_telemetry_info(args)
    assert info["optimizer_id"] == "SMAC3_HPOFacade_lcb"
    assert info["task"] == "BBOB/cfg_8_8_0"
    assert info["dimension"] == 8
    assert info["function_id"] == 8
    assert info["instance_id"] == 0
    assert info["seed"] == 7
    assert info["telemetry_path"] == "results/sweep/telemetry_opt_cfg_8_8_0_seed7.json"
    assert info["timing_path"] == "results/sweep/timing_opt_cfg_8_8_0_seed7.json"


def test_execute_timed_run_mock(tmp_path):
    """Verify timed execution runs command, captures timing, and saves timing JSON."""
    timing_file = tmp_path / "timing_test.json"
    args = [
        f"++optimizer.telemetry_path={tmp_path / 'telemetry.json'}",
        "optimizer_id=test_opt",
        "+task=BBOB/cfg_2_8_0",
        "seed=1",
    ]

    # Command that executes python -c 'import time; time.sleep(0.05)'
    res = execute_timed_run(
        args=args,
        cmd_runner=lambda a: 0,
        timing_output_path=str(timing_file),
    )

    assert res["exit_code"] == 0
    assert res["wallclock_seconds"] >= 0.0
    assert timing_file.exists()

    with open(timing_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["exit_code"] == 0
    assert "hardware" in data
    assert "wallclock_seconds" in data
