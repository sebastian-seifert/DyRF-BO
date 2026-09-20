import json
import os
import shutil
import tempfile
from pathlib import Path
import pytest

from scripts.audit_sweep_batches import (
    parse_task_line,
    evaluate_task_status,
    audit_batch,
    generate_rerun_files,
    DEFAULT_BATCHES,
)


def test_parse_task_line():
    line = (
        "--config-dir carps_integration/configs +optimizer=smac20_proximity_lcb "
        "+task=YAHPO/blackbox/cfg_rbv2_ranger_1040 task.optimization_resources.n_trials=100 "
        "seed=42 baserundir=runs/sweep_yahpo_rbv2_ranger_proximity optimizer_id=SMAC20_ProximityLCB"
    )
    parsed = parse_task_line(line)
    assert parsed["optimizer_id"] == "SMAC20_ProximityLCB"
    assert parsed["seed"] == 42
    assert parsed["n_trials"] == 100
    assert parsed["baserundir"] == "runs/sweep_yahpo_rbv2_ranger_proximity"
    assert "cfg_rbv2_ranger_1040" in parsed["task"]


def test_evaluate_task_status_missing(tmp_path):
    parsed = {
        "optimizer_id": "SMAC20_ProximityLCB",
        "task": "cfg_rbv2_ranger_1040",
        "seed": 1,
        "n_trials": 100,
        "baserundir": str(tmp_path / "runs"),
    }
    status, n_found = evaluate_task_status(parsed, tmp_path / "runs")
    assert status == "MISSING"
    assert n_found == 0


def test_evaluate_task_status_truncated(tmp_path):
    run_dir = tmp_path / "runs" / "SMAC20_ProximityLCB" / "YAHPO" / "cfg_rbv2_ranger_1040" / "1"
    run_dir.mkdir(parents=True, exist_ok=True)
    trial_log = run_dir / "trial_logs.jsonl"
    with open(trial_log, "w") as f:
        for i in range(15):
            f.write(json.dumps({"trial": i, "cost": 0.5}) + "\n")

    parsed = {
        "optimizer_id": "SMAC20_ProximityLCB",
        "task": "cfg_rbv2_ranger_1040",
        "seed": 1,
        "n_trials": 100,
        "baserundir": str(tmp_path / "runs"),
    }
    status, n_found = evaluate_task_status(parsed, tmp_path / "runs")
    assert status == "TRUNCATED"
    assert n_found == 15


def test_evaluate_task_status_complete(tmp_path):
    run_dir = tmp_path / "runs" / "SMAC20_ProximityLCB" / "YAHPO" / "cfg_rbv2_ranger_1040" / "1"
    run_dir.mkdir(parents=True, exist_ok=True)
    trial_log = run_dir / "trial_logs.jsonl"
    with open(trial_log, "w") as f:
        for i in range(100):
            f.write(json.dumps({"trial": i, "cost": 0.2}) + "\n")

    parsed = {
        "optimizer_id": "SMAC20_ProximityLCB",
        "task": "cfg_rbv2_ranger_1040",
        "seed": 1,
        "n_trials": 100,
        "baserundir": str(tmp_path / "runs"),
    }
    status, n_found = evaluate_task_status(parsed, tmp_path / "runs")
    assert status == "COMPLETE"
    assert n_found == 100


def test_audit_batch_and_rerun_generation(tmp_path):
    runs_dir = tmp_path / "runs" / "sweep_test_suite_proximity"
    results_dir = tmp_path / "results" / "sweep_test_suite_proximity"
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        f"+optimizer=opt +task=task_a seed={i} task.optimization_resources.n_trials=100 "
        f"baserundir={runs_dir} optimizer_id=opt"
        for i in range(1, 6)
    ]
    task_file = results_dir / "tasks.txt"
    task_file.write_text("\n".join(tasks) + "\n")

    # Make task 1 complete (100 trials)
    run_1 = runs_dir / "opt" / "task_a" / "1"
    run_1.mkdir(parents=True, exist_ok=True)
    (run_1 / "trial_logs.jsonl").write_text("\n".join(json.dumps({"t": j}) for j in range(100)) + "\n")

    # Make task 2 truncated (20 trials)
    run_2 = runs_dir / "opt" / "task_a" / "2"
    run_2.mkdir(parents=True, exist_ok=True)
    (run_2 / "trial_logs.jsonl").write_text("\n".join(json.dumps({"t": j}) for j in range(20)) + "\n")

    # Tasks 3, 4, 5 are missing

    batch_def = {
        "batch_id": 1,
        "suite": "test_suite",
        "start": 1,
        "end": 5,
    }

    report = audit_batch(batch_def, results_base=tmp_path / "results", runs_base=tmp_path / "runs")
    assert report["total"] == 5
    assert report["complete"] == 1
    assert report["truncated"] == 1
    assert report["missing"] == 3
    assert len(report["failed_tasks"]) == 4  # task 2 (truncated) + 3,4,5 (missing)

    rerun_files = generate_rerun_files(report, results_base=tmp_path / "results")
    assert rerun_files["rerun_tasks_file"].exists()
    rerun_lines = rerun_files["rerun_tasks_file"].read_text().strip().split("\n")
    assert len(rerun_lines) == 4
    assert "seed=2" in rerun_lines[0]
