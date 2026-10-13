#!/usr/bin/env python3
"""Tests for identifying incomplete tasks from runs directory and task manifests."""

import json
import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.identify_incomplete_tasks import (
    TaskSpec,
    parse_task_line,
    is_task_complete,
    find_incomplete_tasks,
    filter_manifest,
)


def test_parse_task_line():
    """Verify parse_task_line extracts all metadata correctly from a task invocation line."""
    line = (
        "--config-dir carps_integration/configs +optimizer=smac3_hpo_facade_lcb "
        "+task=BBOB/cfg_2_1_0 task.optimization_resources.n_trials=100 seed=1 "
        "++optimizer.telemetry_path=results/sweep_bbob_full_comparison/telemetry_SMAC3_HPOFacade_lcb_cfg_2_1_0_seed1.json "
        "baserundir=runs/sweep_bbob_full_comparison optimizer_id=SMAC3_HPOFacade_lcb "
        "optimizer_container_id=SMAC3_HPOFacade_lcb"
    )
    spec = parse_task_line(line, line_number=1)
    assert spec is not None
    assert spec.line_number == 1
    assert spec.optimizer_id == "SMAC3_HPOFacade_lcb"
    assert spec.task_id == "BBOB/cfg_2_1_0"
    assert spec.dimension == 2
    assert spec.seed == 1
    assert spec.n_trials == 100
    assert spec.telemetry_path == Path("results/sweep_bbob_full_comparison/telemetry_SMAC3_HPOFacade_lcb_cfg_2_1_0_seed1.json")
    assert spec.baserundir == Path("runs/sweep_bbob_full_comparison")
    assert spec.expected_run_dir == Path("runs/sweep_bbob_full_comparison/SMAC3_HPOFacade_lcb/BBOB/bbob/2/1/0/1")


def test_audit_tasks_by_dimension(tmp_path):
    """Verify audit_tasks_by_dimension calculates per-dimension statistics correctly."""
    from scripts.identify_incomplete_tasks import audit_tasks_by_dimension

    # Create run dir for dim 2 task 0 (complete)
    run_dir2 = tmp_path / "runs/SMAC3_HPOFacade_lcb/BBOB/bbob/2/1/0/1"
    run_dir2.mkdir(parents=True)
    with open(run_dir2 / "trial_logs.jsonl", "w") as f:
        for i in range(100):
            f.write(json.dumps({"trial": i + 1}) + "\n")

    # Tasks for dim 2, 4, 8, 16, 32
    lines = [
        f"+task=BBOB/cfg_2_1_0 task.optimization_resources.n_trials=100 seed=1 baserundir={tmp_path}/runs optimizer_id=SMAC3_HPOFacade_lcb",
        f"+task=BBOB/cfg_2_1_1 task.optimization_resources.n_trials=100 seed=1 baserundir={tmp_path}/runs optimizer_id=SMAC3_HPOFacade_lcb",
        f"+task=BBOB/cfg_4_1_0 task.optimization_resources.n_trials=100 seed=1 baserundir={tmp_path}/runs optimizer_id=SMAC3_HPOFacade_lcb",
        f"+task=BBOB/cfg_32_1_0 task.optimization_resources.n_trials=100 seed=1 baserundir={tmp_path}/runs optimizer_id=SMAC3_HPOFacade_lcb",
    ]
    manifest = tmp_path / "tasks.txt"
    with open(manifest, "w") as f:
        f.write("\n".join(lines) + "\n")

    summary = audit_tasks_by_dimension(manifest)
    assert 2 in summary
    assert 4 in summary
    assert 32 in summary

    # Dim 2 has 2 total: 1 complete, 1 incomplete (50%)
    assert summary[2]["total"] == 2
    assert summary[2]["completed"] == 1
    assert summary[2]["incomplete"] == 1
    assert summary[2]["completion_rate"] == 50.0

    # Dim 4 has 1 total: 0 complete, 1 incomplete (0%)
    assert summary[4]["total"] == 1
    assert summary[4]["completed"] == 0
    assert summary[4]["incomplete"] == 1
    assert summary[4]["completion_rate"] == 0.0

    # Dim 32 has 1 total: 0 complete, 1 incomplete (0%)
    assert summary[32]["total"] == 1
    assert summary[32]["completed"] == 0
    assert summary[32]["incomplete"] == 1
    assert summary[32]["completion_rate"] == 0.0


def test_is_task_complete_when_trial_logs_completed(tmp_path):
    """Verify task is marked complete when trial_logs.jsonl contains expected number of trials."""
    run_dir = tmp_path / "runs/sweep_bbob_full_comparison/SMAC3_HPOFacade_lcb/BBOB/bbob/2/1/0/1"
    run_dir.mkdir(parents=True)
    trial_logs = run_dir / "trial_logs.jsonl"
    with open(trial_logs, "w") as f:
        for i in range(100):
            f.write(json.dumps({"trial": i + 1, "cost": 0.5}) + "\n")

    line = (
        "--config-dir carps_integration/configs +optimizer=smac3_hpo_facade_lcb "
        "+task=BBOB/cfg_2_1_0 task.optimization_resources.n_trials=100 seed=1 "
        f"baserundir={tmp_path}/runs/sweep_bbob_full_comparison optimizer_id=SMAC3_HPOFacade_lcb"
    )
    spec = parse_task_line(line, line_number=1)
    assert is_task_complete(spec) is True


def test_is_task_complete_when_telemetry_completed(tmp_path):
    """Verify task is marked complete when telemetry JSON exists and has >= n_trials."""
    telem_path = tmp_path / "telemetry.json"
    data = {
        "optimizer_id": "SMAC3_HPOFacade_lcb",
        "task_id": "cfg_2_1_0",
        "seed": 1,
        "history": [{"trial": i + 1, "cost": 0.1} for i in range(100)],
    }
    with open(telem_path, "w") as f:
        json.dump(data, f)

    line = (
        f"--config-dir carps_integration/configs +optimizer=smac3_hpo_facade_lcb "
        f"+task=BBOB/cfg_2_1_0 task.optimization_resources.n_trials=100 seed=1 "
        f"++optimizer.telemetry_path={telem_path} "
        f"baserundir={tmp_path}/non_existent optimizer_id=SMAC3_HPOFacade_lcb"
    )
    spec = parse_task_line(line, line_number=1)
    assert is_task_complete(spec) is True


def test_is_task_incomplete_when_run_dir_empty_or_partial(tmp_path):
    """Verify task is identified as incomplete when trial_logs has fewer trials than expected."""
    run_dir = tmp_path / "runs/sweep_bbob_full_comparison/SMAC3_HPOFacade_lcb/BBOB/bbob/2/1/0/1"
    run_dir.mkdir(parents=True)
    trial_logs = run_dir / "trial_logs.jsonl"
    # Write only 15 trials instead of 100
    with open(trial_logs, "w") as f:
        for i in range(15):
            f.write(json.dumps({"trial": i + 1, "cost": 0.5}) + "\n")

    line = (
        "--config-dir carps_integration/configs +optimizer=smac3_hpo_facade_lcb "
        "+task=BBOB/cfg_2_1_0 task.optimization_resources.n_trials=100 seed=1 "
        f"baserundir={tmp_path}/runs/sweep_bbob_full_comparison optimizer_id=SMAC3_HPOFacade_lcb"
    )
    spec = parse_task_line(line, line_number=1)
    assert is_task_complete(spec) is False


def test_find_incomplete_tasks_and_filter_manifest(tmp_path):
    """Verify manifest filtering produces the exact subset of incomplete tasks."""
    manifest = tmp_path / "tasks.txt"
    out_manifest = tmp_path / "missing_tasks.txt"

    # Task 1: Complete via trial_logs
    run_dir1 = tmp_path / "runs/SMAC3_HPOFacade_lcb/BBOB/bbob/2/1/0/1"
    run_dir1.mkdir(parents=True)
    with open(run_dir1 / "trial_logs.jsonl", "w") as f:
        for i in range(100):
            f.write(json.dumps({"trial": i + 1}) + "\n")

    # Task 2: Missing
    line1 = f"+task=BBOB/cfg_2_1_0 task.optimization_resources.n_trials=100 seed=1 baserundir={tmp_path}/runs optimizer_id=SMAC3_HPOFacade_lcb"
    line2 = f"+task=BBOB/cfg_2_1_1 task.optimization_resources.n_trials=100 seed=1 baserundir={tmp_path}/runs optimizer_id=SMAC3_HPOFacade_lcb"

    with open(manifest, "w") as f:
        f.write(f"{line1}\n{line2}\n")

    incomplete = find_incomplete_tasks(manifest)
    assert len(incomplete) == 1
    assert incomplete[0].line_number == 2
    assert incomplete[0].raw_line == line2

    filter_manifest(manifest, out_manifest)
    assert out_manifest.exists()
    with open(out_manifest) as f:
        written_lines = [l.strip() for l in f if l.strip()]
    assert written_lines == [line2]


def test_filter_manifest_by_dimension(tmp_path):
    """Verify filter_manifest filters by dimension when dimensions argument is supplied."""
    manifest = tmp_path / "tasks.txt"
    out_dim2 = tmp_path / "missing_dim2.txt"

    lines = [
        f"+task=BBOB/cfg_2_1_0 task.optimization_resources.n_trials=100 seed=1 baserundir={tmp_path}/runs optimizer_id=SMAC3_HPOFacade_lcb",
        f"+task=BBOB/cfg_4_1_0 task.optimization_resources.n_trials=100 seed=1 baserundir={tmp_path}/runs optimizer_id=SMAC3_HPOFacade_lcb",
        f"+task=BBOB/cfg_8_1_0 task.optimization_resources.n_trials=100 seed=1 baserundir={tmp_path}/runs optimizer_id=SMAC3_HPOFacade_lcb",
    ]
    with open(manifest, "w") as f:
        f.write("\n".join(lines) + "\n")

    filter_manifest(manifest, out_dim2, dimensions=[2])
    assert out_dim2.exists()
    with open(out_dim2) as f:
        written = [l.strip() for l in f if l.strip()]
    assert len(written) == 1
    assert "cfg_2_1_0" in written[0]

