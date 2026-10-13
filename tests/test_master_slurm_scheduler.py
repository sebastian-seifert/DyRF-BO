#!/usr/bin/env python3
"""Tests for Dynamic Master Slurm Scheduler."""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from scripts.master_slurm_scheduler import (
    MasterSlurmScheduler,
    SchedulerConfig,
    parse_squeue_me_output,
)


def test_parse_squeue_me_output():
    """Verify squeue parser correctly tallies remaining jobs and tasks."""
    # Mock squeue --me -h -o "%i %t" output
    sample_output = (
        "12345_1 R\n"
        "12345_2 R\n"
        "12345_3 PD\n"
        "12346_1 R\n"
        "12347 (None)\n"
    )
    job_count, task_count = parse_squeue_me_output(sample_output)
    # Distinct job IDs: 12345, 12346, 12347 -> 3 distinct jobs
    assert job_count == 3
    # Total tasks/lines listed -> 5 tasks
    assert task_count == 5

    # Empty queue
    job_count_empty, task_count_empty = parse_squeue_me_output("")
    assert job_count_empty == 0
    assert task_count_empty == 0


def test_chunk_generation():
    """Verify chunking handles large task manifests cleanly."""
    config = SchedulerConfig(
        task_manifest="mock_tasks.txt",
        chunk_size=200,
        max_active_tasks=4500,
        min_remaining_jobs_trigger=10,
    )
    scheduler = MasterSlurmScheduler(config=config)
    chunks = scheduler.calculate_chunks(total_tasks=1050)
    assert len(chunks) == 6
    assert chunks[0] == (1, 200)
    assert chunks[1] == (201, 400)
    assert chunks[4] == (801, 1000)
    assert chunks[5] == (1001, 1050)


def test_scheduling_trigger_and_task_limit(tmp_path):
    """Verify scheduler submits when remaining jobs < 10 AND active tasks < 4500."""
    manifest_file = tmp_path / "tasks.txt"
    with open(manifest_file, "w") as f:
        for i in range(1200):
            f.write(f"task_{i}\n")

    state_file = str(tmp_path / "scheduler_state.json")
    config = SchedulerConfig(
        task_manifest=str(manifest_file),
        chunk_size=200,
        max_active_tasks=4500,
        min_remaining_jobs_trigger=10,
        state_file=state_file,
    )
    scheduler = MasterSlurmScheduler(config=config)

    # Case 1: 12 jobs currently remaining in queue (> min_trigger 10) -> Should NOT submit
    submitted = scheduler.step(mock_remaining_jobs=12, mock_active_tasks=1200, dry_run=True)
    assert len(submitted) == 0

    # Case 2: Only 5 jobs remaining (< min_trigger 10) and active tasks 1000 (< 4500)
    # Available task capacity: 4500 - 1000 = 3500. Chunks are 200 each.
    # Total chunks needed: 6 chunks (1200 tasks).
    # Since all 6 chunks (1200 tasks) fit within capacity (1200 <= 3500), all 6 should be submitted!
    submitted = scheduler.step(mock_remaining_jobs=5, mock_active_tasks=1000, dry_run=True)
    assert len(submitted) == 6

    # Verify state persistence
    state = scheduler.load_state()
    assert state["completed_or_submitted_chunks"] == 6
    assert state["all_chunks_dispatched"] is True


def test_scheduling_respects_hard_task_cap(tmp_path):
    """Verify scheduler will not exceed max_active_tasks (4500)."""
    manifest_file = tmp_path / "tasks.txt"
    with open(manifest_file, "w") as f:
        for i in range(2000):
            f.write(f"task_{i}\n")

    state_file = str(tmp_path / "scheduler_state.json")
    config = SchedulerConfig(
        task_manifest=str(manifest_file),
        chunk_size=200,
        max_active_tasks=4500,
        min_remaining_jobs_trigger=10,
        state_file=state_file,
    )
    scheduler = MasterSlurmScheduler(config=config)

    # 4 jobs remaining (< 10), but already 4200 tasks active. Capacity = 300 tasks.
    # Chunks are 200 tasks each.
    # Can only submit 1 chunk (200 tasks), because 2 chunks (400 tasks) would exceed capacity!
    submitted = scheduler.step(mock_remaining_jobs=4, mock_active_tasks=4200, dry_run=True)
    assert len(submitted) == 1
    assert submitted[0] == (0, 1, 200)


def test_scheduler_auto_generates_manifest_when_missing(tmp_path):
    """Verify scheduler automatically generates manifest if it does not exist."""
    missing_manifest = tmp_path / "auto_tasks" / "tasks_all.txt"
    state_file = str(tmp_path / "scheduler_state.json")
    config = SchedulerConfig(
        task_manifest=str(missing_manifest),
        chunk_size=250,
        max_active_tasks=4500,
        min_remaining_jobs_trigger=10,
        state_file=state_file,
    )
    scheduler = MasterSlurmScheduler(config=config)

    # First check that missing_manifest does not exist
    assert not missing_manifest.exists()

    # Step should auto-generate manifest using generate_bbob_full_sweep_tasks
    submitted = scheduler.step(mock_remaining_jobs=0, mock_active_tasks=0, dry_run=True)
    assert missing_manifest.exists()
    assert len(submitted) > 0


def test_dim2_scheduler_config_and_cli(tmp_path):
    """Verify SchedulerConfig handles dim2 paths and CLI parses custom state_file and prefix."""
    dim2_manifest = tmp_path / "sweep_bbob_dim2" / "tasks.txt"
    dim2_manifest.parent.mkdir(parents=True)
    with open(dim2_manifest, "w") as f:
        for i in range(500):
            f.write(f"task_{i}\n")

    state_file = str(tmp_path / "custom_state.json")
    config = SchedulerConfig(
        task_manifest=str(dim2_manifest),
        sbatch_script="scripts/submit_bbob_dim2_sweep_array.sbatch",
        chunk_size=250,
        job_name_prefix="bbob_d2",
        state_file=state_file,
    )
    scheduler = MasterSlurmScheduler(config=config)
    chunks = scheduler.calculate_chunks(500)
    assert len(chunks) == 2
    assert chunks[0] == (1, 250)
    assert chunks[1] == (251, 500)

    # Test step execution
    submitted = scheduler.step(mock_remaining_jobs=0, mock_active_tasks=0, dry_run=True)
    assert len(submitted) == 2
    assert os.path.exists(state_file)
    with open(state_file) as f:
        state = json.load(f)
    assert state["completed_or_submitted_chunks"] == 2
    assert state["submitted_chunks_info"][0]["job_id"] == "mock_job_0"


def test_master_scheduler_cli_arguments(tmp_path):
    """Verify master_slurm_scheduler CLI accepts --state-file, --job-prefix, --concurrency."""
    import subprocess
    manifest = tmp_path / "tasks.txt"
    with open(manifest, "w") as f:
        for i in range(250):
            f.write(f"task_{i}\n")

    state_file = tmp_path / "cli_state.json"
    res = subprocess.run(
        [
            sys.executable,
            "scripts/master_slurm_scheduler.py",
            "--manifest", str(manifest),
            "--state-file", str(state_file),
            "--job-prefix", "test_pfx",
            "--concurrency", "10",
            "--chunk-size", "250",
            "--dry-run",
            "--once",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert res.returncode == 0
    assert state_file.exists()



