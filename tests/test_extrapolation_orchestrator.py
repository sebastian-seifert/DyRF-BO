"""Unit tests for the live interactive Extrapolation UQ Sweep Orchestrator."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_extrapolation_sweep_orchestrator import (
    build_parser,
    dispatch_stage,
    format_dashboard,
    get_disk_progress,
    parse_squeue_output,
    run_orchestrator,
)


class TestOrchestratorParser:
    """Test CLI argument parsing and defaults."""

    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])

        assert args.stage1_start == 1
        assert args.stage1_end == 4800
        assert args.stage2_start == 4801
        assert args.stage2_end == 9600
        assert args.poll_interval == 30
        assert args.aggregate is True
        assert args.dry_run is False
        assert "extrapolation_sweep_tasks.txt" in args.task_file

    def test_parser_custom_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "--stage1-start", "1",
            "--stage1-end", "200",
            "--stage2-start", "201",
            "--stage2-end", "400",
            "--poll-interval", "10",
            "--no-aggregate",
            "--dry-run",
            "--task-file", "results/test_tasks.txt",
        ])

        assert args.stage1_start == 1
        assert args.stage1_end == 200
        assert args.stage2_start == 201
        assert args.stage2_end == 400
        assert args.poll_interval == 10
        assert args.aggregate is False
        assert args.dry_run is True
        assert args.task_file == "results/test_tasks.txt"


class TestSqueueParsing:
    """Test parsing squeue output and filtering for worker array jobs."""

    def test_parse_squeue_mixed_jobs(self):
        # Simulating squeue -h -o "%i %T %j" output:
        # 12345678 RUNNING salloc (master job allocation)
        # 12345679_1 RUNNING extrap_uq
        # 12345679_2 RUNNING extrap_uq
        # 12345679_3 PENDING extrap_uq
        # 12345680_1 PENDING extrap_uq
        # 99999999 RUNNING bash (another shell)
        mock_output = (
            "12345678 RUNNING salloc\n"
            "12345679_1 RUNNING extrap_uq\n"
            "12345679_2 RUNNING extrap_uq\n"
            "12345679_3 PENDING extrap_uq\n"
            "12345680_1 PENDING extrap_uq\n"
            "99999999 RUNNING bash\n"
        )
        counts = parse_squeue_output(mock_output, target_job_name="extrap_uq")
        assert counts["running"] == 2
        assert counts["pending"] == 2
        assert counts["other"] == 0
        assert counts["total"] == 4

    def test_parse_squeue_empty_or_completed(self):
        mock_output = "12345678 RUNNING salloc\n"
        counts = parse_squeue_output(mock_output, target_job_name="extrap_uq")
        assert counts["running"] == 0
        assert counts["pending"] == 0
        assert counts["total"] == 0


class TestDiskProgress:
    """Test real-time counting of generated Parquet and summary JSON files."""

    def test_get_disk_progress(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        sum_dir = tmp_path / "summaries"
        raw_dir.mkdir(parents=True)
        sum_dir.mkdir(parents=True)

        # Create dummy parquet and json files
        for i in range(5):
            (raw_dir / f"extrapolation_sphere_d2_s{i}.parquet").write_text("dummy")
            (sum_dir / f"summary_sphere_d2_s{i}.json").write_text("{}")

        progress = get_disk_progress(raw_dir=raw_dir, summary_dir=sum_dir)
        assert progress["parquet_count"] == 5
        assert progress["summary_count"] == 5
        assert len(progress["recent_files"]) == 5


class TestDashboardFormatting:
    """Test rendering of the live terminal dashboard string."""

    def test_format_dashboard_contains_key_elements(self):
        status = {
            "mode": "salloc Interactive (Job ID: 12345678)",
            "stage": "STAGE 1 IN PROGRESS (Tasks 1 - 4,800)",
            "elapsed_str": "00h 15m 30s",
            "eta_str": "01h 45m 00s",
            "running": 25,
            "pending": 1800,
            "total_active": 1825,
            "parquet_count": 1200,
            "summary_count": 1200,
            "total_tasks": 9600,
            "stage_target": 4800,
            "recent_files": ["extrapolation_sphere_d2_s0.parquet", "extrapolation_sphere_d2_s1.parquet"],
        }
        text = format_dashboard(status, is_tty=True)
        assert "Extrapolation UQ Sweep Master Orchestrator" in text
        assert "STAGE 1 IN PROGRESS" in text
        assert "Running Tasks:" in text
        assert "Parquet Files:" in text
        assert "1200 / 9600" in text
        assert "extrapolation_sphere_d2_s0.parquet" in text


class TestDispatchStage:
    """Test invocation of submit_extrapolation_sweep_all.sh with range environment variables."""

    @patch("subprocess.run")
    def test_dispatch_stage_dry_run(self, mock_run):
        res = dispatch_stage(start_task=1, end_task=4800, dry_run=True)
        assert res is True
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_dispatch_stage_execution(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        res = dispatch_stage(start_task=1, end_task=4800, dry_run=False)
        assert res is True
        mock_run.assert_called_once()
        env_passed = mock_run.call_args[1]["env"]
        assert env_passed["START_TASK"] == "1"
        assert env_passed["END_TASK"] == "4800"
