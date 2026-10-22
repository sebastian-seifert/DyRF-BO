#!/usr/bin/env python3
"""Tests for scripts/run_batch_subset_reports.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_batch_subset_reports import (
    DEFAULT_OPTIMIZERS,
    DEFAULT_SUBFOLDERS,
    build_command,
    run_batch_subset_reports,
    main,
)


def test_defaults():
    assert len(DEFAULT_SUBFOLDERS) == 7
    assert "sweep_bbob_dim2" in DEFAULT_SUBFOLDERS
    assert "sweep_bbob_dim4" in DEFAULT_SUBFOLDERS
    assert "sweep_bbob_dim8" in DEFAULT_SUBFOLDERS
    assert "sweep_bbob_dim16" in DEFAULT_SUBFOLDERS
    assert "sweep_bbob_dim32" in DEFAULT_SUBFOLDERS
    assert "sweep_yahpo_rbv2_ranger" in DEFAULT_SUBFOLDERS
    assert "sweep_yahpo_rbv2_super" in DEFAULT_SUBFOLDERS

    assert DEFAULT_OPTIMIZERS == [
        "SMAC3_HPOFacade_lcb",
        "SMAC3_HPOFacade_ei",
        "SMAC20_ProximityAC_LCB",
        "SMAC20_ProximityBC_LCB",
        "SMAC20_Entropy_LCB",
    ]


def test_build_command():
    cmd = build_command(
        parquet_path=Path("results/sweep_bbob_dim2/logs.parquet"),
        optimizers=["SMAC3_HPOFacade_lcb", "SMAC20_Entropy_LCB"],
        report_name="test_report",
        python_bin="python",
        extra_args=["--no-normalize"],
    )
    expected = [
        "python",
        str(Path("scripts/generate_subset_report.py")),
        "results/sweep_bbob_dim2/logs.parquet",
        "--optimizers",
        "SMAC3_HPOFacade_lcb",
        "SMAC20_Entropy_LCB",
        "--report-name",
        "test_report",
        "--no-normalize",
    ]
    assert cmd == expected


def test_run_batch_subset_reports_dry_run(tmp_path, capsys):
    # Setup dummy subfolder with logs.parquet
    sub1 = tmp_path / "sweep_bbob_dim2"
    sub1.mkdir(parents=True)
    parquet1 = sub1 / "logs.parquet"
    parquet1.touch()

    results = run_batch_subset_reports(
        subfolders=["sweep_bbob_dim2"],
        results_dir=tmp_path,
        optimizers=["SMAC3_HPOFacade_lcb"],
        report_name="subset_report",
        dry_run=True,
    )

    captured = capsys.readouterr()
    assert "[DRY-RUN]" in captured.out
    assert "results/sweep_bbob_dim2/logs.parquet" in captured.out or str(parquet1) in captured.out
    assert results["sweep_bbob_dim2"] == "DRY_RUN"


def test_run_batch_subset_reports_missing_file_skipped(tmp_path, capsys):
    # No logs.parquet exists in tmp_path / sweep_bbob_dim2
    results = run_batch_subset_reports(
        subfolders=["sweep_bbob_dim2"],
        results_dir=tmp_path,
        optimizers=["SMAC3_HPOFacade_lcb"],
        report_name="subset_report",
        skip_missing=True,
        dry_run=False,
    )
    captured = capsys.readouterr()
    assert "Skipping" in captured.out or "WARNING" in captured.out
    assert results["sweep_bbob_dim2"] == "SKIPPED_MISSING"


def test_run_batch_subset_reports_missing_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_batch_subset_reports(
            subfolders=["sweep_bbob_dim2"],
            results_dir=tmp_path,
            optimizers=["SMAC3_HPOFacade_lcb"],
            report_name="subset_report",
            skip_missing=False,
            dry_run=False,
        )


@patch("subprocess.run")
def test_run_batch_subset_reports_execution_success(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)

    sub1 = tmp_path / "sweep_bbob_dim2"
    sub1.mkdir(parents=True)
    parquet1 = sub1 / "logs.parquet"
    parquet1.touch()

    sub2 = tmp_path / "sweep_yahpo_rbv2_super"
    sub2.mkdir(parents=True)
    parquet2 = sub2 / "logs.parquet"
    parquet2.touch()

    results = run_batch_subset_reports(
        subfolders=["sweep_bbob_dim2", "sweep_yahpo_rbv2_super"],
        results_dir=tmp_path,
        optimizers=["SMAC3_HPOFacade_lcb", "SMAC20_Entropy_LCB"],
        report_name="subset_report",
        dry_run=False,
    )

    assert mock_run.call_count == 2
    assert results["sweep_bbob_dim2"] == "SUCCESS"
    assert results["sweep_yahpo_rbv2_super"] == "SUCCESS"


def test_main_cli_dry_run(monkeypatch, capsys):
    test_args = [
        "scripts/run_batch_subset_reports.py",
        "--dry-run",
        "--subfolders",
        "sweep_bbob_dim2",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    main()
    captured = capsys.readouterr()
    assert "[DRY-RUN]" in captured.out or "[WARNING] Missing" in captured.out

