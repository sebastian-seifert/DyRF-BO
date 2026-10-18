#!/usr/bin/env python3
"""Unified Tests for YAHPO Gym rbv2_ranger and rbv2_super statistical sweep analysis."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.compute_yahpo_rbv2_ranger_sweep_analysis import (
    run_yahpo_rbv2_ranger_analysis,
    export_yahpo_rbv2_ranger_analysis,
    build_parser as build_ranger_parser,
)
from scripts.compute_yahpo_rbv2_super_sweep_analysis import (
    run_yahpo_rbv2_super_analysis,
    export_yahpo_rbv2_super_analysis,
    build_parser as build_super_parser,
)


@pytest.fixture
def synthetic_yahpo_records():
    """Generates synthetic multi-task logs with dual controls and superior/inferior methods."""
    records = []
    # 8 tasks, 4 optimizers:
    # Controls: SMAC3_HPOFacade_lcb, SMAC3_HPOFacade_ei
    # Superior: SMAC20_ProximityA_LCB
    # Inferior: SMAC20_Inferior_LCB
    tasks = [f"cfg_rbv2_suite_{i}" for i in range(1, 9)]
    opts = [
        "SMAC3_HPOFacade_lcb",
        "SMAC3_HPOFacade_ei",
        "SMAC20_ProximityA_LCB",
        "SMAC20_Inferior_LCB",
    ]
    for task_id in tasks:
        for opt in opts:
            for seed in [1, 2, 3]:
                for trial in [1, 10]:
                    if opt == "SMAC20_ProximityA_LCB":
                        cost = 0.20 if trial == 10 else 0.40
                    elif opt == "SMAC3_HPOFacade_ei":
                        cost = 0.40 if trial == 10 else 0.60
                    elif opt == "SMAC3_HPOFacade_lcb":
                        cost = 0.50 if trial == 10 else 0.70
                    else:
                        cost = 0.90 if trial == 10 else 1.00
                    records.append({
                        "task_id": task_id,
                        "optimizer_id": opt,
                        "seed": seed,
                        "trial": trial,
                        "trial_value__cost_inc": cost,
                    })
    return pd.DataFrame(records)


def test_schema_and_statistical_core_ranger(synthetic_yahpo_records):
    """Verify ranger analysis returns standardized BBOB schema, one-sided Wilcoxon, and effect sizes."""
    results = run_yahpo_rbv2_ranger_analysis(
        synthetic_yahpo_records,
        cost_col="trial_value__cost_inc",
        controls=["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"],
        holm_scope="joint",
    )

    # Acceptance criterion 1: schema keys
    assert "scorecard" in results
    assert "friedman_stat" in results
    assert "friedman_p" in results
    assert "raw_analysis" in results

    scorecard = results["scorecard"]
    assert isinstance(scorecard, pd.DataFrame)
    assert len(scorecard) == 4

    # Acceptance criterion 2: one-sided Wilcoxon, Cliff's delta, rank-biserial, Holm-Bonferroni
    required_cols = [
        "optimizer_id",
        "n_tasks",
        "n_runs",
        "mean_norm_regret",
        "median_norm_regret",
        "mean_raw_cost",
        "ctrl_lcb_win_rate",
        "ctrl_lcb_cliffs_delta",
        "ctrl_lcb_rank_biserial",
        "ctrl_lcb_wilcoxon_p",
        "ctrl_lcb_wilcoxon_adj_p",
        "ctrl_ei_win_rate",
        "ctrl_ei_cliffs_delta",
        "ctrl_ei_rank_biserial",
        "ctrl_ei_wilcoxon_p",
        "ctrl_ei_wilcoxon_adj_p",
    ]
    for col in required_cols:
        assert col in scorecard.columns, f"Missing expected column {col}"

    prox_row = scorecard[scorecard["optimizer_id"] == "SMAC20_ProximityA_LCB"].iloc[0]
    inferior_row = scorecard[scorecard["optimizer_id"] == "SMAC20_Inferior_LCB"].iloc[0]

    # ProximityA strictly better than controls
    assert prox_row["ctrl_lcb_wilcoxon_p"] < 0.05
    assert prox_row["ctrl_ei_wilcoxon_p"] < 0.05
    assert prox_row["ctrl_lcb_cliffs_delta"] == pytest.approx(-1.0)
    assert prox_row["ctrl_lcb_rank_biserial"] == pytest.approx(-1.0)
    assert prox_row["ctrl_lcb_win_rate"] == pytest.approx(1.0)

    # Inferior strictly worse
    assert inferior_row["ctrl_lcb_wilcoxon_p"] > 0.95
    assert inferior_row["ctrl_ei_wilcoxon_p"] > 0.95
    assert inferior_row["ctrl_lcb_cliffs_delta"] == pytest.approx(1.0)
    assert inferior_row["ctrl_lcb_rank_biserial"] == pytest.approx(1.0)


def test_schema_and_statistical_core_super(synthetic_yahpo_records):
    """Verify super analysis returns standardized BBOB schema, one-sided Wilcoxon, and effect sizes."""
    results = run_yahpo_rbv2_super_analysis(
        synthetic_yahpo_records,
        cost_col="trial_value__cost_inc",
        controls=["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"],
        holm_scope="joint",
    )

    # Acceptance criterion 1: schema keys
    assert "scorecard" in results
    assert "friedman_stat" in results
    assert "friedman_p" in results
    assert "raw_analysis" in results

    scorecard = results["scorecard"]
    assert isinstance(scorecard, pd.DataFrame)
    assert len(scorecard) == 4

    required_cols = [
        "optimizer_id",
        "ctrl_lcb_win_rate",
        "ctrl_lcb_cliffs_delta",
        "ctrl_lcb_rank_biserial",
        "ctrl_lcb_wilcoxon_p",
        "ctrl_lcb_wilcoxon_adj_p",
        "ctrl_ei_win_rate",
        "ctrl_ei_cliffs_delta",
        "ctrl_ei_rank_biserial",
        "ctrl_ei_wilcoxon_p",
        "ctrl_ei_wilcoxon_adj_p",
    ]
    for col in required_cols:
        assert col in scorecard.columns, f"Missing expected column {col}"

    prox_row = scorecard[scorecard["optimizer_id"] == "SMAC20_ProximityA_LCB"].iloc[0]
    assert prox_row["ctrl_lcb_wilcoxon_p"] < 0.05
    assert prox_row["ctrl_ei_wilcoxon_p"] < 0.05
    assert prox_row["ctrl_lcb_cliffs_delta"] == pytest.approx(-1.0)
    assert prox_row["ctrl_lcb_rank_biserial"] == pytest.approx(-1.0)


def test_holm_scope_options(synthetic_yahpo_records):
    """Verify both scripts support both 'joint' and 'per_control' holm_scope options."""
    res_joint = run_yahpo_rbv2_ranger_analysis(synthetic_yahpo_records, holm_scope="joint")
    res_per = run_yahpo_rbv2_ranger_analysis(synthetic_yahpo_records, holm_scope="per_control")
    assert "ctrl_lcb_wilcoxon_adj_p" in res_joint["scorecard"].columns
    assert "ctrl_lcb_wilcoxon_adj_p" in res_per["scorecard"].columns

    res_super_joint = run_yahpo_rbv2_super_analysis(synthetic_yahpo_records, holm_scope="joint")
    res_super_per = run_yahpo_rbv2_super_analysis(synthetic_yahpo_records, holm_scope="per_control")
    assert "ctrl_lcb_wilcoxon_adj_p" in res_super_joint["scorecard"].columns
    assert "ctrl_lcb_wilcoxon_adj_p" in res_super_per["scorecard"].columns


def test_export_ranger_artifacts(synthetic_yahpo_records, tmp_path):
    """Verify export_yahpo_rbv2_ranger_analysis writes both suite-specific and global scorecards & reports."""
    results = run_yahpo_rbv2_ranger_analysis(synthetic_yahpo_records)
    out_dir = tmp_path / "ranger_analysis"
    export_yahpo_rbv2_ranger_analysis(results, out_dir)

    assert (out_dir / "yahpo_rbv2_ranger_scorecard.csv").exists()
    assert (out_dir / "global_scorecard.csv").exists()
    assert (out_dir / "yahpo_rbv2_ranger_analysis_report.md").exists()
    assert (out_dir / "yahpo_rbv2_ranger_scorecard.md").exists()

    df_suite = pd.read_csv(out_dir / "yahpo_rbv2_ranger_scorecard.csv")
    df_global = pd.read_csv(out_dir / "global_scorecard.csv")
    assert len(df_suite) == len(df_global) == 4

    content_report = (out_dir / "yahpo_rbv2_ranger_analysis_report.md").read_text()
    assert "Omnibus Statistical Test (Demšar)" in content_report
    assert "Method Comparison Scorecard" in content_report


def test_export_super_artifacts(synthetic_yahpo_records, tmp_path):
    """Verify export_yahpo_rbv2_super_analysis writes both suite-specific and global scorecards & reports."""
    results = run_yahpo_rbv2_super_analysis(synthetic_yahpo_records)
    out_dir = tmp_path / "super_analysis"
    export_yahpo_rbv2_super_analysis(results, out_dir)

    assert (out_dir / "yahpo_rbv2_super_scorecard.csv").exists()
    assert (out_dir / "global_scorecard.csv").exists()
    assert (out_dir / "yahpo_rbv2_super_analysis_report.md").exists()
    assert (out_dir / "yahpo_rbv2_super_scorecard.md").exists()

    df_suite = pd.read_csv(out_dir / "yahpo_rbv2_super_scorecard.csv")
    df_global = pd.read_csv(out_dir / "global_scorecard.csv")
    assert len(df_suite) == len(df_global) == 4

    content_report = (out_dir / "yahpo_rbv2_super_analysis_report.md").read_text()
    assert "Omnibus Statistical Test (Demšar)" in content_report
    assert "Method Comparison Scorecard" in content_report


@pytest.mark.parametrize("build_parser_fn", [build_ranger_parser, build_super_parser])
def test_cli_argument_parity(build_parser_fn):
    """Verify identical CLI options across both scripts: --input/--data-path, --output-dir, --cost-col, --holm-scope, --controls."""
    parser = build_parser_fn()
    # Check parsing --input
    args1 = parser.parse_args(["--input", "some/path.parquet", "--output-dir", "out/dir", "--cost-col", "my_cost", "--holm-scope", "per_control", "--controls", "c1", "c2"])
    assert args1.input == "some/path.parquet"
    assert args1.output_dir == "out/dir"
    assert args1.cost_col == "my_cost"
    assert args1.holm_scope == "per_control"
    assert args1.controls == ["c1", "c2"]

    # Check --data-path backward-compatibility alias
    args2 = parser.parse_args(["--data-path", "other/path.csv"])
    assert args2.input == "other/path.csv"


def test_cli_execution_e2e(synthetic_yahpo_records, tmp_path):
    """Verify end-to-end execution of both scripts via CLI with --input and --data-path."""
    csv_path = tmp_path / "sample_logs.csv"
    synthetic_yahpo_records.to_csv(csv_path, index=False)

    # 1. Ranger with --input
    out_ranger = tmp_path / "cli_out_ranger"
    cmd_ranger = [
        sys.executable,
        str(Path(PROJECT_ROOT) / "scripts/compute_yahpo_rbv2_ranger_sweep_analysis.py"),
        "--input", str(csv_path),
        "--output-dir", str(out_ranger),
        "--holm-scope", "joint",
    ]
    res_r = subprocess.run(cmd_ranger, capture_output=True, text=True)
    assert res_r.returncode == 0, f"Ranger CLI failed: {res_r.stderr}"
    assert (out_ranger / "yahpo_rbv2_ranger_scorecard.csv").exists()
    assert (out_ranger / "global_scorecard.csv").exists()

    # 2. Super with --data-path alias
    out_super = tmp_path / "cli_out_super"
    cmd_super = [
        sys.executable,
        str(Path(PROJECT_ROOT) / "scripts/compute_yahpo_rbv2_super_sweep_analysis.py"),
        "--data-path", str(csv_path),
        "--output-dir", str(out_super),
        "--holm-scope", "joint",
    ]
    res_s = subprocess.run(cmd_super, capture_output=True, text=True)
    assert res_s.returncode == 0, f"Super CLI failed: {res_s.stderr}"
    assert (out_super / "yahpo_rbv2_super_scorecard.csv").exists()
    assert (out_super / "global_scorecard.csv").exists()
