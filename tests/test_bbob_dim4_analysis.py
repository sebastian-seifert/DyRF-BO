#!/usr/bin/env python3
"""Tests for dedicated BBOB Dimension 4 data gathering and analysis scripts."""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.gather_bbob_dim4_sweep import gather_bbob_dim4_trial_logs
from scripts.compute_bbob_dim4_sweep_analysis import run_bbob_dim4_analysis, export_dim4_analysis


def test_gather_bbob_dim4_trial_logs(tmp_path):
    """Verify gather_bbob_dim4_trial_logs gathers trial logs and filters to dimension 4."""
    runs_dir = tmp_path / "runs/sweep_bbob_dim4"
    out_dir = tmp_path / "results/sweep_bbob_dim4"

    # Create dummy trial logs for dim 4 task
    run_path = runs_dir / "SMAC3_HPOFacade_lcb/BBOB/bbob/4/1/0/1"
    run_path.mkdir(parents=True)
    with open(run_path / "trial_logs.jsonl", "w") as f:
        f.write(json.dumps({"trial": 1, "trial_value": {"cost": 10.0}}) + "\n")
        f.write(json.dumps({"trial": 2, "trial_value": {"cost": 5.0}}) + "\n")

    df = gather_bbob_dim4_trial_logs(runs_base=runs_dir, outdir=out_dir)
    assert len(df) == 2
    assert "dimension" in df.columns
    assert (df["dimension"] == 4).all()
    assert (out_dir / "logs.csv").exists()
    try:
        import pyarrow
        assert (out_dir / "logs.parquet").exists()
    except ImportError:
        pass


def test_compute_bbob_dim4_analysis(tmp_path):
    """Verify statistical scorecard and Demšar pipeline on synthetic dim 4 logs."""
    records = []
    # 2 tasks, 2 optimizers (Control LCB + Prox A), 3 seeds, 2 trials
    for task in ["bbob/4/1/0", "bbob/4/2/0"]:
        for opt, base_cost in [("SMAC3_HPOFacade_lcb", 10.0), ("SMAC20_ProximityA_LCB", 8.0)]:
            for seed in [1, 2, 3]:
                for trial, c in [(1, base_cost), (2, base_cost * 0.8)]:
                    records.append({
                        "task_id": task,
                        "optimizer_id": opt,
                        "seed": seed,
                        "trial": trial,
                        "cost": c,
                        "trial_value__cost_inc": c,
                        "dimension": 4,
                        "fid": 1 if "1" in task else 2,
                        "iid": 0,
                        "function_name": "Sphere",
                        "function_category": "separable",
                    })
    df = pd.DataFrame(records)
    res = run_bbob_dim4_analysis(df)
    assert "scorecard" in res
    assert "friedman_stat" in res
    assert len(res["scorecard"]) == 2

    out_dir = str(tmp_path / "analysis")
    export_dim4_analysis(res, out_dir)
    assert (Path(out_dir) / "bbob_dim4_scorecard.csv").exists()
    assert (Path(out_dir) / "bbob_dim4_analysis_report.md").exists()


def test_compute_bbob_dim4_cli_csv_fallback(tmp_path):
    """Verify CLI executes cleanly when only CSV is available and default .parquet was passed."""
    import subprocess
    csv_file = tmp_path / "logs.csv"
    records = []
    for opt, base_cost in [("SMAC3_HPOFacade_lcb", 10.0), ("SMAC20_ProximityA_LCB", 8.0)]:
        for seed in [1, 2]:
            records.append({
                "task_id": "bbob/4/1/0",
                "optimizer_id": opt,
                "seed": seed,
                "trial": 1,
                "cost": base_cost,
                "trial_value__cost_inc": base_cost,
                "dimension": 4,
            })
    pd.DataFrame(records).to_csv(csv_file, index=False)
    out_dir = tmp_path / "analysis_cli"

    # Pass the parquet path that doesn't exist, CLI should fallback to .csv
    parquet_arg = str(tmp_path / "logs.parquet")
    res = subprocess.run(
        [
            sys.executable,
            "scripts/compute_bbob_dim4_sweep_analysis.py",
            "--input", parquet_arg,
            "--output-dir", str(out_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert res.returncode == 0
    assert (out_dir / "bbob_dim4_scorecard.csv").exists()
