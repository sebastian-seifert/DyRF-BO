#!/usr/bin/env python3
"""Tests for dedicated BBOB Dimensions 8, 16, and 32 data gathering and analysis scripts."""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.gather_bbob_dim8_sweep import gather_bbob_dim8_trial_logs
from scripts.compute_bbob_dim8_sweep_analysis import run_bbob_dim8_analysis, export_dim8_analysis
from scripts.gather_bbob_dim16_sweep import gather_bbob_dim16_trial_logs
from scripts.compute_bbob_dim16_sweep_analysis import run_bbob_dim16_analysis, export_dim16_analysis
from scripts.gather_bbob_dim32_sweep import gather_bbob_dim32_trial_logs
from scripts.compute_bbob_dim32_sweep_analysis import run_bbob_dim32_analysis, export_dim32_analysis


@pytest.mark.parametrize("dim,gather_fn", [
    (8, gather_bbob_dim8_trial_logs),
    (16, gather_bbob_dim16_trial_logs),
    (32, gather_bbob_dim32_trial_logs),
])
def test_gather_bbob_highdim_trial_logs(tmp_path, dim, gather_fn):
    """Verify gather functions correctly filter to the requested dimension."""
    runs_dir = tmp_path / f"runs/sweep_bbob_dim{dim}"
    out_dir = tmp_path / f"results/sweep_bbob_dim{dim}"

    run_path = runs_dir / f"SMAC3_HPOFacade_lcb/BBOB/bbob/{dim}/1/0/1"
    run_path.mkdir(parents=True)
    with open(run_path / "trial_logs.jsonl", "w") as f:
        f.write(json.dumps({"trial": 1, "trial_value": {"cost": 10.0}}) + "\n")
        f.write(json.dumps({"trial": 2, "trial_value": {"cost": 5.0}}) + "\n")

    df = gather_fn(runs_base=runs_dir, outdir=out_dir)
    assert len(df) == 2
    assert "dimension" in df.columns
    assert (df["dimension"] == dim).all()
    assert (out_dir / "logs.csv").exists()


@pytest.mark.parametrize("dim,run_fn,export_fn", [
    (8, run_bbob_dim8_analysis, export_dim8_analysis),
    (16, run_bbob_dim16_analysis, export_dim16_analysis),
    (32, run_bbob_dim32_analysis, export_dim32_analysis),
])
def test_compute_bbob_highdim_analysis(tmp_path, dim, run_fn, export_fn):
    """Verify statistical analysis and scorecards for each high dimension."""
    records = []
    for task in [f"bbob/{dim}/1/0", f"bbob/{dim}/2/0"]:
        for opt, base_cost in [("SMAC3_HPOFacade_lcb", 10.0), ("SMAC20_ProximityA_LCB", 8.0)]:
            for seed in [1, 2]:
                for trial, c in [(1, base_cost), (2, base_cost * 0.8)]:
                    records.append({
                        "task_id": task,
                        "optimizer_id": opt,
                        "seed": seed,
                        "trial": trial,
                        "cost": c,
                        "trial_value__cost_inc": c,
                        "dimension": dim,
                        "fid": 1 if "1" in task else 2,
                        "iid": 0,
                    })
    df = pd.DataFrame(records)
    res = run_fn(df)
    assert "scorecard" in res
    assert "friedman_stat" in res

    out_dir = str(tmp_path / f"analysis_dim{dim}")
    export_fn(res, out_dir)
    assert (Path(out_dir) / f"bbob_dim{dim}_scorecard.csv").exists()
    assert (Path(out_dir) / f"bbob_dim{dim}_analysis_report.md").exists()
