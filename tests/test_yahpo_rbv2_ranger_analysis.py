#!/usr/bin/env python3
"""Tests for dedicated YAHPO Gym rbv2_ranger data gathering and analysis scripts."""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.gather_yahpo_rbv2_ranger_sweep import (
    gather_yahpo_rbv2_ranger_trial_logs,
    extract_yahpo_task_attributes,
)
from scripts.compute_yahpo_rbv2_ranger_sweep_analysis import (
    run_yahpo_rbv2_ranger_analysis,
    export_yahpo_rbv2_ranger_analysis,
)


def test_extract_yahpo_task_attributes():
    """Verify task attribute extraction handles multiple string formats without capturing suite '2'."""
    assert extract_yahpo_task_attributes("cfg_rbv2_ranger_1040")["dataset_id"] == 1040
    assert extract_yahpo_task_attributes("rbv2_ranger_1040")["dataset_id"] == 1040
    assert extract_yahpo_task_attributes("rbv2_ranger/1040")["dataset_id"] == 1040
    assert extract_yahpo_task_attributes("1040")["dataset_id"] == 1040
    assert extract_yahpo_task_attributes("cfg_rbv2_ranger_1040")["task_id"] == "cfg_rbv2_ranger_1040"


def test_gather_yahpo_rbv2_ranger_trial_logs_carps_hierarchy(tmp_path):
    """Verify gatherer parses CARP-S standard path, n_trials, None trial_value, and NaN costs."""
    runs_dir = tmp_path / "runs/sweep_yahpo_rbv2_ranger"
    out_dir = tmp_path / "results/sweep_yahpo_rbv2_ranger"

    # CARP-S nested directory: .../SMAC3_HPOFacade_lcb/YAHPO/yahpo/so/rbv2_ranger/1040/None/1/trial_logs.jsonl
    run_path = runs_dir / "SMAC3_HPOFacade_lcb/YAHPO/yahpo/so/rbv2_ranger/1040/None/1"
    run_path.mkdir(parents=True)
    with open(run_path / "trial_logs.jsonl", "w") as f:
        f.write(json.dumps({"n_trials": 1, "n_function_calls": 1, "trial_value": {"cost": 0.45}}) + "\n")
        f.write(json.dumps({"n_trials": 2, "n_function_calls": 2, "trial_value": None}) + "\n")  # Failed trial
        f.write(json.dumps({"n_trials": 3, "n_function_calls": 3, "trial_value": {"cost": float("nan")}}) + "\n")
        f.write(json.dumps({"n_trials": 4, "n_function_calls": 4, "trial_value": {"cost": 0.32}}) + "\n")

    df = gather_yahpo_rbv2_ranger_trial_logs(runs_base=runs_dir, outdir=out_dir)
    assert len(df) == 4
    assert "task_id" in df.columns
    assert "optimizer_id" in df.columns
    assert "seed" in df.columns
    assert "trial" in df.columns
    assert df["task_id"].iloc[0] == "cfg_rbv2_ranger_1040"
    assert df["dataset_id"].iloc[0] == 1040
    assert df["optimizer_id"].iloc[0] == "SMAC3_HPOFacade_lcb"
    assert df["seed"].iloc[0] == 1
    assert df["trial"].iloc[0] == 1
    assert df["trial"].iloc[3] == 4
    assert df["trial_value__cost_inc"].iloc[3] == 0.32
    assert (out_dir / "logs.csv").exists()
    try:
        import pyarrow
        assert (out_dir / "logs.parquet").exists()
    except ImportError:
        pass


def test_gather_yahpo_rbv2_ranger_telemetry_fallback(tmp_path):
    """Verify gatherer falls back to telemetry JSON files with task_name and filename stem."""
    runs_dir = tmp_path / "runs/empty_runs"
    runs_dir.mkdir(parents=True)
    out_dir = tmp_path / "results/sweep_yahpo_rbv2_ranger"
    out_dir.mkdir(parents=True)

    telemetry_file = out_dir / "telemetry_SMAC20_ProximityA_LCB_cfg_rbv2_ranger_1049_seed2.json"
    t_content = {
        "optimizer_id": "SMAC20_ProximityA_LCB",
        "task_name": "cfg_rbv2_ranger_1049",
        "seed": 2,
        "trials": [
            {"trial_idx": 1, "cost": 0.40},
            {"trial_idx": 2, "cost": 0.28},
        ],
    }
    with open(telemetry_file, "w") as f:
        json.dump(t_content, f)

    df = gather_yahpo_rbv2_ranger_trial_logs(runs_base=runs_dir, outdir=out_dir)
    assert len(df) == 2
    assert df["task_id"].iloc[0] == "cfg_rbv2_ranger_1049"
    assert df["dataset_id"].iloc[0] == 1049
    assert df["optimizer_id"].iloc[0] == "SMAC20_ProximityA_LCB"
    assert df["seed"].iloc[0] == 2
    assert df["trial"].iloc[0] == 1
    assert df["trial_value__cost_inc"].iloc[1] == 0.28


def test_compute_yahpo_rbv2_ranger_analysis(tmp_path):
    """Verify statistical scorecard with both controls on synthetic YAHPO ranger logs."""
    records = []
    # 2 tasks, 3 optimizers (Control LCB + Control EI + Prox A), 3 seeds, 2 trials
    tasks = ["cfg_rbv2_ranger_1040", "cfg_rbv2_ranger_1049"]
    optimizers = ["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei", "SMAC20_ProximityA_LCB"]

    for task in tasks:
        for opt in optimizers:
            if opt == "SMAC3_HPOFacade_lcb":
                base_cost = 0.50
            elif opt == "SMAC3_HPOFacade_ei":
                base_cost = 0.48
            else:
                base_cost = 0.40

            for seed in [1, 2, 3]:
                for trial, c in [(1, base_cost), (2, base_cost * 0.9)]:
                    records.append({
                        "task_id": task,
                        "optimizer_id": opt,
                        "seed": seed,
                        "trial": trial,
                        "cost": c,
                        "trial_value__cost_inc": c,
                    })

    df = pd.DataFrame(records)
    results = run_yahpo_rbv2_ranger_analysis(
        df,
        cost_col="trial_value__cost_inc",
        controls=["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"],
    )

    assert "scorecard" in results
    scorecard = results["scorecard"]
    assert isinstance(scorecard, pd.DataFrame)
    assert len(scorecard) == 3
    assert "optimizer_id" in scorecard.columns
    assert "mean_norm_regret" in scorecard.columns
    assert "ctrl_lcb_cliffs_delta" in scorecard.columns
    assert "ctrl_ei_cliffs_delta" in scorecard.columns

    out_dir = str(tmp_path / "results_analysis")
    export_yahpo_rbv2_ranger_analysis(results, out_dir)
    assert (Path(out_dir) / "yahpo_rbv2_ranger_scorecard.csv").exists()
    assert (Path(out_dir) / "yahpo_rbv2_ranger_scorecard.md").exists()
