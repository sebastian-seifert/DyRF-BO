#!/usr/bin/env python3
"""Tests for BBOB Full Sweep Data Gathering Script."""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest

from scripts.gather_bbob_full_sweep import (
    extract_bbob_task_attributes,
    gather_bbob_trial_logs,
)


def test_extract_bbob_task_attributes():
    """Verify metadata extraction handles task IDs correctly."""
    attrs = extract_bbob_task_attributes("cfg_16_12_0")
    assert attrs["dimension"] == 16
    assert attrs["fid"] == 12
    assert attrs["iid"] == 0
    assert attrs["function_name"] == "Bent Cigar"
    assert attrs["function_category"] == "ill_conditioned"

    attrs_2d = extract_bbob_task_attributes("+task=BBOB/cfg_2_1_1")
    assert attrs_2d["dimension"] == 2
    assert attrs_2d["fid"] == 1
    assert attrs_2d["iid"] == 1
    assert attrs_2d["function_name"] == "Sphere"
    assert attrs_2d["function_category"] == "separable"


def test_gather_bbob_trial_logs_from_jsonl(tmp_path):
    """Verify parsing from directory containing trial_logs.jsonl."""
    runs_dir = tmp_path / "runs"
    task_run_dir = runs_dir / "SMAC20_ProximityA_LCB" / "BBOB" / "cfg_8_3_0" / "1"
    task_run_dir.mkdir(parents=True)

    # Mock trial_logs.jsonl
    log_file = task_run_dir / "trial_logs.jsonl"
    with open(log_file, "w") as f:
        f.write(json.dumps({"trial": 1, "trial_value": {"cost": 10.5}}) + "\n")
        f.write(json.dumps({"trial": 2, "trial_value": {"cost": 4.2}}) + "\n")
        f.write(json.dumps({"trial": 3, "trial_value": {"cost": 8.0}}) + "\n")

    out_dir = tmp_path / "results"
    out_dir.mkdir(parents=True)

    df = gather_bbob_trial_logs(runs_base=runs_dir, outdir=out_dir)

    assert not df.empty
    assert len(df) == 3
    assert "dimension" in df.columns
    assert "fid" in df.columns
    assert "function_category" in df.columns
    assert "trial_value__cost_inc" in df.columns

    assert df["dimension"].iloc[0] == 8
    assert df["fid"].iloc[0] == 3
    assert df["function_name"].iloc[0] == "Rastrigin"
    assert df["function_category"].iloc[0] == "separable"
    # Check running incumbent
    assert list(df["trial_value__cost_inc"]) == [10.5, 4.2, 4.2]

    # Verify export files exist
    assert (out_dir / "logs.parquet").exists()
    assert (out_dir / "logs.csv").exists()


def test_gather_bbob_trial_logs_from_telemetry_fallback(tmp_path):
    """Verify fallback parsing from telemetry_*.json files when trial_logs.jsonl is absent."""
    runs_dir = tmp_path / "empty_runs"
    runs_dir.mkdir(parents=True)
    out_dir = tmp_path / "results"
    out_dir.mkdir(parents=True)

    telemetry_file = out_dir / "telemetry_SMAC3_HPOFacade_lcb_cfg_32_20_0_seed2.json"
    telem_data = {
        "optimizer_id": "SMAC3_HPOFacade_lcb",
        "task_id": "cfg_32_20_0",
        "seed": 2,
        "history": [
            {"trial": 1, "cost": 500.0},
            {"trial": 2, "cost": 350.0},
        ],
    }
    with open(telemetry_file, "w") as f:
        json.dump(telem_data, f)

    df = gather_bbob_trial_logs(runs_base=runs_dir, outdir=out_dir)
    assert not df.empty
    assert len(df) == 2
    assert df["dimension"].iloc[0] == 32
    assert df["fid"].iloc[0] == 20
    assert df["function_category"].iloc[0] == "multimodal_weak_global"
