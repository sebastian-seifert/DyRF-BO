#!/usr/bin/env python3
"""Tests for BBOB dimension wrapper analysis scripts (dim2 through dim32)."""

import os
import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.compute_bbob_dim2_sweep_analysis import run_bbob_dim2_analysis, export_dim2_analysis
from scripts.compute_bbob_dim4_sweep_analysis import run_bbob_dim4_analysis, export_dim4_analysis
from scripts.compute_bbob_dim8_sweep_analysis import run_bbob_dim8_analysis, export_dim8_analysis
from scripts.compute_bbob_dim16_sweep_analysis import run_bbob_dim16_analysis, export_dim16_analysis
from scripts.compute_bbob_dim32_sweep_analysis import run_bbob_dim32_analysis, export_dim32_analysis


@pytest.mark.parametrize(
    "dim, fn",
    [
        (2, run_bbob_dim2_analysis),
        (4, run_bbob_dim4_analysis),
        (8, run_bbob_dim8_analysis),
        (16, run_bbob_dim16_analysis),
        (32, run_bbob_dim32_analysis),
    ],
)
def test_dim_wrapper_filters_correctly_without_dimension_column(dim, fn):
    """Verify that wrappers infer dimension from task_id and filter strictly to target dimension."""
    # Create mixed dataframe containing tasks from dim 2, 4, 8, 16, 32 WITHOUT dimension column
    records = []
    for d in [2, 4, 8, 16, 32]:
        for opt in ["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei", "SMAC20_ProximityA_LCB"]:
            for seed in [1, 2]:
                records.append({
                    "task_id": f"cfg_{d}_1_0",
                    "optimizer_id": opt,
                    "seed": seed,
                    "trial": 100,
                    "cost": 1.0 if opt == "SMAC20_ProximityA_LCB" else 2.0,
                    "trial_value__cost_inc": 1.0 if opt == "SMAC20_ProximityA_LCB" else 2.0,
                })

    df_mixed = pd.DataFrame(records)
    assert "dimension" not in df_mixed.columns

    results = fn(df_mixed)
    scorecard = results["scorecard"]
    assert not scorecard.empty

    # Each dimension should only have 1 task (cfg_{d}_1_0)
    for _, row in scorecard.iterrows():
        assert row["n_tasks"] == 1
        assert row["n_runs"] == 2


@pytest.mark.parametrize(
    "dim, fn, export_fn",
    [
        (2, run_bbob_dim2_analysis, export_dim2_analysis),
        (4, run_bbob_dim4_analysis, export_dim4_analysis),
        (8, run_bbob_dim8_analysis, export_dim8_analysis),
        (16, run_bbob_dim16_analysis, export_dim16_analysis),
        (32, run_bbob_dim32_analysis, export_dim32_analysis),
    ],
)
def test_dim_wrapper_export(tmp_path, dim, fn, export_fn):
    """Verify that dim wrapper results are properly exported to CSV and Markdown."""
    records = []
    for opt in ["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei", "SMAC20_ProximityA_LCB"]:
        for seed in [1, 2]:
            records.append({
                "task_id": f"cfg_{dim}_1_0",
                "optimizer_id": opt,
                "seed": seed,
                "trial": 100,
                "cost": 1.0 if opt == "SMAC20_ProximityA_LCB" else 2.0,
                "trial_value__cost_inc": 1.0 if opt == "SMAC20_ProximityA_LCB" else 2.0,
            })
    df = pd.DataFrame(records)
    res = fn(df)

    out_dir = tmp_path / f"dim{dim}_out"
    export_fn(res, str(out_dir))

    assert (out_dir / f"bbob_dim{dim}_scorecard.csv").exists()
    assert (out_dir / f"bbob_dim{dim}_analysis_report.md").exists()
    md_content = (out_dir / f"bbob_dim{dim}_analysis_report.md").read_text()
    assert f"Dimension {dim}" in md_content

