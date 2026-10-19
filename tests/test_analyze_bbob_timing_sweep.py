#!/usr/bin/env python3
"""Tests for BBOB Dimensionality Timing Sweep statistical analysis and table generation."""

import json
import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.analyze_bbob_timing_sweep import (
    load_timing_records,
    compute_dimension_approach_aggregates,
    compute_relative_slowdown,
    format_markdown_table,
    format_slowdown_markdown_table,
)


@pytest.fixture
def sample_timing_dir(tmp_path):
    """Creates synthetic timing JSON records for 2 optimizers across 2 dimensions."""
    sweep_dir = tmp_path / "sweep_results"
    sweep_dir.mkdir()

    # Create records for SMAC3_HPOFacade_lcb and SMAC20_ProximityA_LCB across dim 2 and 4
    configs = [
        ("SMAC3_HPOFacade_lcb", 2, [10.0, 11.0, 12.0]),
        ("SMAC3_HPOFacade_lcb", 4, [20.0, 21.0, 22.0]),
        ("SMAC20_ProximityA_LCB", 2, [15.0, 16.5, 18.0]),
        ("SMAC20_ProximityA_LCB", 4, [40.0, 42.0, 44.0]),
    ]

    for opt_id, dim, times in configs:
        for idx, t in enumerate(times, start=1):
            record = {
                "exit_code": 0,
                "wallclock_seconds": t,
                "bo_overhead_seconds": t - 0.01,
                "target_eval_seconds": 0.01,
                "task_info": {
                    "optimizer_id": opt_id,
                    "dimension": dim,
                    "function_id": 8,
                    "instance_id": 0,
                    "seed": idx,
                },
                "hardware": {"cpu_model": "Test-CPU", "hostname": "worker-1"},
            }
            fname = sweep_dir / f"timing_{opt_id}_cfg_{dim}_8_0_seed{idx}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(record, f)

    return sweep_dir


def test_load_timing_records(sample_timing_dir):
    """Verify loading and parsing of timing records."""
    records = load_timing_records(str(sample_timing_dir))
    assert len(records) == 12
    for r in records:
        assert r["exit_code"] == 0
        assert r["optimizer_id"] in ["SMAC3_HPOFacade_lcb", "SMAC20_ProximityA_LCB"]
        assert r["dimension"] in [2, 4]


def test_compute_dimension_approach_aggregates(sample_timing_dir):
    """Verify statistical aggregation (mean, std, median, IQR)."""
    records = load_timing_records(str(sample_timing_dir))
    aggs = compute_dimension_approach_aggregates(records, metric_key="wallclock_seconds")

    lcb_d2 = aggs["SMAC3_HPOFacade_lcb"][2]
    assert pytest.approx(lcb_d2["mean"], rel=1e-3) == 11.0
    assert pytest.approx(lcb_d2["median"], rel=1e-3) == 11.0
    assert lcb_d2["count"] == 3

    prox_d4 = aggs["SMAC20_ProximityA_LCB"][4]
    assert pytest.approx(prox_d4["mean"], rel=1e-3) == 42.0
    assert pytest.approx(prox_d4["median"], rel=1e-3) == 42.0


def test_compute_relative_slowdown(sample_timing_dir):
    """Verify slowdown calculation relative to baseline."""
    records = load_timing_records(str(sample_timing_dir))
    aggs = compute_dimension_approach_aggregates(records, metric_key="wallclock_seconds")
    slowdowns = compute_relative_slowdown(aggs, baseline_id="SMAC3_HPOFacade_lcb", agg_type="mean")

    # In dim 2: 16.5 / 11.0 = 1.5x
    assert pytest.approx(slowdowns["SMAC20_ProximityA_LCB"][2], rel=1e-3) == 1.5
    # In dim 4: 42.0 / 21.0 = 2.0x
    assert pytest.approx(slowdowns["SMAC20_ProximityA_LCB"][4], rel=1e-3) == 2.0
    # Baseline is 1.0x
    assert pytest.approx(slowdowns["SMAC3_HPOFacade_lcb"][2], rel=1e-3) == 1.0


def test_format_markdown_tables(sample_timing_dir):
    """Verify markdown table rendering."""
    records = load_timing_records(str(sample_timing_dir))
    aggs = compute_dimension_approach_aggregates(records)
    slowdowns = compute_relative_slowdown(aggs, baseline_id="SMAC3_HPOFacade_lcb")

    md_timing = format_markdown_table(aggs, dimensions=[2, 4])
    assert "| Approach | Dim 2 | Dim 4 |" in md_timing
    assert "SMAC3_HPOFacade_lcb" in md_timing
    assert "SMAC20_ProximityA_LCB" in md_timing

    md_slowdown = format_slowdown_markdown_table(slowdowns, dimensions=[2, 4])
    assert "| Approach | Dim 2 | Dim 4 |" in md_slowdown
    assert "1.00x" in md_slowdown
    assert "1.50x" in md_slowdown
