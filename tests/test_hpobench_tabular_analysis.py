#!/usr/bin/env python3
"""Tests for HPOBench Tabular ML & NAS log gathering and statistical analysis scripts."""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.gather_hpobench_ml_sweep import (
    HPOBENCH_ML_OUTDIR,
    HPOBENCH_ML_RUNS_BASE,
    extract_hpobench_ml_task_attributes,
    gather_hpobench_ml_trial_logs,
    parse_run_dir_hierarchy as parse_ml_run_dir_hierarchy,
)
from scripts.compute_hpobench_ml_sweep_analysis import (
    export_hpobench_ml_analysis,
    run_hpobench_ml_analysis,
)
from scripts.gather_hpobench_nas_sweep import (
    HPOBENCH_NAS_OUTDIR,
    HPOBENCH_NAS_RUNS_BASE,
    extract_hpobench_nas_task_attributes,
    gather_hpobench_nas_trial_logs,
    parse_run_dir_hierarchy as parse_nas_run_dir_hierarchy,
)
from scripts.compute_hpobench_nas_sweep_analysis import (
    export_hpobench_nas_analysis,
    run_hpobench_nas_analysis,
)


def test_extract_hpobench_ml_task_attributes():
    """Verify HPOBench ML task attribute extraction handles CARP-S hierarchical paths and config names."""
    assert HPOBENCH_ML_RUNS_BASE == "runs/sweep_hpobench_ml"
    assert HPOBENCH_ML_OUTDIR == "results/sweep_hpobench_ml"

    a1 = extract_hpobench_ml_task_attributes("hpobench/blackbox/tabular/ml/lr/146818")
    assert a1 == {
        "model": "lr",
        "dataset_id": 146818,
        "task_id": "cfg_ml_lr_146818",
        "suite": "hpobench_ml",
    }

    a2 = extract_hpobench_ml_task_attributes("cfg_ml_xgboost_12")
    assert a2 == {
        "model": "xgboost",
        "dataset_id": 12,
        "task_id": "cfg_ml_xgboost_12",
        "suite": "hpobench_ml",
    }


def test_gather_hpobench_ml_trial_logs_and_fallback(tmp_path):
    """Verify HPOBench ML log gathering handles CARP-S nested hierarchy, flattened hierarchy, and telemetry fallback."""
    runs_dir = tmp_path / "runs/sweep_hpobench_ml"
    out_dir = tmp_path / "results/sweep_hpobench_ml"

    # Nested CARP-S path
    p1 = runs_dir / "SMAC3_HPOFacade_lcb/HPOBench/hpobench/blackbox/tabular/ml/lr/146818/1"
    p1.mkdir(parents=True)
    with open(p1 / "trial_logs.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"n_trials": 1, "trial_value": {"cost": 0.45}}) + "\n")
        f.write(json.dumps({"n_trials": 2, "trial_value": {"cost": 0.30}}) + "\n")

    # Flattened path
    p2 = runs_dir / "SMAC20_ProximityA_LCB/cfg_ml_xgboost_12/2"
    p2.mkdir(parents=True)
    with open(p2 / "trial_logs.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"n_trials": 1, "trial_value": {"cost": 0.25}}) + "\n")

    df = gather_hpobench_ml_trial_logs(runs_base=runs_dir, outdir=out_dir)
    assert len(df) == 3
    assert set(df["task_id"]) == {"cfg_ml_lr_146818", "cfg_ml_xgboost_12"}
    assert (out_dir / "logs.csv").exists()
    assert (out_dir / "logs.parquet").exists()
    assert (out_dir / "trajectory.parquet").exists()

    # Fallback telemetry test
    empty_runs = tmp_path / "runs/empty_ml"
    empty_runs.mkdir(parents=True)
    telem_out = tmp_path / "results/telem_ml"
    telem_out.mkdir(parents=True)
    t_file = telem_out / "telemetry_SMAC20_ProximityA_LCB_cfg_ml_rf_31_seed3.json"
    with open(t_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "optimizer_id": "SMAC20_ProximityA_LCB",
                "task_name": "cfg_ml_rf_31",
                "seed": 3,
                "trials": [{"trial_idx": 1, "cost": 0.20}, {"trial_idx": 2, "cost": 0.15}],
            },
            f,
        )
    df_telem = gather_hpobench_ml_trial_logs(runs_base=empty_runs, outdir=telem_out)
    assert len(df_telem) == 2
    assert df_telem["task_id"].iloc[0] == "cfg_ml_rf_31"
    assert df_telem["model"].iloc[0] == "rf"
    assert df_telem["dataset_id"].iloc[0] == 31


def test_extract_hpobench_nas_task_attributes():
    """Verify HPOBench NAS task attribute extraction handles CARP-S hierarchical paths and config names."""
    assert HPOBENCH_NAS_RUNS_BASE == "runs/sweep_hpobench_nas"
    assert HPOBENCH_NAS_OUTDIR == "results/sweep_hpobench_nas"

    a1 = extract_hpobench_nas_task_attributes("hpobench/blackbox/tabular/nas/Cifar100NasBench201Benchmark")
    assert a1 == {
        "benchmark": "Cifar100NasBench201Benchmark",
        "task_id": "cfg_Cifar100NasBench201Benchmark",
        "suite": "hpobench_nas",
    }

    a2 = extract_hpobench_nas_task_attributes("cfg_ProteinStructureBenchmark")
    assert a2 == {
        "benchmark": "ProteinStructureBenchmark",
        "task_id": "cfg_ProteinStructureBenchmark",
        "suite": "hpobench_nas",
    }


def test_gather_hpobench_nas_trial_logs_and_fallback(tmp_path):
    """Verify HPOBench NAS log gathering handles CARP-S nested hierarchy, flattened hierarchy, and telemetry fallback."""
    runs_dir = tmp_path / "runs/sweep_hpobench_nas"
    out_dir = tmp_path / "results/sweep_hpobench_nas"

    p1 = runs_dir / "SMAC3_HPOFacade_lcb/HPOBench/hpobench/blackbox/tabular/nas/Cifar100NasBench201Benchmark/1"
    p1.mkdir(parents=True)
    with open(p1 / "trial_logs.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"n_trials": 1, "trial_value": {"cost": 0.40}}) + "\n")
        f.write(json.dumps({"n_trials": 2, "trial_value": {"cost": 0.22}}) + "\n")

    p2 = runs_dir / "SMAC20_ProximityA_LCB/cfg_SliceLocalizationBenchmark/2"
    p2.mkdir(parents=True)
    with open(p2 / "trial_logs.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"n_trials": 1, "trial_value": {"cost": 0.18}}) + "\n")

    df = gather_hpobench_nas_trial_logs(runs_base=runs_dir, outdir=out_dir)
    assert len(df) == 3
    assert set(df["task_id"]) == {"cfg_Cifar100NasBench201Benchmark", "cfg_SliceLocalizationBenchmark"}
    assert (out_dir / "logs.csv").exists()
    assert (out_dir / "logs.parquet").exists()
    assert (out_dir / "trajectory.parquet").exists()

    # Fallback telemetry test
    empty_runs = tmp_path / "runs/empty_nas"
    empty_runs.mkdir(parents=True)
    telem_out = tmp_path / "results/telem_nas"
    telem_out.mkdir(parents=True)
    t_file = telem_out / "telemetry_SMAC20_ProximityA_LCB_cfg_NavalPropulsionBenchmark_seed4.json"
    with open(t_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "optimizer_id": "SMAC20_ProximityA_LCB",
                "task_name": "cfg_NavalPropulsionBenchmark",
                "seed": 4,
                "trials": [{"trial_idx": 1, "cost": 0.12}],
            },
            f,
        )
    df_telem = gather_hpobench_nas_trial_logs(runs_base=empty_runs, outdir=telem_out)
    assert len(df_telem) == 1
    assert df_telem["task_id"].iloc[0] == "cfg_NavalPropulsionBenchmark"
    assert df_telem["benchmark"].iloc[0] == "NavalPropulsionBenchmark"


def test_compute_hpobench_ml_and_nas_analysis(tmp_path):
    """Verify statistical analysis and scorecard export for both HPOBench ML and NAS."""
    records = []
    for task in ["cfg_ml_lr_146818", "cfg_ml_xgboost_12"]:
        for opt, base_cost in [
            ("SMAC3_HPOFacade_lcb", 0.50),
            ("SMAC3_HPOFacade_ei", 0.48),
            ("SMAC20_ProximityA_LCB", 0.35),
        ]:
            for seed in [1, 2, 3]:
                for trial, c in [(1, base_cost), (2, base_cost * 0.9)]:
                    records.append(
                        {
                            "task_id": task,
                            "optimizer_id": opt,
                            "seed": seed,
                            "trial": trial,
                            "cost": c,
                            "trial_value__cost_inc": c,
                        }
                    )
    df = pd.DataFrame(records)

    ml_res = run_hpobench_ml_analysis(df)
    ml_out = tmp_path / "results/sweep_hpobench_ml"
    export_hpobench_ml_analysis(ml_res, str(ml_out))
    assert (ml_out / "hpobench_ml_scorecard.csv").exists()
    assert (ml_out / "hpobench_ml_scorecard.md").exists()

    nas_res = run_hpobench_nas_analysis(df)
    nas_out = tmp_path / "results/sweep_hpobench_nas"
    export_hpobench_nas_analysis(nas_res, str(nas_out))
    assert (nas_out / "hpobench_nas_scorecard.csv").exists()
    assert (nas_out / "hpobench_nas_scorecard.md").exists()
