#!/usr/bin/env python3
"""Tests for HPOBench Tabular ML (88 tasks) and HPOBench Tabular NAS (10 tasks) registry and sweep generators."""

import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_realworld_registry import CarpsRealworldRegistry
from scripts.generate_bbob_full_sweep_tasks import ALL_OPTIMIZERS
from scripts.generate_hpobench_ml_sweep_tasks import (
    HPOBENCH_ML_OUTPUT_DIR,
    HPOBENCH_ML_RUNS_DIR,
    build_hpobench_ml_command_line,
    generate_hpobench_ml_sweep_tasks,
)
from scripts.generate_hpobench_nas_sweep_tasks import (
    HPOBENCH_NAS_OUTPUT_DIR,
    HPOBENCH_NAS_RUNS_DIR,
    build_hpobench_nas_command_line,
    generate_hpobench_nas_sweep_tasks,
)


EXPECTED_NAS_CONFIGS = [
    "cfg_Cifar100NasBench201Benchmark",
    "cfg_Cifar10ValidNasBench201Benchmark",
    "cfg_ImageNetNasBench201Benchmark",
    "cfg_NASCifar10ABenchmark",
    "cfg_NASCifar10BBenchmark",
    "cfg_NASCifar10CBenchmark",
    "cfg_NavalPropulsionBenchmark",
    "cfg_ParkinsonsTelemonitoringBenchmark",
    "cfg_ProteinStructureBenchmark",
    "cfg_SliceLocalizationBenchmark",
]


def test_registry_hpobench_ml_and_nas_tasks():
    """Verify CarpsRealworldRegistry returns all 88 HPOBench ML tasks and all 10 HPOBench NAS tasks."""
    ml_tasks = CarpsRealworldRegistry.get_hpobench_ml_tasks()
    assert len(ml_tasks) == 88
    assert all(t.startswith("task=HPOBench/blackbox/tabular/ml/cfg_ml_") for t in ml_tasks)
    assert CarpsRealworldRegistry.get_tasks_for_suite("hpobench_ml") == ml_tasks

    nas_tasks = CarpsRealworldRegistry.get_hpobench_nas_tasks()
    assert len(nas_tasks) == 10
    expected_nas_tasks = [f"task=HPOBench/blackbox/tabular/nas/{c}" for c in EXPECTED_NAS_CONFIGS]
    assert nas_tasks == expected_nas_tasks
    assert CarpsRealworldRegistry.get_tasks_for_suite("hpobench_nas") == nas_tasks
    assert CarpsRealworldRegistry.get_tasks_for_suite("nas") == nas_tasks


def test_generate_hpobench_ml_sweep_tasks_dry(tmp_path):
    """Verify HPOBench ML task generator creates 88 * 11 * N_seeds tasks and writes manifest."""
    assert HPOBENCH_ML_OUTPUT_DIR == "results/sweep_hpobench_ml"
    assert HPOBENCH_ML_RUNS_DIR == "runs/sweep_hpobench_ml"

    out_dir = str(tmp_path / "results_ml")
    runs_dir = str(tmp_path / "runs_ml")

    tasks = generate_hpobench_ml_sweep_tasks(
        seeds=2,
        trials=100,
        output_dir=out_dir,
        runs_dir=out_dir,
        baserundir=runs_dir,
    )
    assert len(tasks) == 88 * len(ALL_OPTIMIZERS) * 2
    assert len(tasks) == 88 * 11 * 2

    for cmd in tasks:
        assert "+task=HPOBench/blackbox/tabular/ml/cfg_ml_" in cmd
        assert f"baserundir={runs_dir}" in cmd
        assert "task.optimization_resources.n_trials=100" in cmd

    manifest_file = Path(out_dir) / "tasks.txt"
    assert manifest_file.exists()
    lines = [l.strip() for l in manifest_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 88 * 11 * 2


def test_generate_hpobench_ml_full_scale_count(tmp_path):
    """Verify full-scale default 20-seed HPOBench ML manifest has 19,360 tasks across seeds 1..20."""
    out_dir = str(tmp_path / "results_ml_full")
    runs_dir = str(tmp_path / "runs_ml_full")

    tasks = generate_hpobench_ml_sweep_tasks(
        output_dir=out_dir,
        runs_dir=out_dir,
        baserundir=runs_dir,
    )
    assert len(tasks) == 19360
    for s in range(1, 21):
        assert any(f"seed={s} " in cmd for cmd in tasks[:20])


def test_generate_hpobench_nas_sweep_tasks_dry(tmp_path):
    """Verify HPOBench NAS task generator creates 10 * 11 * N_seeds tasks and writes manifest."""
    assert HPOBENCH_NAS_OUTPUT_DIR == "results/sweep_hpobench_nas"
    assert HPOBENCH_NAS_RUNS_DIR == "runs/sweep_hpobench_nas"

    out_dir = str(tmp_path / "results_nas")
    runs_dir = str(tmp_path / "runs_nas")

    tasks = generate_hpobench_nas_sweep_tasks(
        seeds=2,
        trials=100,
        output_dir=out_dir,
        runs_dir=out_dir,
        baserundir=runs_dir,
    )
    assert len(tasks) == 10 * len(ALL_OPTIMIZERS) * 2
    assert len(tasks) == 10 * 11 * 2

    for cmd in tasks:
        assert "+task=HPOBench/blackbox/tabular/nas/cfg_" in cmd
        assert f"baserundir={runs_dir}" in cmd
        assert "task.optimization_resources.n_trials=100" in cmd

    manifest_file = Path(out_dir) / "tasks.txt"
    assert manifest_file.exists()
    lines = [l.strip() for l in manifest_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 10 * 11 * 2


def test_generate_hpobench_nas_full_scale_count(tmp_path):
    """Verify full-scale default 20-seed HPOBench NAS manifest has 2,200 tasks across seeds 1..20."""
    out_dir = str(tmp_path / "results_nas_full")
    runs_dir = str(tmp_path / "runs_nas_full")

    tasks = generate_hpobench_nas_sweep_tasks(
        output_dir=out_dir,
        runs_dir=out_dir,
        baserundir=runs_dir,
    )
    assert len(tasks) == 2200
    for s in range(1, 21):
        assert any(f"seed={s} " in cmd for cmd in tasks[:20])
