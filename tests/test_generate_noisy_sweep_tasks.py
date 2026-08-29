"""TDD Test Suite for Noisy Benchmark EI Head-to-Head Sweep Task Generator."""

import os
import sys
import pytest

# Ensure DyRF-BO root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from scripts.generate_noisy_sweep_tasks import (
    generate_noisy_sweep_tasks,
    get_noisy_tasks,
    get_approach_configs,
    NOISY_TASKS_HETGP,
    NOISY_TASKS_BBOB,
)


def test_noisy_tasks_registry():
    """Ensure all 16 noisy tasks are present and their YAML files exist on disk."""
    tasks = get_noisy_tasks(suite="all")
    assert len(tasks) == 16
    assert len(get_noisy_tasks("hetgp")) == 5
    assert len(get_noisy_tasks("bbob")) == 11
    
    for task_arg in tasks:
        # task_arg is like +task/Noisy/hetgp=cfg_branin_2d
        rel_path = os.path.join(
            PROJECT_ROOT,
            "carps_integration",
            "configs",
            "task",
            task_arg.replace("+task/", "").replace("=", "/") + ".yaml"
        )
        assert os.path.isfile(rel_path), f"Missing task YAML: {rel_path}"


def test_approach_matrix_completeness():
    """Verify exact 8 approaches (1 baseline, 4 direct, 3 additive) and 0 Chen variance."""
    approaches = get_approach_configs()
    assert len(approaches) == 8
    
    optimizer_ids = [app["optimizer_id"] for app in approaches]
    assert len(optimizer_ids) == 8
    assert len(set(optimizer_ids)) == 8  # 100% distinct
    
    # Assert zero chen occurrences
    for app in approaches:
        assert "chen" not in app["optimizer_id"].lower()
        if "extractor_name" in app:
            assert "chen" not in app["extractor_name"].lower()


def test_default_10_seeds_generation_task_count(tmp_path):
    """Verify default 10-seed matrix produces exactly 1,280 tasks (16 tasks * 8 approaches * 10 seeds)."""
    out_file = str(tmp_path / "tasks_10seeds.txt")
    runs_dir = str(tmp_path / "runs")
    tasks = generate_noisy_sweep_tasks(
        output_file=out_file,
        runs_dir=runs_dir,
        n_seeds=10,
        trials=50
    )
    assert len(tasks) == 1280
    with open(out_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 1280


def test_custom_30_seeds_generation_task_count(tmp_path):
    """Verify 30-seed matrix produces exactly 3,840 tasks."""
    out_file = str(tmp_path / "tasks_30seeds.txt")
    runs_dir = str(tmp_path / "runs")
    tasks = generate_noisy_sweep_tasks(
        output_file=out_file,
        runs_dir=runs_dir,
        n_seeds=30,
        trials=50
    )
    assert len(tasks) == 3840


def test_optimizer_breakdown_10_seeds(tmp_path):
    """Verify distribution across baseline, direct, and additive paradigms for 10 seeds."""
    out_file = str(tmp_path / "tasks.txt")
    runs_dir = str(tmp_path / "runs")
    tasks = generate_noisy_sweep_tasks(output_file=out_file, runs_dir=runs_dir, n_seeds=10)
    
    baseline_count = sum(1 for line in tasks if "optimizer_id=SMAC3_HPOFacade_ei" in line)
    direct_count = sum(1 for line in tasks if "optimizer_container_id=SMAC20_CustomUncertainty" in line)
    additive_count = sum(1 for line in tasks if "optimizer_container_id=CARPSDynamicRF" in line)
    
    assert baseline_count == 16 * 10 * 1  # 160
    assert direct_count == 16 * 10 * 4    # 640
    assert additive_count == 16 * 10 * 3  # 480
    assert baseline_count + direct_count + additive_count == 1280


def test_benchmark_coverage_10_seeds(tmp_path):
    """Verify benchmark distribution across BBOB (11 tasks) and hetGP (5 tasks) for 10 seeds."""
    out_file = str(tmp_path / "tasks.txt")
    runs_dir = str(tmp_path / "runs")
    tasks = generate_noisy_sweep_tasks(output_file=out_file, runs_dir=runs_dir, n_seeds=10)
    
    hetgp_count = sum(1 for line in tasks if "+task/Noisy/hetgp" in line)
    bbob_count = sum(1 for line in tasks if "+task/Noisy/bbob" in line)
    
    assert hetgp_count == 5 * 8 * 10   # 400
    assert bbob_count == 11 * 8 * 10  # 880


def test_seed_distribution_10_seeds(tmp_path):
    """Verify seeds 1..10 each have exactly 128 task executions."""
    out_file = str(tmp_path / "tasks.txt")
    runs_dir = str(tmp_path / "runs")
    tasks = generate_noisy_sweep_tasks(output_file=out_file, runs_dir=runs_dir, n_seeds=10)
    
    for seed in range(1, 11):
        seed_count = sum(1 for line in tasks if f"seed={seed} " in line or line.endswith(f"seed={seed}"))
        assert seed_count == 16 * 8  # 128 runs per seed


def test_strictly_ei_acquisition(tmp_path):
    """Verify that all commands evaluate solely Expected Improvement (EI)."""
    out_file = str(tmp_path / "tasks.txt")
    runs_dir = str(tmp_path / "runs")
    tasks = generate_noisy_sweep_tasks(output_file=out_file, runs_dir=runs_dir, n_seeds=10)
    
    for line in tasks:
        assert "acq_func_name=pi" not in line
        assert "acq_func_name=lcb" not in line
        assert "optimizer_id=" in line
        assert "_ei" in line


def test_telemetry_uniqueness_10_seeds(tmp_path):
    """Verify 1,280 distinct telemetry paths with zero collisions."""
    out_file = str(tmp_path / "tasks.txt")
    runs_dir = str(tmp_path / "runs")
    tasks = generate_noisy_sweep_tasks(output_file=out_file, runs_dir=runs_dir, n_seeds=10)
    
    telemetry_paths = []
    for line in tasks:
        assert "++optimizer.telemetry_path=" in line
        parts = line.split("++optimizer.telemetry_path=")[1].split()[0]
        telemetry_paths.append(parts)
        
    assert len(telemetry_paths) == 1280
    assert len(set(telemetry_paths)) == 1280
