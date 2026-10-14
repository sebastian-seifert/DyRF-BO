#!/usr/bin/env python3
"""End-to-End Verification Suite for YAHPO Gym rbv2_ranger Benchmark Pipeline."""

import os
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_realworld_registry import CarpsRealworldRegistry
from scripts.generate_bbob_full_sweep_tasks import ALL_OPTIMIZERS
from scripts.generate_yahpo_rbv2_ranger_sweep_tasks import (
    generate_yahpo_rbv2_ranger_sweep_tasks,
    build_yahpo_command_line,
)


def test_e2e_all_optimizers_and_all_tasks_command_syntax():
    """Verify syntactic correctness of Hydra overrides across all 11 optimizers and all 119 tasks."""
    ranger_tasks = CarpsRealworldRegistry.get_ranger_tasks()
    assert len(ranger_tasks) == 119
    assert len(ALL_OPTIMIZERS) == 11

    # Thoroughly validate all 11 optimizers
    for opt in ALL_OPTIMIZERS:
        cmd = build_yahpo_command_line(
            opt_info=opt,
            task_arg=ranger_tasks[0],
            seed=42,
            trials=100,
            runs_dir="runs/test_e2e",
            baserundir="runs/test_e2e",
        )
        assert "--config-dir carps_integration/configs" in cmd
        assert opt["config_arg"] in cmd
        assert f"optimizer_id={opt['id']}" in cmd
        assert f"optimizer_container_id={opt['container_id']}" in cmd

    # Thoroughly validate all 119 tasks
    for task_arg in ranger_tasks:
        cmd = build_yahpo_command_line(
            opt_info=ALL_OPTIMIZERS[0],
            task_arg=task_arg,
            seed=42,
            trials=100,
            runs_dir="runs/test_e2e",
            baserundir="runs/test_e2e",
        )
        assert "+task=YAHPO/blackbox/cfg_rbv2_ranger_" in cmd
        assert "task.optimization_resources.n_trials=100" in cmd
        assert "seed=42" in cmd


def test_e2e_total_manifest_cardinality(tmp_path):
    """Verify exact full-scale task count (119 * 11 * 20 = 26,180) in generated manifest."""
    out_dir = str(tmp_path / "results_ranger_full")
    runs_dir = str(tmp_path / "runs_ranger_full")

    # Generate with 20 seeds
    commands = generate_yahpo_rbv2_ranger_sweep_tasks(
        seeds=20,
        trials=100,
        output_dir=out_dir,
        runs_dir=out_dir,
        baserundir=runs_dir,
    )
    assert len(commands) == 26180
    assert (Path(out_dir) / "tasks.txt").exists()


def test_e2e_submission_scripts_executable_and_syntax():
    """Verify executable bits and bash syntax of all shell and sbatch scripts."""
    scripts = [
        "scripts/submit_yahpo_rbv2_ranger_sweep_array.sbatch",
        "scripts/submit_yahpo_rbv2_ranger_sweep_master.sbatch",
        "scripts/submit_yahpo_rbv2_ranger_sweep_all.sh",
    ]
    for s in scripts:
        path = Path(PROJECT_ROOT) / s
        assert path.exists(), f"Script {s} does not exist at {path}"
        assert os.access(path, os.X_OK), f"Script {s} is not executable"
        # Syntax check via bash -n
        res = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert res.returncode == 0, f"Bash syntax error in {s}: {res.stderr}"
