#!/usr/bin/env python3
"""Tests for Multi-Suite Proximity LCB Sweep Generator and Realworld Task Registry."""

import json
import os
import sys
import tempfile
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_realworld_registry import CarpsRealworldRegistry
from scripts.generate_multisuite_proximity_sweep_tasks import (
    load_proximity_params,
    generate_suite_tasks,
    chunk_tasks,
    generate_sbatch_content,
    generate_all_suite_artifacts,
)


class TestCarpsRealworldRegistry:
    def test_ranger_tasks_count_and_format(self):
        tasks = CarpsRealworldRegistry.get_ranger_tasks()
        assert len(tasks) == 119
        for t in tasks:
            assert t.startswith("task=YAHPO/blackbox/cfg_rbv2_ranger_")

    def test_super_tasks_count_and_format(self):
        tasks = CarpsRealworldRegistry.get_super_tasks()
        assert len(tasks) == 103
        for t in tasks:
            assert t.startswith("task=YAHPO/blackbox/cfg_rbv2_super_")

    def test_hpobench_ml_tasks_count_and_format(self):
        tasks = CarpsRealworldRegistry.get_hpobench_ml_tasks()
        assert len(tasks) == 88
        for t in tasks:
            assert t.startswith("task=HPOBench/blackbox/tabular/ml/cfg_ml_")

    def test_get_tasks_for_suite(self):
        ranger = CarpsRealworldRegistry.get_tasks_for_suite("yahpo_rbv2_ranger")
        super_t = CarpsRealworldRegistry.get_tasks_for_suite("yahpo_rbv2_super")
        hpobench = CarpsRealworldRegistry.get_tasks_for_suite("hpobench_ml")

        assert len(ranger) == 119
        assert len(super_t) == 103
        assert len(hpobench) == 88

        with pytest.raises(ValueError):
            CarpsRealworldRegistry.get_tasks_for_suite("invalid_suite_name")


class TestMultiSuiteProximitySweepGenerator:
    def test_load_proximity_params_default(self):
        params = load_proximity_params(None)
        assert params["k"] == 25
        assert params["decay_lambda"] == 1.345
        assert params["eps"] == 0.16
        assert params["level"] == 0.95
        assert params["uncertainty_func"] == "proximity_b"

    def test_load_proximity_params_from_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "k": 32,
                "decay_lambda": 1.75,
                "eps": 0.05,
                "level": 0.90,
                "uncertainty_func": "proximity_bc",
            }, f)
            temp_path = f.name

        try:
            params = load_proximity_params(temp_path)
            assert params["k"] == 32
            assert params["decay_lambda"] == 1.75
            assert params["eps"] == 0.05
            assert params["level"] == 0.90
            assert params["uncertainty_func"] == "proximity_bc"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_generate_suite_tasks_small(self):
        tasks = ["task=YAHPO/blackbox/cfg_rbv2_ranger_1040", "task=YAHPO/blackbox/cfg_rbv2_ranger_1049"]
        suite = "yahpo_rbv2_ranger"
        cmds = generate_suite_tasks(
            suite=suite,
            task_list=tasks,
            seeds=2,
            trials=100,
            proximity_params={"k": 25, "decay_lambda": 1.345, "eps": 0.16, "level": 0.95, "uncertainty_func": "proximity_b"},
            kappa_baseline=1.96,
        )
        # 2 tasks * 2 optimizers * 2 seeds = 8 commands
        assert len(cmds) == 8

        # 4 proposed + 4 baseline
        proposed = [c for c in cmds if "SMAC20_ProximityLCB" in c]
        baseline = [c for c in cmds if "SMAC3_HPOFacade_lcb" in c]
        assert len(proposed) == 4
        assert len(baseline) == 4

        for cmd in proposed:
            assert "+optimizer=smac20_proximity_lcb" in cmd
            assert "++optimizer.acq_func_kwargs.k=25" in cmd
            assert "++optimizer.acq_func_kwargs.level=0.95" in cmd
            assert "++optimizer.acq_func_kwargs.eps=0.16" in cmd
            assert "++optimizer.smac_cfg.model_kwargs.uncertainty_func=proximity_b" in cmd
            assert "++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda=1.345" in cmd
            assert "task.optimization_resources.n_trials=100" in cmd
            assert f"baserundir=runs/sweep_{suite}_proximity" in cmd
            assert f"results/sweep_{suite}_proximity/telemetry_SMAC20_ProximityLCB_" in cmd

        for cmd in baseline:
            assert "+optimizer/smac20=hpo" in cmd
            assert "++optimizer.acq_func_name=lcb" in cmd
            assert "++optimizer.acq_func_kwargs.beta=3.8416" in cmd
            assert "++optimizer.acq_func_kwargs.update_beta=false" in cmd
            assert "task.optimization_resources.n_trials=100" in cmd
            assert f"baserundir=runs/sweep_{suite}_proximity" in cmd
            assert f"results/sweep_{suite}_proximity/telemetry_SMAC3_HPOFacade_lcb_" in cmd

    def test_chunk_tasks_stay_under_limit(self):
        items = [f"cmd_{i}" for i in range(7140)]
        chunks = chunk_tasks(items, max_chunk_size=2500)
        assert len(chunks) == 3
        # Verify all stay strictly under max_chunk_size and 5000 limit
        for c in chunks:
            assert len(c) <= 2500
            assert len(c) <= 5000
        # Verify union is identical and in order
        flat = [item for c in chunks for item in c]
        assert flat == items

    def test_generate_sbatch_content_luis_compliant(self):
        content = generate_sbatch_content(
            suite="yahpo_rbv2_ranger",
            task_file="results/sweep_yahpo_rbv2_ranger_proximity/tasks.txt",
            log_dir="results/sweep_yahpo_rbv2_ranger_proximity/slurm_logs",
            partition="ai",
        )
        assert "#SBATCH --job-name=rngr_prox" in content
        # Crucial: Must NOT contain any hardcoded --array > 300
        assert "--array" not in content or "%" not in content or int(content.split("--array=")[1].split("-")[1].split("%")[0]) <= 300
        assert "TASK_FILE=\"results/sweep_yahpo_rbv2_ranger_proximity/tasks.txt\"" in content
        assert "scripts/run_carps_patched.py" in content
        assert "results/sweep_yahpo_rbv2_ranger_proximity/slurm_logs" in content

    def test_generate_all_suite_artifacts_luis_compliant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                summary = generate_all_suite_artifacts(
                    suites=["hpobench_ml"],
                    seeds=1,
                    trials=10,
                    chunk_size=200,
                )
                info = summary["hpobench_ml"]
                # Submit scripts should be in scripts/
                assert info["submit_sh"].startswith("scripts/")
                assert info["sbatch_file"].startswith("scripts/")
                assert os.path.isfile(info["submit_sh"])
                assert os.path.isfile(info["sbatch_file"])

                # Verify launcher script adheres to LUIS <= 300 chunking
                with open(info["submit_sh"], "r", encoding="utf-8") as f:
                    sh_content = f.read()
                assert "CHUNK_SIZE=200" in sh_content or "CHUNK_SIZE=250" in sh_content
                assert "sbatch --parsable --array=" in sh_content

                # Tasks should be in results/
                assert info["master_file"].startswith("results/")
                assert os.path.isfile(info["master_file"])

                # Results dir should NOT contain any .sbatch or .sh
                suite_results_dir = Path("results/sweep_hpobench_ml_proximity")
                assert not list(suite_results_dir.glob("*.sbatch"))
                assert not list(suite_results_dir.glob("*.sh"))
            finally:
                os.chdir(orig_cwd)


