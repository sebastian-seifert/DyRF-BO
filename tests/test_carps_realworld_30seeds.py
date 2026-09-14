"""Unit and Integration Test Suite for CARP-S 30-Seed Real-World ML Benchmark Suite.

Covers:
1. CarpsBBSubsetRegistry extension with REALWORLD_ML_DEV_TASKS and get_realworld_dev_tasks().
2. Backward compatibility of DEV_TASKS and get_dev_tasks().
3. Task generator generate_bbsubset_realworld_30seeds_tasks producing 840 runs with strict pairing.
4. SLURM array and launcher scripts syntax and chunking logic.
5. Aggregation script gather_bbsubset_realworld_30seeds.
6. Statistical analysis compute_bbsubset_realworld_30seeds_wilcoxon on trial_value__cost_inc.
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


class TestCarpsRegistryRealworldDevTasks:
    """Verifies that the registry isolates real-world ML tasks while preserving the full BBsubset."""

    def test_whole_bbsubset_dev_tasks_preserved(self):
        """CRITICAL: DEV_TASKS and get_dev_tasks() must remain 100% backward compatible (20 tasks)."""
        dev_tasks = CarpsBBSubsetRegistry.get_dev_tasks()
        assert isinstance(dev_tasks, list)
        assert len(dev_tasks) == 20, f"Expected 20 dev tasks in full BBsubset, found {len(dev_tasks)}"
        assert len(CarpsBBSubsetRegistry.DEV_TASKS) == 20

        # Verify BBOB tasks are present in DEV_TASKS
        bbob_tasks = [t for t in dev_tasks if "bbob" in t]
        assert len(bbob_tasks) == 4, f"Expected 4 BBOB tasks in full BBsubset, found {len(bbob_tasks)}"

    def test_realworld_ml_dev_tasks_count(self):
        """Real-world ML tasks must contain exactly 14 tasks (11 YAHPO + 3 HPOBench ML)."""
        assert hasattr(CarpsBBSubsetRegistry, "REALWORLD_ML_DEV_TASKS"), "Missing REALWORLD_ML_DEV_TASKS attribute"
        ml_tasks = CarpsBBSubsetRegistry.REALWORLD_ML_DEV_TASKS
        assert len(ml_tasks) == 14, f"Expected 14 real-world ML dev tasks, found {len(ml_tasks)}"

        getter_tasks = CarpsBBSubsetRegistry.get_realworld_dev_tasks(include_nas=False)
        assert len(getter_tasks) == 14
        assert getter_tasks == ml_tasks

    def test_realworld_dev_tasks_include_nas(self):
        """When include_nas=True, the 2 NAS tasks are included, totaling 16 tasks."""
        tasks_with_nas = CarpsBBSubsetRegistry.get_realworld_dev_tasks(include_nas=True)
        assert len(tasks_with_nas) == 16, f"Expected 16 tasks with NAS, found {len(tasks_with_nas)}"
        nas_tasks = [t for t in tasks_with_nas if "nas" in t.lower() or "naval" in t.lower() or "slice" in t.lower()]
        assert len(nas_tasks) == 2, f"Expected 2 NAS tasks, found {len(nas_tasks)}"

    def test_excluded_bbob_and_nas_tasks_in_default(self):
        """Default realworld dev tasks must strictly exclude the 4 BBOB tasks and 2 NAS tasks."""
        ml_tasks = CarpsBBSubsetRegistry.get_realworld_dev_tasks(include_nas=False)
        for task in ml_tasks:
            assert "bbob" not in task, f"BBOB task found in realworld ML tasks: {task}"
            assert "NavalPropulsionBenchmark" not in task, f"NavalPropulsion found in realworld ML tasks: {task}"
            assert "SliceLocalizationBenchmark" not in task, f"SliceLocalization found in realworld ML tasks: {task}"

    def test_task_composition_breakdown(self):
        """Verify exact breakdown: 11 YAHPO tasks and 3 HPOBench ML tasks."""
        ml_tasks = CarpsBBSubsetRegistry.get_realworld_dev_tasks(include_nas=False)
        yahpo_tasks = [t for t in ml_tasks if "yahpo" in t]
        hpobench_tasks = [t for t in ml_tasks if "hpobench" in t]

        assert len(yahpo_tasks) == 11, f"Expected 11 YAHPO tasks, found {len(yahpo_tasks)}"
        assert len(hpobench_tasks) == 3, f"Expected 3 HPOBench ML tasks, found {len(hpobench_tasks)}"

    def test_task_yaml_files_exist(self):
        """Every realworld task YAML file must exist on disk."""
        tasks = CarpsBBSubsetRegistry.get_realworld_dev_tasks(include_nas=True)
        config_dir = os.path.join(PROJECT_ROOT, "carps_integration", "configs", "task")

        for task_str in tasks:
            assert task_str.startswith("+task=subselection/blackbox/dev/"), f"Invalid task format: {task_str}"
            task_rel = task_str.split("+task=")[-1] + ".yaml"
            yaml_path = os.path.join(config_dir, task_rel)
            assert os.path.exists(yaml_path), f"Missing task YAML file: {yaml_path}"

    def test_no_test_set_exposure(self):
        """Ensure no held-out test tasks are exposed."""
        tasks = CarpsBBSubsetRegistry.get_realworld_dev_tasks(include_nas=True)
        for task_str in tasks:
            assert "/test/" not in task_str, f"Test set task exposed in realworld dev registry: {task_str}"


class TestTaskGeneratorRealworld30Seeds:
    """Verifies task generation logic, 840 total runs (14 tasks * 30 seeds * 2 optimizers), and CLI flags."""

    def test_generate_bbsubset_realworld_30seeds_tasks(self, tmp_path):
        from scripts.generate_bbsubset_realworld_30seeds_tasks import (
            generate_bbsubset_realworld_30seeds_tasks,
        )

        output_txt = str(tmp_path / "bbsubset_realworld_30seeds_tasks.txt")
        lines = generate_bbsubset_realworld_30seeds_tasks(
            output_path=output_txt,
            seeds=list(range(1, 31)),
            trials=50,
            include_nas=False,
        )

        assert os.path.exists(output_txt)
        with open(output_txt, "r") as f:
            file_lines = [l.strip() for l in f if l.strip()]

        # 14 tasks * 30 seeds * 2 optimizers = 840 runs
        assert len(file_lines) == 840
        assert len(lines) == 840

        proposed_lines = [l for l in lines if "+optimizer=dyrf_da_ehrf_additive_ei" in l]
        baseline_lines = [l for l in lines if "+optimizer=smac3_hpo_facade_ei" in l]

        assert len(proposed_lines) == 420, f"Expected 420 proposed runs, found {len(proposed_lines)}"
        assert len(baseline_lines) == 420, f"Expected 420 baseline runs, found {len(baseline_lines)}"

        # Strict seed pairing check
        realworld_tasks = CarpsBBSubsetRegistry.get_realworld_dev_tasks(include_nas=False)
        for task_spec in realworld_tasks:
            task_name = task_spec.split("/")[-1]
            for seed in range(1, 31):
                prop_match = [
                    l for l in proposed_lines if task_spec in l and f"seed={seed} " in l
                ]
                base_match = [
                    l for l in baseline_lines if task_spec in l and f"seed={seed} " in l
                ]
                assert len(prop_match) == 1, f"Missing or duplicate proposed task for {task_spec} seed {seed}"
                assert len(base_match) == 1, f"Missing or duplicate baseline task for {task_spec} seed {seed}"

                # Check proposed line specifics
                pline = prop_match[0]
                assert "++optimizer.beta_max=1.0" in pline
                assert "++optimizer.beta_min=0.0" in pline
                assert "++optimizer.warmup_ratio=0.20" in pline
                assert "optimizer_id=CARPSDynamicRF_DAEHRF_AdditiveEI" in pline
                assert "optimizer_container_id=CARPSDynamicRF_DAEHRF_AdditiveEI" in pline
                assert f"results/bbsubset_realworld_30seeds/proposed/telemetry_da_ehrf_additive_ei_{task_name}_seed{seed}.json" in pline
                assert "task.optimization_resources.n_trials=50" in pline

                # Check baseline line specifics
                bline = base_match[0]
                assert "optimizer_id=SMAC3_HPOFacade_ei" in bline
                assert "optimizer_container_id=SMAC3_HPOFacade_ei" in bline
                assert "task.optimization_resources.n_trials=50" in bline

    def test_include_nas_yields_960_runs(self, tmp_path):
        from scripts.generate_bbsubset_realworld_30seeds_tasks import (
            generate_bbsubset_realworld_30seeds_tasks,
        )

        output_txt = str(tmp_path / "nas_tasks.txt")
        lines = generate_bbsubset_realworld_30seeds_tasks(
            output_path=output_txt,
            seeds=list(range(1, 31)),
            trials=50,
            include_nas=True,
        )
        # 16 tasks * 30 seeds * 2 optimizers = 960 runs
        assert len(lines) == 960

    def test_cli_execution_flags(self, tmp_path):
        out_file = str(tmp_path / "cli_tasks.txt")
        script = os.path.join(PROJECT_ROOT, "scripts", "generate_bbsubset_realworld_30seeds_tasks.py")
        cmd = [
            sys.executable,
            script,
            "--output", out_file,
            "--seeds", "1,2,3",
            "--trials", "25",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        assert res.returncode == 0, f"Task generator CLI failed:\n{res.stderr}"
        assert os.path.exists(out_file)
        lines = [l.strip() for l in open(out_file) if l.strip()]
        # 14 tasks * 3 seeds * 2 optimizers = 84 runs
        assert len(lines) == 84
        for l in lines:
            assert "task.optimization_resources.n_trials=25" in l


class TestSLURMClusterPipeline:
    """Verifies bash syntax, sbatch directives, and chunking launcher logic."""

    def test_sbatch_script_syntax_and_directives(self):
        sbatch_path = os.path.join(PROJECT_ROOT, "scripts", "submit_bbsubset_realworld_30seeds_array.sbatch")
        assert os.path.exists(sbatch_path), f"File not found: {sbatch_path}"

        res = subprocess.run(["bash", "-n", sbatch_path], capture_output=True, text=True)
        assert res.returncode == 0, f"Bash syntax error in sbatch script:\n{res.stderr}"

        content = open(sbatch_path).read()
        assert "#SBATCH -p ai" in content
        assert "#SBATCH --cpus-per-task=8" in content
        assert "#SBATCH --mem=16G" in content
        assert "#SBATCH --time=12:00:00" in content
        assert "export PYTHONPATH=." in content
        assert "results/bbsubset_realworld_30seeds_tasks.txt" in content
        assert "run_carps_patched.py" in content

    def test_launcher_script_syntax_and_chunking(self):
        sh_path = os.path.join(PROJECT_ROOT, "scripts", "submit_bbsubset_realworld_30seeds_all.sh")
        assert os.path.exists(sh_path), f"File not found: {sh_path}"

        res = subprocess.run(["bash", "-n", sh_path], capture_output=True, text=True)
        assert res.returncode == 0, f"Bash syntax error in launcher script:\n{res.stderr}"

        content = open(sh_path).read()
        assert "CHUNK_SIZE=200" in content
        assert "submit_bbsubset_realworld_30seeds_array.sbatch" in content

    def test_chunking_arithmetic_for_840_tasks(self):
        total_tasks = 840
        chunk_size = 200
        chunks = []
        for start in range(1, total_tasks + 1, chunk_size):
            end = min(start + chunk_size - 1, total_tasks)
            chunks.append((start, end))

        assert len(chunks) == 5
        assert chunks[0] == (1, 200)
        assert chunks[1] == (201, 400)
        assert chunks[2] == (401, 600)
        assert chunks[3] == (601, 800)
        assert chunks[4] == (801, 840)


class TestGatherAndWilcoxonAnalysis:
    """Verifies data gathering wrapper and statistical Wilcoxon analysis on trial_value__cost_inc."""

    def test_gather_script_exists_and_parses(self):
        script_path = os.path.join(PROJECT_ROOT, "scripts", "gather_bbsubset_realworld_30seeds.py")
        assert os.path.exists(script_path), f"Gather script not found: {script_path}"
        content = open(script_path).read()
        assert "CARPSDynamicRF_DAEHRF_AdditiveEI" in content
        assert "SMAC3_HPOFacade_ei" in content
        assert "filelogs_to_df" in content

    def test_wilcoxon_analysis_computations_on_cost_inc(self, tmp_path):
        from scripts.compute_bbsubset_realworld_30seeds_wilcoxon import (
            compute_bbsubset_realworld_30seeds_wilcoxon,
        )

        np.random.seed(42)
        records = []
        tasks = [f"rw_task_{i}" for i in range(14)]
        for task in tasks:
            for seed in range(1, 31):
                base_cost_inc = np.random.uniform(0.1, 0.5)
                # Proposed has lower cost incumbent
                prop_cost_inc = base_cost_inc - np.random.uniform(0.005, 0.03)

                records.append({
                    "task_id": task,
                    "seed": seed,
                    "optimizer_id": "SMAC3_HPOFacade_ei",
                    "n_trials": 50,
                    "trial_value__cost_inc": base_cost_inc,
                })
                records.append({
                    "task_id": task,
                    "seed": seed,
                    "optimizer_id": "CARPSDynamicRF_DAEHRF_AdditiveEI",
                    "n_trials": 50,
                    "trial_value__cost_inc": prop_cost_inc,
                })

        df = pd.DataFrame(records)
        input_csv = str(tmp_path / "synthetic_realworld_30seeds_logs.csv")
        df.to_csv(input_csv, index=False)

        out_dir = str(tmp_path / "stats_tables")
        summary_df = compute_bbsubset_realworld_30seeds_wilcoxon(
            input_file=input_csv,
            output_dir=out_dir,
            baseline_id="SMAC3_HPOFacade_ei",
            proposed_id="CARPSDynamicRF_DAEHRF_AdditiveEI",
        )

        assert summary_df is not None
        assert len(summary_df) == 1
        assert "Wilcoxon W" in summary_df.columns
        assert "p_val" in summary_df.columns
        assert "Cliff's delta" in summary_df.columns
        assert "Win / Loss / Tie" in summary_df.columns

        # Verify output files
        assert os.path.exists(os.path.join(out_dir, "wilcoxon_results.md"))
        assert os.path.exists(os.path.join(out_dir, "wilcoxon_results.csv"))
        assert os.path.exists(os.path.join(out_dir, "wilcoxon_results.tex"))


class TestCARPSSmokeExecutionRealworld:
    """Smoke tests executing 2 trials with run_carps_patched.py for both proposed and baseline optimizers."""

    def test_baseline_smac3_smoke_execution(self, tmp_path):
        run_dir = str(tmp_path / "baseline_smoke")
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "run_carps_patched.py"),
            "--config-dir", os.path.join(PROJECT_ROOT, "carps_integration", "configs"),
            "+optimizer=smac3_hpo_facade_ei",
            "+task/Noisy/bbob=cfg_ackley_2d_gaussian",
            "task.optimization_resources.n_trials=2",
            "seed=1",
            f"outdir={run_dir}",
            f"hydra.run.dir={run_dir}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        assert res.returncode == 0, f"Baseline run failed with stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

        log_file = Path(run_dir) / "trial_logs.jsonl"
        assert log_file.exists(), f"Log file {log_file} was not generated."
        lines = [line.strip() for line in log_file.read_text().splitlines() if line.strip()]
        assert len(lines) >= 2, f"Expected >= 2 logged trials, found {len(lines)}."

    def test_proposed_dyrf_smoke_execution(self, tmp_path):
        run_dir = str(tmp_path / "proposed_smoke")
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "run_carps_patched.py"),
            "--config-dir", os.path.join(PROJECT_ROOT, "carps_integration", "configs"),
            "+optimizer=dyrf_da_ehrf_additive_ei",
            "++optimizer.n_init=2",
            "+task/Noisy/bbob=cfg_ackley_2d_gaussian",
            "task.optimization_resources.n_trials=2",
            "seed=1",
            f"outdir={run_dir}",
            f"hydra.run.dir={run_dir}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        assert res.returncode == 0, f"Proposed run failed with stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

        log_file = Path(run_dir) / "trial_logs.jsonl"
        assert log_file.exists(), f"Log file {log_file} was not generated."
        lines = [line.strip() for line in log_file.read_text().splitlines() if line.strip()]
        assert len(lines) >= 2, f"Expected >= 2 logged trials, found {len(lines)}."

