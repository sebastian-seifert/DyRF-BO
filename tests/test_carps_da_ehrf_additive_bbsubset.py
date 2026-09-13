"""Comprehensive Unit, Contract, and Integration Test Suite for CARP-S DA-EHRF Additive Uncertainty on BBsubset.

Tests:
1. Hydra config instantiation for `dyrf_da_ehrf_additive_ei.yaml`.
2. AdditiveEpistemicAcquisition and WarmupCosineScheduler mathematical guarantees.
3. DA-EHRF Tree-Path integration inside CARPSDynamicRFOptimizer with telemetry logging.
4. Task generator producing exactly 400 tasks with strict seed pairing across 20 dev tasks.
5. SLURM cluster execution pipeline (sbatch script and chunked submission script).
6. CARP-S data gathering and Wilcoxon signed-rank analysis scripts.
7. End-to-end 3-trial CARP-S smoke test with `scripts/run_carps_patched.py`.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ConfigSpace import ConfigurationSpace, Float

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from carps.utils.task import OptimizationResources
from carps.utils.trials import TrialInfo, TrialValue
from carps_integration.optimizer import CARPSDynamicRFOptimizer
from carps_integration.acquisitions import (
    ExpectedImprovement,
    WarmupCosineScheduler,
    AdditiveEpistemicAcquisition,
    normalize_max_relative,
    AcquisitionRegistry
)
from ep_extractors.distance_evidential import DistanceAwareEvidentialExtractor
from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


# =====================================================================
# Mock Objects for Fast Local Testing
# =====================================================================

class MockObjectiveFunction:
    def __init__(self, cs: ConfigurationSpace):
        self.configspace = cs

    def __call__(self, config) -> float:
        x1 = config["x1"]
        x2 = config["x2"]
        return float(x1 ** 2 + x2 ** 2 + 0.1 * np.sin(5.0 * x1))


class MockTask:
    def __init__(self, name="mock_bbsubset_dev_task", seed=42, n_trials=50):
        cs = ConfigurationSpace(seed=seed)
        cs.add([
            Float("x1", bounds=(-5.0, 5.0), default=0.0),
            Float("x2", bounds=(-5.0, 5.0), default=0.0),
        ])
        self.name = name
        self.seed = seed
        self.objective_function = MockObjectiveFunction(cs)
        self.optimization_resources = OptimizationResources(n_trials=n_trials, time_budget=None)


# =====================================================================
# 1. Hydra Config Instantiation Tests
# =====================================================================

class TestHydraConfigInstantiation:
    """Verifies Hydra configuration schema, instantiation, and parameter values for dyrf_da_ehrf_additive_ei."""

    def test_config_file_exists_and_parses(self):
        config_path = os.path.join(
            PROJECT_ROOT, "carps_integration", "configs", "optimizer", "dyrf_da_ehrf_additive_ei.yaml"
        )
        assert os.path.exists(config_path), f"Config file not found: {config_path}"
        cfg = OmegaConf.load(config_path)
        assert cfg.optimizer_id == "CARPSDynamicRF_DAEHRF_AdditiveEI"
        assert cfg.optimizer_container_id == "CARPSDynamicRF_DAEHRF_AdditiveEI"
        assert cfg.optimizer.extractor_name == "distance_evidential"
        assert cfg.optimizer.acq_mode == "additive_epistemic"
        assert cfg.optimizer.acq_uncertainty_type == "epistemic"
        assert cfg.optimizer.acq_func_name == "ei"
        assert np.isclose(cfg.optimizer.beta_max, 1.0)
        assert np.isclose(cfg.optimizer.beta_min, 0.0)
        assert np.isclose(cfg.optimizer.warmup_ratio, 0.20)
        assert cfg.optimizer.enable_adaptation is False
        assert cfg.optimizer.n_init == 10

        # DA-EHRF extractor kwargs
        ext_kwargs = cfg.optimizer.extractor_kwargs
        assert ext_kwargs.spatial_metric == "tree_path"
        assert np.isclose(ext_kwargs.tree_decay_lambda, 3.0)
        assert np.isclose(ext_kwargs.kappa_leaf, 1.0)
        assert np.isclose(ext_kwargs.c_spatial, 1.0)

    def test_hydra_compose_resolution(self):
        configs_dir = os.path.join(PROJECT_ROOT, "carps_integration", "configs")
        with initialize_config_dir(config_dir=configs_dir, version_base="1.3"):
            cfg = compose(
                config_name="optimizer/dyrf_da_ehrf_additive_ei",
                overrides=[
                    "++seed=1",
                    "++task.optimization_resources.n_trials=50",
                ],
            )
            assert cfg.optimizer_id == "CARPSDynamicRF_DAEHRF_AdditiveEI"
            assert cfg.optimizer.extractor_name == "distance_evidential"
            assert cfg.optimizer.extractor_kwargs.spatial_metric == "tree_path"
            assert cfg.optimizer.beta_max == 1.0
            assert cfg.task.optimization_resources.n_trials == 50


# =====================================================================
# 2. Additive Acquisition & Warmup Cosine Scheduler Math Tests
# =====================================================================

class TestAdditiveMathAndScheduler:
    """Verifies mathematical guarantees and boundaries for WarmupCosineScheduler and AdditiveEpistemicAcquisition."""

    def test_warmup_cosine_scheduler_schedule(self):
        scheduler = WarmupCosineScheduler(
            total_trials=50,
            warmup_ratio=0.20,
            beta_max=1.0,
            beta_min=0.0
        )
        assert scheduler.t_warmup == 10

        # Warmup phase: strictly beta_max
        for t in range(0, 11):
            assert np.isclose(scheduler.get_beta(t), 1.0), f"Warmup failed at t={t}"

        # Midpoint cosine decay at t=30: progress = (30-10)/(50-10) = 20/40 = 0.5 -> cos(pi/2)=0 -> 0.5
        assert np.isclose(scheduler.get_beta(30), 0.5, atol=1e-6)

        # Terminal budget at t=50: beta_min = 0.0
        assert np.isclose(scheduler.get_beta(50), 0.0, atol=1e-6)

        # Beyond budget: remains clamped to beta_min
        assert np.isclose(scheduler.get_beta(60), 0.0, atol=1e-6)

        # Monotonicity check post-warmup
        betas = [scheduler.get_beta(t) for t in range(10, 51)]
        assert all(betas[i] >= betas[i + 1] - 1e-12 for i in range(len(betas) - 1))

    def test_max_relative_normalization(self):
        # Normal array
        arr = np.array([2.0, 5.0, 10.0, 1.0])
        normed = normalize_max_relative(arr)
        assert np.isclose(np.max(normed), 1.0, atol=1e-6)
        assert np.isclose(normed[2], 1.0, atol=1e-6)
        assert np.isclose(normed[1], 0.5, atol=1e-6)
        assert np.all(normed >= 0.0)

        # All zeros or negatives
        zero_arr = np.zeros(5)
        assert np.all(normalize_max_relative(zero_arr) == 0.0)
        neg_arr = np.array([-5.0, -1.0])
        assert np.all(normalize_max_relative(neg_arr) == 0.0)

    def test_additive_epistemic_acquisition_computation(self):
        base_ei = ExpectedImprovement(xi=0.0)
        additive_acq = AdditiveEpistemicAcquisition(base_acq=base_ei)

        preds = np.array([1.0, 2.0, 0.5])
        unc_tot = np.array([0.5, 0.1, 0.8])
        u_ep = np.array([0.2, 0.9, 0.1])
        y_best = 0.8

        # When beta_t = 0, score is strictly normalized base EI
        scores_beta0 = additive_acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=0.0)
        raw_base = base_ei.compute(preds, unc_tot, y_best)
        expected_beta0 = normalize_max_relative(raw_base)
        np.testing.assert_allclose(scores_beta0, expected_beta0, atol=1e-6)

        # When beta_t = 1.0, scores strictly incorporate normalized u_ep
        scores_beta1 = additive_acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=1.0)
        norm_ep = normalize_max_relative(u_ep)
        np.testing.assert_allclose(scores_beta1, expected_beta0 + norm_ep, atol=1e-6)


# =====================================================================
# 3. DA-EHRF Tree-Path Integration in CARPSDynamicRFOptimizer Tests
# =====================================================================

class TestDAEHRFTreePathCARPSOptimizerIntegration:
    """Verifies that CARPSDynamicRFOptimizer properly instantiates DistanceAwareEvidentialExtractor with tree_path metric."""

    def test_optimizer_initial_design_and_bo_step(self, tmp_path):
        task = MockTask(n_trials=10)
        telemetry_path = str(tmp_path / "telemetry_test.json")

        optimizer = CARPSDynamicRFOptimizer(
            task=task,
            extractor_name="distance_evidential",
            extractor_kwargs={
                "spatial_metric": "tree_path",
                "tree_decay_lambda": 3.0,
                "kappa_leaf": 1.0,
                "c_spatial": 1.0,
            },
            acq_mode="additive_epistemic",
            acq_uncertainty_type="epistemic",
            acq_func_name="ei",
            beta_max=1.0,
            beta_min=0.0,
            warmup_ratio=0.20,
            enable_adaptation=False,
            n_init=3,
            total_trials=10,
            telemetry_path=telemetry_path,
        )

        # Run 3 initial design trials
        for step in range(3):
            trial_info = optimizer.ask()
            cost = task.objective_function(trial_info.config)
            optimizer.tell(trial_info, TrialValue(cost=cost, virtual_time=1.0))

        assert len(optimizer.history) == 3
        assert optimizer.surrogate is not None
        assert isinstance(optimizer.surrogate.extractor, DistanceAwareEvidentialExtractor)
        assert optimizer.surrogate.extractor.spatial_metric == "tree_path"
        assert np.isclose(optimizer.surrogate.extractor.tree_decay_lambda, 3.0)

        # Step 4: First Bayesian Optimization step (acq_mode == 'additive_epistemic')
        trial_bo = optimizer.ask()
        assert trial_bo is not None
        cost_bo = task.objective_function(trial_bo.config)
        optimizer.tell(trial_bo, TrialValue(cost=cost_bo, virtual_time=1.0))

        assert len(optimizer.history) == 4
        assert os.path.exists(telemetry_path)

        with open(telemetry_path, "r") as f:
            telem = json.load(f)

        assert telem["extractor_name"] == "distance_evidential"
        assert len(telem["trials"]) == 4
        # Check telemetry logged valid beta_t
        last_trial = telem["trials"][-1]
        assert "beta_t" in last_trial
        assert 0.0 <= last_trial["beta_t"] <= 1.0


# =====================================================================
# 4. Task Generator Tests
# =====================================================================

class TestTaskGeneratorBbsubset:
    """Verifies task generation logic, 400 total task count, strict seed pairing, and CLI parsing."""

    def test_generate_bbsubset_tasks_output_and_pairing(self, tmp_path):
        from scripts.generate_bbsubset_da_ehrf_additive_tasks import generate_bbsubset_da_ehrf_additive_tasks

        output_txt = str(tmp_path / "tasks.txt")
        tasks = generate_bbsubset_da_ehrf_additive_tasks(
            output_path=output_txt,
            seeds=list(range(1, 11)),
            trials=50,
            beta_max=1.0,
            beta_min=0.0,
            warmup_ratio=0.20,
        )

        assert os.path.exists(output_txt)
        with open(output_txt, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        # 20 dev tasks * 10 seeds = 200 proposed
        # 20 dev tasks * 10 seeds = 200 baseline
        # Total = 400 tasks
        assert len(lines) == 400
        assert len(tasks) == 400

        dev_tasks = CarpsBBSubsetRegistry.get_dev_tasks()
        assert len(dev_tasks) == 20

        proposed_lines = [l for l in lines if "+optimizer=dyrf_da_ehrf_additive_ei" in l]
        baseline_lines = [l for l in lines if "+optimizer/smac20=hpo" in l]

        assert len(proposed_lines) == 200
        assert len(baseline_lines) == 200

        # Strict seed pairing check: for every task and seed 1..10, both proposed and baseline exist
        for task_spec in dev_tasks:
            for seed in range(1, 11):
                prop_match = [
                    l for l in proposed_lines if task_spec in l and f"seed={seed} " in l
                ]
                base_match = [
                    l for l in baseline_lines if task_spec in l and f"seed={seed} " in l
                ]
                assert len(prop_match) == 1, f"Missing or duplicate proposed task for {task_spec} seed {seed}"
                assert len(base_match) == 1, f"Missing or duplicate baseline task for {task_spec} seed {seed}"

        # Syntactic checks
        for l in lines:
            assert "--config-dir carps_integration/configs" in l
            assert "task.optimization_resources.n_trials=50" in l
            assert "telemetry_path=" in l

    def test_task_generator_cli_execution(self, tmp_path):
        out_file = str(tmp_path / "cli_tasks.txt")
        script = os.path.join(PROJECT_ROOT, "scripts", "generate_bbsubset_da_ehrf_additive_tasks.py")
        cmd = [
            sys.executable,
            script,
            "--output", out_file,
            "--seeds", "1,2",
            "--trials", "20",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        assert res.returncode == 0, f"Task generator failed:\n{res.stderr}"
        assert os.path.exists(out_file)
        lines = [l.strip() for l in open(out_file) if l.strip()]
        # 20 tasks * 2 seeds * 2 approaches = 80 tasks
        assert len(lines) == 80


# =====================================================================
# 5. SLURM Cluster Pipeline Tests
# =====================================================================

class TestSLURMClusterPipeline:
    """Verifies shell script syntax, LUIS MaxArraySize chunking logic (200), and sbatch configuration."""

    def test_sbatch_script_syntax_and_directives(self):
        sbatch_path = os.path.join(PROJECT_ROOT, "scripts", "submit_bbsubset_da_ehrf_additive_array.sbatch")
        assert os.path.exists(sbatch_path), f"File not found: {sbatch_path}"

        # Syntax check via bash -n
        res = subprocess.run(["bash", "-n", sbatch_path], capture_output=True, text=True)
        assert res.returncode == 0, f"Bash syntax error in sbatch script:\n{res.stderr}"

        content = open(sbatch_path).read()
        assert "#SBATCH -p ai" in content
        assert "#SBATCH --cpus-per-task=8" in content
        assert "#SBATCH --mem=16G" in content
        assert "#SBATCH --time=12:00:00" in content
        assert "export PYTHONPATH=." in content
        assert "results/bbsubset_da_ehrf_additive_tasks.txt" in content
        assert "run_carps_patched.py" in content

    def test_submission_script_chunking_logic(self):
        sh_path = os.path.join(PROJECT_ROOT, "scripts", "submit_bbsubset_da_ehrf_additive_all.sh")
        assert os.path.exists(sh_path), f"File not found: {sh_path}"

        res = subprocess.run(["bash", "-n", sh_path], capture_output=True, text=True)
        assert res.returncode == 0, f"Bash syntax error in submission shell script:\n{res.stderr}"

        content = open(sh_path).read()
        assert "CHUNK_SIZE=200" in content
        assert "submit_bbsubset_da_ehrf_additive_array.sbatch" in content

    def test_chunking_arithmetic_simulation(self):
        total_tasks = 400
        chunk_size = 200
        chunks = []
        for start in range(1, total_tasks + 1, chunk_size):
            end = min(start + chunk_size - 1, total_tasks)
            chunks.append((start, end))

        assert len(chunks) == 2
        assert chunks[0] == (1, 200)
        assert chunks[1] == (201, 400)


# =====================================================================
# 6. CARP-S Data Gathering and Wilcoxon Analysis Tests
# =====================================================================

class TestAnalysisScripts:
    """Verifies gathering wrapper script and Wilcoxon statistical analysis pipeline."""

    def test_gather_script_exists_and_parses_flags(self):
        script_path = os.path.join(PROJECT_ROOT, "scripts", "gather_bbsubset_da_ehrf_additive_carps.py")
        assert os.path.exists(script_path), f"Gather script not found: {script_path}"
        content = open(script_path).read()
        assert "CARPSDynamicRF_DAEHRF_AdditiveEI" in content
        assert "SMAC3_HPOFacade" in content
        assert "filelogs_to_df" in content

    def test_wilcoxon_analysis_script_computations(self, tmp_path):
        from scripts.compute_bbsubset_da_ehrf_additive_wilcoxon import compute_bbsubset_da_ehrf_additive_wilcoxon

        # Create synthetic normalized logs with 20 tasks, 10 seeds
        np.random.seed(42)
        records = []
        tasks = [f"task_{i}" for i in range(20)]
        for task in tasks:
            for seed in range(1, 11):
                # Baseline cost
                base_cost = np.random.uniform(0.3, 0.8)
                # Proposed cost with slight improvement on most tasks
                prop_cost = base_cost - np.random.uniform(-0.02, 0.08)

                records.append({
                    "task_id": task,
                    "seed": seed,
                    "optimizer_id": "SMAC3_HPOFacade_ei",
                    "n_trials": 50,
                    "trial_value__cost_inc_norm": base_cost,
                })
                records.append({
                    "task_id": task,
                    "seed": seed,
                    "optimizer_id": "CARPSDynamicRF_DAEHRF_AdditiveEI",
                    "n_trials": 50,
                    "trial_value__cost_inc_norm": prop_cost,
                })

        df = pd.DataFrame(records)
        input_csv = str(tmp_path / "synthetic_normalized_logs.csv")
        df.to_csv(input_csv, index=False)

        output_dir = str(tmp_path / "stats_out")
        df_result = compute_bbsubset_da_ehrf_additive_wilcoxon(
            input_file=input_csv,
            output_dir=output_dir,
            baseline_id="SMAC3_HPOFacade_ei",
            proposed_id="CARPSDynamicRF_DAEHRF_AdditiveEI",
        )

        assert df_result is not None
        assert len(df_result) == 1
        assert "Wilcoxon W" in df_result.columns
        assert "p_val" in df_result.columns
        assert "Cliff's delta" in df_result.columns
        assert "Win / Loss / Tie" in df_result.columns

        # Verify output files generated
        assert os.path.exists(os.path.join(output_dir, "wilcoxon_results.md"))
        assert os.path.exists(os.path.join(output_dir, "wilcoxon_results.csv"))
        assert os.path.exists(os.path.join(output_dir, "wilcoxon_results.tex"))


# =====================================================================
# 7. CARP-S Smoke Execution Test
# =====================================================================

class TestCARPSSmokeExecution:
    """Executes a 3-trial CARP-S run with run_carps_patched.py using dyrf_da_ehrf_additive_ei."""

    def test_carps_runner_smoke_execution(self, tmp_path):
        run_dir = str(tmp_path / "smoke_run")
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "run_carps_patched.py"),
            "--config-dir", os.path.join(PROJECT_ROOT, "carps_integration", "configs"),
            "+optimizer=dyrf_da_ehrf_additive_ei",
            "++optimizer.n_init=2",
            "+task/Noisy/bbob=cfg_ackley_2d_gaussian",
            "task.optimization_resources.n_trials=3",
            "seed=1",
            f"outdir={run_dir}",
            f"hydra.run.dir={run_dir}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        assert res.returncode == 0, f"Run failed with stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

        log_file = Path(run_dir) / "trial_logs.jsonl"
        assert log_file.exists(), f"Log file {log_file} was not generated."
        lines = [line.strip() for line in log_file.read_text().splitlines() if line.strip()]
        assert len(lines) >= 3, f"Expected >= 3 logged trials, found {len(lines)}."
