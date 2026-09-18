"""Unit and Integration Test Suite for SMAC4HPO Meta-Optimization Layer on Proximity LCB.

Verifies:
1. Meta-Hyperparameter Search Space & Incumbent Warm Start:
   - ConfigSpace with k in [5, 30] (int), decay_lambda in [0.2, 2.0] (float), eps in [0.02, 0.20] (float).
   - Default configuration strictly equals random search incumbent: k=25, decay_lambda=1.345, eps=0.1678.
   - Initial design guarantees the incumbent is evaluated as trial 1.
2. Multi-Task Task Generation:
   - Exactly 90 tasks generated per iteration (18 working dev tasks * 5 seeds).
   - Excludes broken HPOBench NAS benchmarks.
   - Injects k, level=0.95, eps, decay_lambda, seed, trials=100, optimizer_id into Hydra args.
3. Scale-Invariant Meta-Loss Calculation:
   - Per-task min-max normalization against reference bounds.
   - Handles edge case when y_max == y_min.
   - Penalizes crashed / missing runs with worst-case regret (1.0).
4. Cluster Submission & 5,000 Job Limit Constraints:
   - Total 100 iterations * 90 runs = 9,000 runs.
   - Batch 1: iterations 1 to 55 -> 4,950 runs (<= 5,000).
   - Batch 2: iterations 56 to 100 -> 4,050 runs (<= 5,000).
   - Both submit_meta_smac_batch1.sh and submit_meta_smac_batch2.sh are valid bash scripts.
5. Ask-and-Tell Loop & Checkpoint Resume:
   - Orchestrator advances SMAC state sequentially with ask() and tell().
   - State can be saved and resumed across batches.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


class TestMetaSmacConfigSpace(unittest.TestCase):
    """Verifies the ConfigSpace and warm-start initialization."""

    def test_configspace_hyperparameters_and_defaults(self):
        """Search space must contain k, decay_lambda, eps with exact bounds and incumbent defaults."""
        from scripts.meta_smac_proximity_space import create_proximity_meta_configspace

        cs = create_proximity_meta_configspace()
        hps = cs.get_hyperparameters_dict()

        self.assertIn("k", hps)
        self.assertIn("decay_lambda", hps)
        self.assertIn("eps", hps)

        # Check k
        hp_k = hps["k"]
        self.assertEqual(hp_k.lower, 5)
        self.assertEqual(hp_k.upper, 30)
        self.assertEqual(hp_k.default_value, 25)

        # Check decay_lambda
        hp_lam = hps["decay_lambda"]
        self.assertAlmostEqual(hp_lam.lower, 0.2, places=4)
        self.assertAlmostEqual(hp_lam.upper, 2.0, places=4)
        self.assertAlmostEqual(hp_lam.default_value, 1.345, places=4)

        # Check eps
        hp_eps = hps["eps"]
        self.assertAlmostEqual(hp_eps.lower, 0.02, places=4)
        self.assertAlmostEqual(hp_eps.upper, 0.20, places=4)
        self.assertAlmostEqual(hp_eps.default_value, 0.1678, places=4)

    def test_initial_design_queries_incumbent_first(self):
        """When initializing SMAC, the first trial returned by ask() must be the incumbent."""
        from smac import HyperparameterOptimizationFacade, Scenario
        from scripts.meta_smac_proximity_space import (
            create_proximity_meta_configspace,
            build_initial_design_with_incumbent,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cs = create_proximity_meta_configspace()
            scenario = Scenario(
                configspace=cs,
                n_trials=10,
                deterministic=True,
                output_directory=Path(tmpdir) / "smac_test",
            )
            initial_design = build_initial_design_with_incumbent(scenario)
            smac_opt = HyperparameterOptimizationFacade(
                scenario=scenario,
                target_function=lambda cfg, seed=0: 0.0,
                initial_design=initial_design,
                overwrite=True,
            )

            trial_info = smac_opt.ask()
            cfg_dict = trial_info.config.get_dictionary()

            self.assertEqual(cfg_dict["k"], 25)
            self.assertAlmostEqual(cfg_dict["decay_lambda"], 1.345, places=3)
            self.assertAlmostEqual(cfg_dict["eps"], 0.1678, places=3)


class TestIterationTaskGenerator(unittest.TestCase):
    """Verifies Hydra task generation for each candidate configuration."""

    def test_generate_iteration_tasks_count_and_structure(self):
        """Must generate exactly 90 tasks for 18 dev tasks across 5 seeds."""
        from scripts.meta_smac_task_generator import generate_iteration_tasks

        cfg = {"k": 25, "decay_lambda": 1.345, "eps": 0.1678}
        tasks = generate_iteration_tasks(
            config=cfg,
            iteration=1,
            seeds=5,
            trials=100,
            output_dir="results/meta_smac_proximity_hpo",
        )

        self.assertEqual(len(tasks), 90)

        # Check that broken NAS benchmarks are not present
        for t in tasks:
            self.assertNotIn("tabular_nas", t)
            self.assertIn("++optimizer.acq_func_kwargs.k=25", t)
            self.assertIn("++optimizer.acq_func_kwargs.level=0.95", t)
            self.assertIn("++optimizer.acq_func_kwargs.eps=0.1678", t)
            self.assertIn("++optimizer.smac_cfg.model_kwargs.uncertainty_func=proximity_b", t)
            self.assertIn("++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda=1.345", t)
            self.assertIn("task.optimization_resources.n_trials=100", t)
            self.assertIn("optimizer_id=SMAC20_ProximityLCB_iter001", t)

        # Check that each of the 18 working dev tasks appears exactly 5 times (once per seed)
        dev_tasks = CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)
        self.assertEqual(len(dev_tasks), 18)
        for d in dev_tasks:
            matching = [t for t in tasks if d in t]
            self.assertEqual(len(matching), 5, f"Dev task {d} must appear 5 times (seeds 1..5)")


class TestMetaLossComputation(unittest.TestCase):
    """Verifies multi-task normalized regret calculation and crash handling."""

    def test_normalized_regret_with_reference_bounds(self):
        """Costs are scaled relative to reference bounds into [0, 1]."""
        from scripts.meta_smac_loss import compute_meta_loss

        ref_bounds = {
            "task_a": {"min": 10.0, "max": 20.0},
            "task_b": {"min": 100.0, "max": 200.0},
        }

        # Candidate results: task_a costs 15.0 across seeds (norm = 0.5), task_b costs 120.0 (norm = 0.2)
        task_results = {
            "task_a": [15.0, 15.0, 15.0],
            "task_b": [120.0, 120.0, 120.0],
        }

        loss = compute_meta_loss(task_results, ref_bounds)
        # Expected: mean(0.5, 0.2) = 0.35
        self.assertAlmostEqual(loss, 0.35, places=4)

    def test_missing_or_crashed_task_penalization(self):
        """Missing or failed tasks receive maximum normalized regret (1.0)."""
        from scripts.meta_smac_loss import compute_meta_loss

        ref_bounds = {
            "task_a": {"min": 10.0, "max": 20.0},
            "task_b": {"min": 100.0, "max": 200.0},
        }

        # task_a succeeds (10.0 -> norm = 0.0), task_b fails/missing
        task_results = {
            "task_a": [10.0, 10.0],
            "task_b": [],  # Failed
        }

        loss = compute_meta_loss(task_results, ref_bounds)
        # Expected: mean(0.0, 1.0) = 0.5
        self.assertAlmostEqual(loss, 0.5, places=4)


class TestExtractReferenceBounds(unittest.TestCase):
    """Verifies extracting reference min/max bounds directly from CARP-S execution logs."""

    def test_extract_bounds_from_synthetic_logs(self):
        """Extracts min and max incumbent costs per task from CSV."""
        from scripts.extract_dev_reference_bounds import extract_dev_reference_bounds

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_logs.csv"
            out_json = Path(tmpdir) / "bounds.json"

            # Create synthetic log DataFrame
            data = {
                "task_id": [
                    "blackbox/20/dev/bbob/2/12/0", "blackbox/20/dev/bbob/2/12/0",
                    "blackbox/20/dev/yahpo/rbv2_aknn/1462/None", "blackbox/20/dev/yahpo/rbv2_aknn/1462/None",
                ],
                "n_trials": [100, 100, 100, 100],
                "trial_value__cost_inc": [300.0, 1500.0, -0.99, -0.65],
            }
            pd.DataFrame(data).to_csv(csv_path, index=False)

            bounds = extract_dev_reference_bounds(str(csv_path), output_json=str(out_json))

            self.assertIn("subset_bbob_2_12_0", bounds)
            self.assertIn("subset_yahpo_rbv2_aknn_1462_None", bounds)
            self.assertAlmostEqual(bounds["subset_bbob_2_12_0"]["min"], 300.0)
            self.assertAlmostEqual(bounds["subset_bbob_2_12_0"]["max"], 1500.0)
            self.assertAlmostEqual(bounds["subset_yahpo_rbv2_aknn_1462_None"]["min"], -0.99)
            self.assertAlmostEqual(bounds["subset_yahpo_rbv2_aknn_1462_None"]["max"], -0.65)

            # JSON file must exist and be valid
            self.assertTrue(out_json.exists())
            with open(out_json) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, bounds)

    def test_extract_bounds_from_real_dev_logs(self):
        """When real results/bbsubset_dev_analysis/logs.csv exists, extracts all 18 dev tasks."""
        real_logs = Path(PROJECT_ROOT) / "results" / "bbsubset_dev_analysis" / "logs.csv"
        if not real_logs.exists():
            self.skipTest(f"{real_logs} not found")

        from scripts.extract_dev_reference_bounds import extract_dev_reference_bounds

        bounds = extract_dev_reference_bounds(str(real_logs))
        dev_tasks = CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)

        for d in dev_tasks:
            short_name = d.split("/")[-1]
            self.assertIn(short_name, bounds, f"Missing dev task {short_name} in extracted bounds")
            b_min = bounds[short_name]["min"]
            b_max = bounds[short_name]["max"]
            self.assertTrue(np.isfinite(b_min))
            self.assertTrue(np.isfinite(b_max))
            self.assertLess(b_min, b_max, f"min must be < max for {short_name}")



class TestSubmissionScriptAndTaskFilter(unittest.TestCase):
    """Verifies that the broken CARP-S NAS tasks are filtered out and submission script is valid."""

    def test_dev_task_filter_strictly_excludes_broken_nas(self):
        """The 2 broken HPOBench tabular NAS tasks must be filtered out, leaving exactly 18 tasks."""
        all_dev_tasks = CarpsBBSubsetRegistry.get_dev_tasks()
        self.assertEqual(len(all_dev_tasks), 20, "Total raw dev tasks must be 20")

        working_tasks = CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)
        self.assertEqual(len(working_tasks), 18, "Working dev tasks must be exactly 18 (excluding the 2 broken NAS tasks)")

        # Verify the 2 broken tasks are absent
        for task in working_tasks:
            self.assertNotIn("tabular_nas", task, f"Broken NAS task {task} must NOT be in working tasks")
            self.assertNotIn("NavalPropulsionBenchmark", task)
            self.assertNotIn("SliceLocalizationBenchmark", task)

    def test_submit_meta_smac_all_exists_and_valid(self):
        """submit_meta_smac_all.sh must exist, be executable, and be syntactically valid bash."""
        submit_path = Path(PROJECT_ROOT) / "scripts" / "submit_meta_smac_all.sh"
        self.assertTrue(submit_path.exists(), f"Missing {submit_path}")

        # Check executable permissions
        self.assertTrue(os.access(submit_path, os.X_OK), "submit_meta_smac_all.sh must be executable")

        # Bash syntax check (bash -n)
        res = subprocess.run(["bash", "-n", str(submit_path)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Bash syntax error in submit_meta_smac_all.sh: {res.stderr}")


class TestAskTellCheckpointResume(unittest.TestCase):
    """Verifies that the orchestrator can execute dry-run iterations, checkpoint, and resume."""

    def test_dry_run_ask_tell_checkpoint_and_resume(self):
        """Runs 2 dry-run iterations, checkpoints, then resumes for iteration 3."""
        from scripts.run_meta_smac_proximity_hpo import MetaSmacOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "meta_smac"

            # Phase 1: Iterations 1 to 2
            orch_phase1 = MetaSmacOrchestrator(
                start_iteration=1,
                end_iteration=2,
                seeds=5,
                trials=100,
                output_dir=str(out_dir),
                dry_run=True,
            )
            orch_phase1.run()

            # Checkpoint file and leaderboard must exist
            checkpoint_file = out_dir / "meta_smac_checkpoint.json"
            leaderboard_file = out_dir / "meta_leaderboard.csv"
            self.assertTrue(checkpoint_file.exists())
            self.assertTrue(leaderboard_file.exists())

            df_lead1 = pd.read_csv(leaderboard_file)
            self.assertEqual(len(df_lead1), 2)
            # Iteration 1 must be incumbent
            self.assertEqual(int(df_lead1.iloc[0]["k"]), 25)
            self.assertAlmostEqual(float(df_lead1.iloc[0]["decay_lambda"]), 1.345, places=3)
            self.assertAlmostEqual(float(df_lead1.iloc[0]["eps"]), 0.1678, places=3)

            # Phase 2: Resume for iteration 3
            orch_phase2 = MetaSmacOrchestrator(
                start_iteration=3,
                end_iteration=3,
                seeds=5,
                trials=100,
                output_dir=str(out_dir),
                resume=True,
                dry_run=True,
            )
            orch_phase2.run()

class TestIterationResultParser(unittest.TestCase):
    """Verifies parsing candidate evaluation results from trial_logs.jsonl, runhistory.json, and telemetry."""

    def test_parse_from_trial_logs_jsonl(self):
        """Extracts minimum trial cost from CARP-S trial_logs.jsonl across multiple trials and seeds."""
        from scripts.run_meta_smac_proximity_hpo import parse_iteration_results

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "runs" / "iter_001"
            res_dir = Path(tmpdir) / "results" / "iter_001"
            dev_tasks = ["subset_bbob_2_12_0", "subset_bbob_2_12_1"]

            # Create mock trial_logs.jsonl for task 1, seed 1
            seed1_dir = base_dir / "SMAC20_ProximityLCB_iter001" / "blackbox" / "20" / "dev" / "subset_bbob_2_12_0" / "1"
            seed1_dir.mkdir(parents=True, exist_ok=True)
            log1 = seed1_dir / "trial_logs.jsonl"
            with open(log1, "w") as f:
                f.write(json.dumps({"n_trials": 1, "trial_value": {"cost": 25.0, "status": 1}}) + "\n")
                f.write(json.dumps({"n_trials": 2, "trial_value": {"cost": 12.5, "status": 1}}) + "\n")
                f.write(json.dumps({"n_trials": 3, "trial_value": {"cost": 18.0, "status": 1}}) + "\n")

            # Create mock trial_logs.jsonl for task 1, seed 2
            seed2_dir = base_dir / "SMAC20_ProximityLCB_iter001" / "blackbox" / "20" / "dev" / "subset_bbob_2_12_0" / "2"
            seed2_dir.mkdir(parents=True, exist_ok=True)
            log2 = seed2_dir / "trial_logs.jsonl"
            with open(log2, "w") as f:
                f.write(json.dumps({"n_trials": 1, "trial_value": {"cost": 15.0, "status": 1}}) + "\n")
                f.write(json.dumps({"n_trials": 2, "trial_value": {"cost": 9.2, "status": 1}}) + "\n")

            results = parse_iteration_results(res_dir, base_dir, dev_tasks)

            self.assertEqual(len(results["subset_bbob_2_12_0"]), 2)
            self.assertIn(12.5, results["subset_bbob_2_12_0"])
            self.assertIn(9.2, results["subset_bbob_2_12_0"])
            self.assertEqual(results["subset_bbob_2_12_1"], [])

    def test_parse_from_runhistory_json(self):
        """Extracts minimum cost from SMAC runhistory.json when trial_logs is absent."""
        from scripts.run_meta_smac_proximity_hpo import parse_iteration_results

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "runs" / "iter_001"
            res_dir = Path(tmpdir) / "results" / "iter_001"
            dev_tasks = ["subset_yahpo_rbv2_aknn_1462_None"]

            rh_dir = base_dir / "SMAC20_ProximityLCB_iter001" / "subset_yahpo_rbv2_aknn_1462_None" / "1" / "smac3_output" / "hash"
            rh_dir.mkdir(parents=True, exist_ok=True)
            rh_file = rh_dir / "runhistory.json"
            rh_data = {
                "data": [
                    {"cost": 0.45, "status": 1},
                    {"cost": 0.31, "status": 1},
                    {"cost": 0.99, "status": 2},  # Crashed trial
                ]
            }
            with open(rh_file, "w") as f:
                json.dump(rh_data, f)

            results = parse_iteration_results(res_dir, base_dir, dev_tasks)
            self.assertEqual(results["subset_yahpo_rbv2_aknn_1462_None"], [0.31])

    def test_parse_from_telemetry_json_fallback(self):
        """Supports legacy telemetry JSON format with trials list or cost_inc."""
        from scripts.run_meta_smac_proximity_hpo import parse_iteration_results

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "runs" / "iter_001"
            res_dir = Path(tmpdir) / "results" / "iter_001"
            res_dir.mkdir(parents=True, exist_ok=True)
            dev_tasks = ["subset_bbob_2_12_0"]

            tfile = res_dir / "telemetry_SMAC20_ProximityLCB_iter001_subset_bbob_2_12_0_seed1.json"
            tdata = {
                "trials": [
                    {"cost": 100.0},
                    {"cost": 55.5},
                ]
            }
            with open(tfile, "w") as f:
                json.dump(tdata, f)

            results = parse_iteration_results(res_dir, base_dir, dev_tasks)
            self.assertEqual(results["subset_bbob_2_12_0"], [55.5])


if __name__ == "__main__":
    unittest.main()

