"""Unit and Integration Test Suite for CARP-S BBsubset Held-Out Test Evaluation Suite.

Verifies:
1. CarpsBBSubsetRegistry.get_test_tasks() returns exactly 20 held-out test tasks.
2. Task generator generates exactly 1,200 lines with 30 seeds paired (1..30) for both optimizers.
3. Parameter overrides for Tuned Proximity LCB and SMAC3 baseline:
   - SMAC20_ProximityLCB_tuned: k=25, decay_lambda=1.345, eps=0.1678, level=0.95, uncertainty_func=proximity_b
   - SMAC3_HPOFacade_lcb: beta=3.8416 (kappa=1.96)
4. SLURM sbatch and bash submission scripts respect LUIS cluster limits (MaxArraySize <= 200, %25 concurrency).
5. Gathering script and Wilcoxon + Holm-Bonferroni statistical analysis with normalized regret scorecard export.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


class TestCarpsBBSubsetRegistryTestTasks(unittest.TestCase):
    def test_get_test_tasks_count_and_composition(self):
        """Should return exactly 20 held-out test tasks matching the CARP-S BBsubset test suite."""
        test_tasks = CarpsBBSubsetRegistry.get_test_tasks()
        self.assertEqual(len(test_tasks), 20)

        # Expected 8 BBOB tasks
        expected_bbob = [
            "subset_bbob_2_6_1",
            "subset_bbob_2_9_0",
            "subset_bbob_2_12_2",
            "subset_bbob_8_22_0",
            "subset_bbob_16_1_1",
            "subset_bbob_16_11_0",
            "subset_bbob_32_9_0",
            "subset_bbob_32_11_0",
        ]
        # Expected 1 HPOBench task
        expected_hpobench = [
            "subset_hpobench_blackbox_tabular_ml_svm_12",
        ]
        # Expected 11 YAHPO tasks
        expected_yahpo = [
            "subset_yahpo_lcbench_167184_None",
            "subset_yahpo_rbv2_glmnet_32_None",
            "subset_yahpo_rbv2_glmnet_375_None",
            "subset_yahpo_rbv2_ranger_29_None",
            "subset_yahpo_rbv2_rpart_18_None",
            "subset_yahpo_rbv2_rpart_4534_None",
            "subset_yahpo_rbv2_svm_1493_None",
            "subset_yahpo_rbv2_xgboost_1457_None",
            "subset_yahpo_rbv2_xgboost_1493_None",
            "subset_yahpo_rbv2_xgboost_1510_None",
            "subset_yahpo_rbv2_xgboost_41027_None",
        ]

        expected_all = expected_bbob + expected_hpobench + expected_yahpo
        self.assertEqual(len(expected_all), 20)

        for task_name in expected_all:
            matched = [t for t in test_tasks if task_name in t]
            self.assertEqual(len(matched), 1, f"Task {task_name} should appear exactly once in test tasks")
            self.assertTrue(
                matched[0].startswith("+task=subselection/blackbox/test/"),
                f"Task prefix invalid: {matched[0]}",
            )

        # Ensure all tasks are test tasks, not dev tasks
        for t in test_tasks:
            self.assertIn("subselection/blackbox/test/", t)
            self.assertNotIn("subselection/blackbox/dev/", t)


class TestGenerateBBSUBSETTestProximityTasks(unittest.TestCase):
    def test_generate_tasks_1200_lines_and_pairing(self):
        """Should generate exactly 1,200 lines for 20 tasks * 30 seeds * 2 optimizers."""
        from scripts.generate_bbsubset_test_proximity_tasks import (
            generate_bbsubset_test_proximity_tasks,
        )

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            lines = generate_bbsubset_test_proximity_tasks(
                output_file=tmp_path,
                runs_dir="results/sweep_bbsubset_test_proximity",
                baserundir="runs/sweep_bbsubset_test_proximity",
                seeds=30,
                trials=100,
            )
            self.assertEqual(len(lines), 1200)

            # Check file contents
            with open(tmp_path, "r") as f:
                saved_lines = [l.strip() for l in f if l.strip()]
            self.assertEqual(len(saved_lines), 1200)

            # Check 600 runs per optimizer
            tuned_lines = [l for l in lines if "SMAC20_ProximityLCB_tuned" in l]
            baseline_lines = [l for l in lines if "SMAC3_HPOFacade_lcb" in l]
            self.assertEqual(len(tuned_lines), 600)
            self.assertEqual(len(baseline_lines), 600)

            # Check trials=100
            for l in lines:
                self.assertIn("task.optimization_resources.n_trials=100", l)
                self.assertIn("baserundir=runs/sweep_bbsubset_test_proximity", l)

            # Verify seed pairing (1..30) for each of the 20 test tasks
            test_tasks = CarpsBBSubsetRegistry.get_test_tasks()
            for t in test_tasks:
                task_name = t.split("/")[-1]
                p_seeds = sorted([
                    int(l.split("seed=")[1].split()[0])
                    for l in tuned_lines if task_name in l
                ])
                b_seeds = sorted([
                    int(l.split("seed=")[1].split()[0])
                    for l in baseline_lines if task_name in l
                ])
                self.assertEqual(p_seeds, list(range(1, 31)))
                self.assertEqual(b_seeds, list(range(1, 31)))

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_parameter_overrides(self):
        """Verify tuned parameter overrides and baseline settings."""
        from scripts.generate_bbsubset_test_proximity_tasks import (
            generate_bbsubset_test_proximity_tasks,
        )

        lines = generate_bbsubset_test_proximity_tasks(
            output_file=None,
            seeds=1,
            trials=100,
        )
        # 20 tasks * 1 seed * 2 optimizers = 40 lines
        self.assertEqual(len(lines), 40)

        tuned_lines = [l for l in lines if "SMAC20_ProximityLCB_tuned" in l]
        baseline_lines = [l for l in lines if "SMAC3_HPOFacade_lcb" in l]

        for l in tuned_lines:
            self.assertIn("+optimizer=smac20_proximity_lcb", l)
            self.assertIn("++optimizer.acq_func_kwargs.k=25", l)
            self.assertIn("++optimizer.acq_func_kwargs.level=0.95", l)
            self.assertIn("++optimizer.acq_func_kwargs.eps=0.1678", l)
            self.assertIn("++optimizer.smac_cfg.model_kwargs.uncertainty_func=proximity_b", l)
            self.assertIn("++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda=1.345", l)
            self.assertIn("optimizer_id=SMAC20_ProximityLCB_tuned", l)
            self.assertIn("optimizer_container_id=SMAC20_ProximityLCB", l)

        for l in baseline_lines:
            self.assertIn("+optimizer/smac20=hpo", l)
            self.assertIn("++optimizer.acq_func_name=lcb", l)
            self.assertIn("++optimizer.acq_func_kwargs.beta=3.8416", l)
            self.assertIn("optimizer_id=SMAC3_HPOFacade_lcb", l)
            self.assertIn("optimizer_container_id=SMAC3_HPOFacade", l)


class TestSlurmSubmissionScripts(unittest.TestCase):
    def test_sbatch_and_launcher_scripts(self):
        """Array tasks must be chunked into <= 200 tasks with %25 concurrency."""
        sbatch_path = Path(PROJECT_ROOT) / "scripts/submit_bbsubset_test_proximity_array.sbatch"
        all_sh_path = Path(PROJECT_ROOT) / "scripts/submit_bbsubset_test_proximity_all.sh"

        self.assertTrue(sbatch_path.exists(), f"Missing: {sbatch_path}")
        self.assertTrue(all_sh_path.exists(), f"Missing: {all_sh_path}")

        sbatch_content = sbatch_path.read_text()
        all_sh_content = all_sh_path.read_text()

        self.assertIn("run_carps_patched.py", sbatch_content)
        self.assertIn("tasks.txt", sbatch_content)
        self.assertIn("%25", all_sh_content)
        self.assertIn("CHUNK_SIZE=200", all_sh_content)
        self.assertIn("sbatch", all_sh_content)
        self.assertIn("submit_bbsubset_test_proximity_array.sbatch", all_sh_content)


class TestGatherBBSUBSETTestProximity(unittest.TestCase):
    def test_gather_script_exists_and_callable(self):
        """Gather script must exist and provide gather_bbsubset_test_proximity_data."""
        gather_path = Path(PROJECT_ROOT) / "scripts/gather_bbsubset_test_proximity.py"
        self.assertTrue(gather_path.exists(), f"Missing: {gather_path}")
        from scripts.gather_bbsubset_test_proximity import gather_bbsubset_test_proximity_data
        self.assertTrue(callable(gather_bbsubset_test_proximity_data))


class TestComputeBBSUBSETTestProximityAnalysis(unittest.TestCase):
    def test_holm_bonferroni_adjustment(self):
        """Verify Holm-Bonferroni step-down adjustment logic."""
        from scripts.compute_bbsubset_test_proximity_analysis import apply_holm_bonferroni

        raw_p = [0.001, 0.01, 0.03, 0.04, 0.05]
        adj_p = apply_holm_bonferroni(raw_p)

        self.assertEqual(len(adj_p), len(raw_p))
        # Holm adjustment:
        # rank 1: 0.001 * 5 = 0.005
        # rank 2: 0.01 * 4 = 0.04
        # rank 3: 0.03 * 3 = 0.09
        # rank 4: 0.04 * 2 = 0.08 -> max(0.09, 0.08) = 0.09 (monotonic)
        # rank 5: 0.05 * 1 = 0.05 -> max(0.09, 0.05) = 0.09 (monotonic)
        self.assertAlmostEqual(adj_p[0], 0.005, places=5)
        self.assertAlmostEqual(adj_p[1], 0.04, places=5)
        self.assertAlmostEqual(adj_p[2], 0.09, places=5)
        self.assertAlmostEqual(adj_p[3], 0.09, places=5)
        self.assertAlmostEqual(adj_p[4], 0.09, places=5)

        # Monotonicity test
        for i in range(len(adj_p) - 1):
            self.assertLessEqual(adj_p[i], adj_p[i + 1])

    def test_compute_analysis_end_to_end(self):
        """Verify end-to-end analysis on synthetic multi-task, 30-seed dataset."""
        from scripts.compute_bbsubset_test_proximity_analysis import (
            compute_bbsubset_test_proximity_analysis,
        )

        test_tasks = CarpsBBSubsetRegistry.get_test_tasks()
        records = []

        # Synthetic benchmark: Tuned Proximity LCB outperforms Baseline on most tasks
        for t_idx, task_arg in enumerate(test_tasks):
            task_name = task_arg.split("/")[-1]
            for seed in range(1, 31):
                # Proposed: better cost
                cost_p = 5.0 + t_idx * 0.5 - 0.02 * seed
                # Baseline: higher cost
                cost_b = 6.0 + t_idx * 0.5 - 0.01 * seed

                records.append({
                    "task_id": task_name,
                    "optimizer_id": "SMAC20_ProximityLCB_tuned",
                    "seed": seed,
                    "n_trials": 100,
                    "trial_value__cost_inc": cost_p,
                })
                records.append({
                    "task_id": task_name,
                    "optimizer_id": "SMAC3_HPOFacade_lcb",
                    "seed": seed,
                    "n_trials": 100,
                    "trial_value__cost_inc": cost_b,
                })

        df = pd.DataFrame(records)

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, "logs.parquet")
            df.to_parquet(parquet_path)

            summary = compute_bbsubset_test_proximity_analysis(
                input_file=parquet_path,
                output_dir=tmpdir,
                proposed_id="SMAC20_ProximityLCB_tuned",
                baseline_id="SMAC3_HPOFacade_lcb",
                alpha=0.05,
            )

            self.assertEqual(summary["n_tasks"], 20)
            self.assertEqual(summary["total_paired_runs"], 600)
            self.assertGreater(summary["overall_wins"], 0)
            self.assertEqual(summary["overall_losses"], 0)

            # Verify scorecard outputs
            md_scorecard = Path(tmpdir) / "test_proximity_scorecard.md"
            csv_scorecard = Path(tmpdir) / "test_proximity_scorecard.csv"
            self.assertTrue(md_scorecard.exists())
            self.assertTrue(csv_scorecard.exists())
            self.assertGreater(md_scorecard.stat().st_size, 100)
            self.assertGreater(csv_scorecard.stat().st_size, 100)

            # Check normalized regret columns exist in scorecard
            scorecard_df = pd.read_csv(csv_scorecard)
            self.assertEqual(len(scorecard_df), 20)
            self.assertIn("norm_regret_proposed", scorecard_df.columns)
            self.assertIn("norm_regret_baseline", scorecard_df.columns)
            self.assertIn("p_raw", scorecard_df.columns)
            self.assertIn("p_holm", scorecard_df.columns)
            self.assertIn("cliffs_delta", scorecard_df.columns)

            # Statistical rigor: task-level Wilcoxon and valid Cliff's delta aggregation
            self.assertIn("task_wilcoxon_stat", summary)
            self.assertIn("task_wilcoxon_p_twosided", summary)
            self.assertIn("task_wilcoxon_p_onesided", summary)
            self.assertIn("mean_cliffs_delta", summary)
            self.assertIn("task_level_cliffs_delta", summary)
            self.assertIn("stratified_analysis", summary)
            self.assertIn("bbob_high_d_16", summary["stratified_analysis"])
            self.assertIn("suite_high_d_8", summary["stratified_analysis"])

            # Verify markdown scorecard includes task-level test and high-D stratification
            with open(md_scorecard) as f:
                md_text = f.read()
            self.assertIn("Task-level Wilcoxon (Demšar)", md_text)
            self.assertIn("High-Dimensional Stratification", md_text)

    def test_compute_analysis_min_dim_filter(self):
        """Verify min_dim filtering creates targeted scorecard for high-D tasks."""
        from scripts.compute_bbsubset_test_proximity_analysis import (
            compute_bbsubset_test_proximity_analysis,
        )

        test_tasks = CarpsBBSubsetRegistry.get_test_tasks()
        records = []

        for t_idx, task_arg in enumerate(test_tasks):
            task_name = task_arg.split("/")[-1]
            for seed in range(1, 31):
                cost_p = 5.0 + t_idx * 0.5 - 0.02 * seed
                cost_b = 6.0 + t_idx * 0.5 - 0.01 * seed
                records.append({
                    "task_id": task_name,
                    "optimizer_id": "SMAC20_ProximityLCB_tuned",
                    "seed": seed,
                    "n_trials": 100,
                    "trial_value__cost_inc": cost_p,
                })
                records.append({
                    "task_id": task_name,
                    "optimizer_id": "SMAC3_HPOFacade_lcb",
                    "seed": seed,
                    "n_trials": 100,
                    "trial_value__cost_inc": cost_b,
                })

        df = pd.DataFrame(records)

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, "logs.parquet")
            df.to_parquet(parquet_path)

            summary_high_d = compute_bbsubset_test_proximity_analysis(
                input_file=parquet_path,
                output_dir=tmpdir,
                proposed_id="SMAC20_ProximityLCB_tuned",
                baseline_id="SMAC3_HPOFacade_lcb",
                min_dim=8,
                out_prefix="test_proximity_scorecard_high_d",
            )

            self.assertEqual(summary_high_d["n_tasks"], 10)
            md_high_d = Path(tmpdir) / "test_proximity_scorecard_high_d.md"
            csv_high_d = Path(tmpdir) / "test_proximity_scorecard_high_d.csv"
            self.assertTrue(md_high_d.exists())
            self.assertTrue(csv_high_d.exists())


if __name__ == "__main__":
    unittest.main()
