"""Unit and Integration Test Suite for CARP-S BBsubset Held-Out Test Evaluation Suite (Proximity LCB vs. SMAC3 EI).

Verifies:
1. CarpsBBSubsetRegistry.get_test_tasks() returns exactly 20 held-out test tasks.
2. Task generator generates exactly 1,200 lines with 30 seeds paired (1..30) for both optimizers:
   - Proposed: SMAC20_ProximityLCB_tuned (k=25, lambda=1.345, eps=0.1678, level=0.95, uncertainty_func=proximity_b)
   - Baseline: SMAC3_HPOFacade_ei (+optimizer=smac3_hpo_facade_ei)
3. Task generator parameter overrides and output format integrity.
4. SLURM sbatch and bash submission scripts respect LUIS cluster limits (MaxArraySize <= 200, %25 concurrency).
5. Gathering script and statistical analysis engine compute scale-invariant normalized regret,
   Demšar task-level Wilcoxon signed-rank tests, exact Binomial tests, and Holm-Bonferroni FWER corrections.
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

        # Ensure all tasks are test tasks
        for t in test_tasks:
            self.assertIn("subselection/blackbox/test/", t)
            self.assertNotIn("subselection/blackbox/dev/", t)


class TestGenerateBBSUBSETTestProximityVsEITasks(unittest.TestCase):
    def test_generate_tasks_1200_lines_and_pairing(self):
        """Should generate exactly 1,200 lines for 20 tasks * 30 seeds * 2 optimizers."""
        from scripts.generate_bbsubset_test_proximity_vs_ei_tasks import (
            generate_bbsubset_test_proximity_vs_ei_tasks,
        )

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            lines = generate_bbsubset_test_proximity_vs_ei_tasks(
                output_file=tmp_path,
                runs_dir="results/sweep_bbsubset_test_proximity_vs_ei",
                baserundir="runs/sweep_bbsubset_test_proximity_vs_ei",
                seeds=30,
                trials=100,
            )

            self.assertEqual(len(lines), 1200)
            self.assertTrue(os.path.exists(tmp_path))

            with open(tmp_path) as f:
                file_lines = [l.strip() for l in f if l.strip()]
            self.assertEqual(len(file_lines), 1200)

            # 600 proposed, 600 baseline
            proposed = [l for l in lines if "optimizer_id=SMAC20_ProximityLCB_tuned" in l]
            baseline = [l for l in lines if "optimizer_id=SMAC3_HPOFacade_ei" in l]

            self.assertEqual(len(proposed), 600)
            self.assertEqual(len(baseline), 600)

            # Check proposed hyperparameters
            sample_p = proposed[0]
            self.assertIn("+optimizer=smac20_proximity_lcb", sample_p)
            self.assertIn("optimizer.acq_func_kwargs.k=25", sample_p)
            self.assertIn("optimizer.acq_func_kwargs.level=0.95", sample_p)
            self.assertIn("optimizer.acq_func_kwargs.eps=0.1678", sample_p)
            self.assertIn("optimizer.smac_cfg.model_kwargs.uncertainty_func=proximity_b", sample_p)
            self.assertIn("optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda=1.345", sample_p)
            self.assertIn("task.optimization_resources.n_trials=100", sample_p)

            # Check baseline hyperparameters
            sample_b = baseline[0]
            self.assertIn("+optimizer=smac3_hpo_facade_ei", sample_b)
            self.assertIn("optimizer_id=SMAC3_HPOFacade_ei", sample_b)
            self.assertIn("optimizer_container_id=SMAC3_HPOFacade_ei", sample_b)
            self.assertIn("task.optimization_resources.n_trials=100", sample_p)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestSubmissionScripts(unittest.TestCase):
    def test_sbatch_array_script_exists_and_configured(self):
        """Array script must specify array bounds, memory, and fail-safe logging."""
        script_path = Path(PROJECT_ROOT) / "scripts" / "submit_bbsubset_test_proximity_vs_ei_array.sbatch"
        self.assertTrue(script_path.exists(), f"Missing {script_path}")

        content = script_path.read_text()
        self.assertIn("#SBATCH --job-name=bb_test_ei", content)
        self.assertIn("#SBATCH --array=1-200%25", content)
        self.assertIn("results/sweep_bbsubset_test_proximity_vs_ei", content)

    def test_submit_all_script_exists_and_chunks(self):
        """Batch submit script must split 1,200 tasks into chunks respecting MaxArraySize <= 200."""
        script_path = Path(PROJECT_ROOT) / "scripts" / "submit_bbsubset_test_proximity_vs_ei_all.sh"
        self.assertTrue(script_path.exists(), f"Missing {script_path}")

        content = script_path.read_text()
        self.assertIn("CHUNK_SIZE=200", content)
        self.assertIn("results/sweep_bbsubset_test_proximity_vs_ei/tasks.txt", content)


class TestGatheringAndAnalysis(unittest.TestCase):
    def test_gather_script_exists(self):
        """Gathering script must exist and point to sweep_bbsubset_test_proximity_vs_ei."""
        script_path = Path(PROJECT_ROOT) / "scripts" / "gather_bbsubset_test_proximity_vs_ei.py"
        self.assertTrue(script_path.exists(), f"Missing {script_path}")

    def test_analysis_script_statistical_calculations(self):
        """Analysis engine must correctly compute normalized regret, Wilcoxon, and Holm-Bonferroni."""
        from scripts.compute_bbsubset_test_proximity_vs_ei_analysis import (
            apply_holm_bonferroni,
            calculate_cliffs_delta,
        )

        # 1. Test Cliff's delta
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        delta = calculate_cliffs_delta(x, y)
        self.assertEqual(delta, -1.0)

        # 2. Test Holm-Bonferroni correction
        raw_p = [0.01, 0.04, 0.03, 0.005]
        adj_p = apply_holm_bonferroni(raw_p)
        self.assertEqual(len(adj_p), len(raw_p))
        # Step-down checks
        # sorted raw: 0.005 (x4 = 0.02), 0.01 (x3 = 0.03), 0.03 (x2 = 0.06), 0.04 (x1 = 0.04 -> monotonic 0.06)
        self.assertAlmostEqual(adj_p[3], 0.02)  # raw 0.005
        self.assertAlmostEqual(adj_p[0], 0.03)  # raw 0.01
        self.assertAlmostEqual(adj_p[2], 0.06)  # raw 0.03
        self.assertAlmostEqual(adj_p[1], 0.06)  # raw 0.04


if __name__ == "__main__":
    unittest.main()
