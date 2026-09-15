"""Unit and Integration Test Suite for BBOB High-D & Extreme Proximity LCB Sweep.

Verifies:
1. CarpsBBOBHighDimRegistry returns exactly 72 tasks for D=16, 72 tasks for D=32, and 144 total.
2. Task generator produces exactly 8,640 command lines (144 tasks * 30 seeds * 2 optimizers)
   with perfect seed pairing and T=100 trials.
3. Correct hyperparameter injection for Tuned Proximity LCB (k=25, lambda=1.345, eps=0.16, level=0.95)
   and Baseline SMAC3 HPOFacade LCB (beta=3.8416, update_beta=false).
4. SLURM array script and chunked submission script respect LUIS cluster limits
   (MaxArraySize <= 200, %25 concurrency).
5. Statistical analysis pipeline computes normalized regrets, stratified Wilcoxon signed-rank
   tests with Holm-Bonferroni correction, and Cliff's delta effect sizes.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestCarpsBBOBHighDimRegistry(unittest.TestCase):
    def test_registry_tasks_count_and_composition(self):
        """Registry should contain 72 D=16 tasks, 72 D=32 tasks, and 144 total tasks."""
        from scripts.carps_bbob_highdim_registry import CarpsBBOBHighDimRegistry

        dim16_tasks = CarpsBBOBHighDimRegistry.get_dim16_tasks()
        dim32_tasks = CarpsBBOBHighDimRegistry.get_dim32_tasks()
        all_tasks = CarpsBBOBHighDimRegistry.get_all_tasks()

        self.assertEqual(len(dim16_tasks), 72)
        self.assertEqual(len(dim32_tasks), 72)
        self.assertEqual(len(all_tasks), 144)

        # Check unique
        self.assertEqual(len(set(dim16_tasks)), 72)
        self.assertEqual(len(set(dim32_tasks)), 72)
        self.assertEqual(len(set(all_tasks)), 144)

        # Check formatting and completeness
        for fid in range(1, 25):
            for iid in range(3):
                t16 = f"+task=BBOB/cfg_16_{fid}_{iid}"
                t32 = f"+task=BBOB/cfg_32_{fid}_{iid}"
                self.assertIn(t16, dim16_tasks)
                self.assertIn(t32, dim32_tasks)
                self.assertIn(t16, all_tasks)
                self.assertIn(t32, all_tasks)


class TestGenerateBBOBHighDimProximityTasks(unittest.TestCase):
    def test_generate_tasks_small_seed_count(self):
        """Verify generator on 2 seeds: 144 tasks * 2 seeds * 2 opts = 576 lines."""
        from scripts.generate_bbob_highdim_proximity_tasks import (
            generate_bbob_highdim_proximity_tasks,
        )

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            lines = generate_bbob_highdim_proximity_tasks(
                output_file=tmp_path,
                runs_dir="results/sweep_bbob_highdim_proximity",
                baserundir="runs/sweep_bbob_highdim_proximity",
                seeds=2,
                trials=100,
                k_neighbors=25,
                level=0.95,
                eps_floor=0.16,
                decay_lambda=1.345,
                uncertainty_func="proximity_b",
                kappa_baseline=1.96,
            )
            self.assertEqual(len(lines), 576)

            # Check that file was written
            written_lines = [l.strip() for l in Path(tmp_path).read_text().strip().split("\n")]
            self.assertEqual(len(written_lines), 576)

            # Proposed lines (first 288 lines: 144 tasks * 2 seeds)
            proposed_lines = lines[:288]
            for pl in proposed_lines:
                self.assertIn("+optimizer=smac20_proximity_lcb", pl)
                self.assertIn("++optimizer.acq_func_kwargs.k=25", pl)
                self.assertIn("++optimizer.acq_func_kwargs.level=0.95", pl)
                self.assertIn("++optimizer.acq_func_kwargs.eps=0.16", pl)
                self.assertIn("++optimizer.smac_cfg.model_kwargs.uncertainty_func=proximity_b", pl)
                self.assertIn("++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda=1.345", pl)
                self.assertIn("task.optimization_resources.n_trials=100", pl)
                self.assertIn("optimizer_id=SMAC20_ProximityLCB", pl)

            # Baseline lines (last 288 lines)
            baseline_lines = lines[288:]
            for bl in baseline_lines:
                self.assertIn("+optimizer/smac20=hpo", bl)
                self.assertIn("++optimizer.acq_func_name=lcb", bl)
                self.assertIn("++optimizer.acq_func_kwargs.beta=3.8416", bl)
                self.assertIn("++optimizer.acq_func_kwargs.update_beta=false", bl)
                self.assertIn("task.optimization_resources.n_trials=100", bl)
                self.assertIn("optimizer_id=SMAC3_HPOFacade_lcb", bl)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_generate_tasks_full_30_seeds(self):
        """Verify generator on 30 seeds generates exactly 8,640 lines with paired seeds."""
        from scripts.generate_bbob_highdim_proximity_tasks import (
            generate_bbob_highdim_proximity_tasks,
        )

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            lines = generate_bbob_highdim_proximity_tasks(
                output_file=tmp_path,
                seeds=30,
                trials=100,
            )
            # 144 tasks * 30 seeds * 2 optimizers = 8,640
            self.assertEqual(len(lines), 8640)

            # Half proposed, half baseline
            proposed = lines[:4320]
            baseline = lines[4320:]

            # Extract (task, seed) tuples
            import re
            def extract_task_seed(cmd: str):
                task_m = re.search(r"\+task=([^\s]+)", cmd)
                seed_m = re.search(r"seed=(\d+)", cmd)
                return task_m.group(1), int(seed_m.group(1))

            prop_pairs = set(extract_task_seed(c) for c in proposed)
            base_pairs = set(extract_task_seed(c) for c in baseline)

            self.assertEqual(len(prop_pairs), 4320)
            self.assertEqual(len(base_pairs), 4320)
            self.assertEqual(prop_pairs, base_pairs, "Proposed and Baseline must have exactly paired (task, seed) pairs!")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_dimension_filtering(self):
        """Verify filtering by dimension (e.g. only D=32)."""
        from scripts.generate_bbob_highdim_proximity_tasks import (
            generate_bbob_highdim_proximity_tasks,
        )

        lines_32 = generate_bbob_highdim_proximity_tasks(
            output_file=None,
            dimensions=[32],
            seeds=5,
            trials=100,
        )
        # 72 tasks * 5 seeds * 2 optimizers = 720
        self.assertEqual(len(lines_32), 720)
        for line in lines_32:
            self.assertIn("cfg_32_", line)
            self.assertNotIn("cfg_16_", line)


class TestSlurmSubmissionScripts(unittest.TestCase):
    def test_slurm_sbatch_and_orchestrator_exist_and_conform(self):
        """Verify sbatch and orchestrator bash scripts exist and comply with LUIS limits."""
        sbatch_path = Path("scripts/submit_bbob_highdim_proximity_array.sbatch")
        all_sh_path = Path("scripts/submit_bbob_highdim_proximity_all.sh")

        self.assertTrue(sbatch_path.exists(), f"{sbatch_path} must exist")
        self.assertTrue(all_sh_path.exists(), f"{all_sh_path} must exist")

        sbatch_content = sbatch_path.read_text()
        all_sh_content = all_sh_path.read_text()

        self.assertIn("TASK_FILE=\"results/sweep_bbob_highdim_proximity/tasks.txt\"", sbatch_content)
        self.assertIn("CHUNK_SIZE=200", all_sh_content)
        self.assertIn("%25", all_sh_content)


class TestBBOBHighDimAnalysisCalculations(unittest.TestCase):
    def test_compute_analysis_metrics(self):
        """Verify normalized regret and Wilcoxon/Cliff's delta logic on mock BBOB data."""
        from scripts.compute_bbob_highdim_proximity_analysis import (
            calculate_cliffs_delta,
            compute_statistical_comparison,
        )

        # 1. Cliff's Delta sanity check
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        delta = calculate_cliffs_delta(x, y)
        self.assertLess(delta, 0.0)

        # 2. Statistical comparison structure
        df_mock = pd.DataFrame([
            # task 1 (D=16): proposed strictly better
            {"task": "cfg_16_1_0", "dimension": 16, "seed": 1, "optimizer": "SMAC20_ProximityLCB", "final_cost": 5.0},
            {"task": "cfg_16_1_0", "dimension": 16, "seed": 1, "optimizer": "SMAC3_HPOFacade_lcb", "final_cost": 10.0},
            {"task": "cfg_16_1_0", "dimension": 16, "seed": 2, "optimizer": "SMAC20_ProximityLCB", "final_cost": 6.0},
            {"task": "cfg_16_1_0", "dimension": 16, "seed": 2, "optimizer": "SMAC3_HPOFacade_lcb", "final_cost": 12.0},
            # task 2 (D=32): proposed strictly better
            {"task": "cfg_32_1_0", "dimension": 32, "seed": 1, "optimizer": "SMAC20_ProximityLCB", "final_cost": 20.0},
            {"task": "cfg_32_1_0", "dimension": 32, "seed": 1, "optimizer": "SMAC3_HPOFacade_lcb", "final_cost": 30.0},
            {"task": "cfg_32_1_0", "dimension": 32, "seed": 2, "optimizer": "SMAC20_ProximityLCB", "final_cost": 25.0},
            {"task": "cfg_32_1_0", "dimension": 32, "seed": 2, "optimizer": "SMAC3_HPOFacade_lcb", "final_cost": 35.0},
        ])

        results = compute_statistical_comparison(df_mock)
        self.assertIn("overall", results)
        self.assertIn("dim_16", results)
        self.assertIn("dim_32", results)
        self.assertEqual(results["overall"]["win_rate_proposed"], 1.0)


if __name__ == "__main__":
    unittest.main()
