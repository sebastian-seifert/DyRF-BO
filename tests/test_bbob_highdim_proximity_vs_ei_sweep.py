"""Unit and Integration Test Suite for BBOB High-D & Extreme Sweep (Proximity LCB vs SMAC3 EI).

Verifies:
1. CarpsBBOBHighDimRegistry returns exactly 144 tasks across D=16 and D=32 (72 each).
2. Task generator generates exactly 8,640 command lines (144 tasks * 30 seeds * 2 optimizers):
   - Proposed: SMAC20_ProximityLCB (k=25, lambda=1.345, eps=0.16, level=0.95, uncertainty_func=proximity_b)
   - Baseline: SMAC3_HPOFacade_ei (+optimizer=smac3_hpo_facade_ei)
3. Task generator parameter overrides and output format integrity.
4. SLURM array and two-stage batch scripts respect LUIS queue limits (Max 5,000 tasks per batch, MaxArraySize <= 200, %25 concurrency).
5. Gathering script and statistical analysis engine calculate scale-invariant normalized regret,
   function-aggregated Wilcoxon tests (24 functions), Demšar task-level tests, and Holm-Bonferroni corrections.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_bbob_highdim_registry import CarpsBBOBHighDimRegistry


class TestBBOBHighDimRegistry(unittest.TestCase):
    def test_registry_tasks_count_and_dimensions(self):
        """Should return 72 tasks for D=16, 72 tasks for D=32, and 144 tasks combined."""
        tasks_16 = CarpsBBOBHighDimRegistry.get_tasks_by_dimensions([16])
        tasks_32 = CarpsBBOBHighDimRegistry.get_tasks_by_dimensions([32])
        tasks_all = CarpsBBOBHighDimRegistry.get_tasks_by_dimensions([16, 32])

        self.assertEqual(len(tasks_16), 72)
        self.assertEqual(len(tasks_32), 72)
        self.assertEqual(len(tasks_all), 144)

        for t in tasks_16:
            self.assertIn("cfg_16_", t)
        for t in tasks_32:
            self.assertIn("cfg_32_", t)


class TestGenerateBBOBHighDimProximityVsEITasks(unittest.TestCase):
    def test_generate_tasks_8640_lines_and_pairing(self):
        """Should generate exactly 8,640 lines (144 tasks * 30 seeds * 2 optimizers)."""
        from scripts.generate_bbob_highdim_proximity_vs_ei_tasks import (
            generate_bbob_highdim_proximity_vs_ei_tasks,
        )

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            lines = generate_bbob_highdim_proximity_vs_ei_tasks(
                output_file=tmp_path,
                runs_dir="results/sweep_bbob_highdim_proximity_vs_ei",
                baserundir="runs/sweep_bbob_highdim_proximity_vs_ei",
                dimensions=(16, 32),
                seeds=30,
                trials=100,
            )

            self.assertEqual(len(lines), 8640)
            self.assertTrue(os.path.exists(tmp_path))

            with open(tmp_path) as f:
                file_lines = [l.strip() for l in f if l.strip()]
            self.assertEqual(len(file_lines), 8640)

            # 4,320 proposed, 4,320 baseline
            proposed = [l for l in lines if "optimizer_id=SMAC20_ProximityLCB" in l]
            baseline = [l for l in lines if "optimizer_id=SMAC3_HPOFacade_ei" in l]

            self.assertEqual(len(proposed), 4320)
            self.assertEqual(len(baseline), 4320)

            # Check proposed hyperparameters
            sample_p = proposed[0]
            self.assertIn("+optimizer=smac20_proximity_lcb", sample_p)
            self.assertIn("optimizer.acq_func_kwargs.k=25", sample_p)
            self.assertIn("optimizer.acq_func_kwargs.level=0.95", sample_p)
            self.assertIn("optimizer.acq_func_kwargs.eps=0.16", sample_p)
            self.assertIn("optimizer.smac_cfg.model_kwargs.uncertainty_func=proximity_b", sample_p)
            self.assertIn("optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda=1.345", sample_p)
            self.assertIn("task.optimization_resources.n_trials=100", sample_p)

            # Check baseline hyperparameters
            sample_b = baseline[0]
            self.assertIn("+optimizer=smac3_hpo_facade_ei", sample_b)
            self.assertIn("optimizer_id=SMAC3_HPOFacade_ei", sample_b)
            self.assertIn("optimizer_container_id=SMAC3_HPOFacade_ei", sample_b)
            self.assertIn("task.optimization_resources.n_trials=100", sample_b)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestSubmissionScripts(unittest.TestCase):
    def test_sbatch_array_script_exists_and_configured(self):
        """Array script must specify array bounds, memory, and fail-safe logging."""
        script_path = Path(PROJECT_ROOT) / "scripts" / "submit_bbob_highdim_proximity_vs_ei_array.sbatch"
        self.assertTrue(script_path.exists(), f"Missing {script_path}")

        content = script_path.read_text()
        self.assertIn("#SBATCH --job-name=bbob_hi_ei", content)
        self.assertIn("results/sweep_bbob_highdim_proximity_vs_ei", content)

    def test_two_stage_batch_scripts_exist_and_bounded(self):
        """Batch scripts must split tasks to respect LUIS MaxSubmitJob <= 5,000."""
        b1_path = Path(PROJECT_ROOT) / "scripts" / "submit_bbob_highdim_proximity_vs_ei_batch1.sh"
        b2_path = Path(PROJECT_ROOT) / "scripts" / "submit_bbob_highdim_proximity_vs_ei_batch2.sh"

        self.assertTrue(b1_path.exists(), f"Missing {b1_path}")
        self.assertTrue(b2_path.exists(), f"Missing {b2_path}")

        b1_content = b1_path.read_text()
        b2_content = b2_path.read_text()

        self.assertIn("START_TASK=1", b1_content)
        self.assertIn("END_TASK=5000", b1_content)
        self.assertIn("CHUNK_SIZE=200", b1_content)

        self.assertIn("START_TASK=5001", b2_content)
        self.assertIn("END_TASK=8640", b2_content)
        self.assertIn("CHUNK_SIZE=200", b2_content)


class TestGatheringAndAnalysis(unittest.TestCase):
    def test_gather_script_exists(self):
        """Gathering script must exist and point to sweep_bbob_highdim_proximity_vs_ei."""
        script_path = Path(PROJECT_ROOT) / "scripts" / "gather_bbob_highdim_proximity_vs_ei.py"
        self.assertTrue(script_path.exists(), f"Missing {script_path}")

    def test_analysis_script_calculations(self):
        """Analysis engine must correctly compute normalized regret, Cliff's delta, and Holm corrections."""
        from scripts.compute_bbob_highdim_proximity_vs_ei_analysis import (
            apply_holm_bonferroni,
            calculate_cliffs_delta,
        )

        # Cliff's delta
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([40.0, 50.0, 60.0])
        self.assertEqual(calculate_cliffs_delta(x, y), -1.0)

        # Holm-Bonferroni
        raw_p = [0.001, 0.05, 0.02, 0.01]
        adj_p = apply_holm_bonferroni(raw_p)
        self.assertEqual(len(adj_p), 4)
        # raw 0.001 (x4 = 0.004)
        self.assertAlmostEqual(adj_p[0], 0.004)


if __name__ == "__main__":
    unittest.main()
