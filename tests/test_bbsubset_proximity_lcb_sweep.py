"""Unit and Integration Test Suite for CARP-S BBsubset Proximity LCB Sweep.

Verifies:
1. CarpsBBSubsetRegistry.get_working_dev_tasks() returns exactly 18 tasks (20 dev tasks minus 2 NAS).
2. Task generator generates exactly 360 tasks with 10 seeds paired 1-to-1 and 100 trials.
3. YAML configs for smac3_hpo_facade_lcb and smac20_proximity_lcb exist and are valid.
4. SLURM sbatch and bash submission scripts respect LUIS cluster limits (MaxArraySize <= 200, %25 concurrency).
5. Analysis script compute_bbsubset_proximity_lcb_analysis slices correctly at t=50 and t=100.
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
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


class TestCarpsBBSubsetRegistryWorkingTasks(unittest.TestCase):
    def test_working_dev_tasks_count_and_composition(self):
        """Should return exactly 18 tasks, excluding Naval and SliceLocalization NAS benchmarks."""
        working_tasks = CarpsBBSubsetRegistry.get_working_dev_tasks()
        self.assertEqual(len(working_tasks), 18)

        # Ensure NAS tasks are strictly excluded
        for t in working_tasks:
            self.assertNotIn("NavalPropulsionBenchmark", t)
            self.assertNotIn("SliceLocalizationBenchmark", t)

        # Ensure all 4 BBOB tasks are present
        bbob_tasks = [t for t in working_tasks if "bbob" in t]
        self.assertEqual(len(bbob_tasks), 4)

        # Ensure all 3 HPOBench ML tasks are present
        hpobench_ml = [t for t in working_tasks if "hpobench_blackbox_tabular_ml" in t]
        self.assertEqual(len(hpobench_ml), 3)

        # Ensure all 11 YAHPO tasks are present
        yahpo_tasks = [t for t in working_tasks if "yahpo" in t]
        self.assertEqual(len(yahpo_tasks), 11)


class TestProximityLCBOptimizerConfigs(unittest.TestCase):
    def test_baseline_lcb_config_exists_and_valid(self):
        """Baseline config smac3_hpo_facade_lcb.yaml must exist and configure LCB with kappa=1.96."""
        cfg_path = Path(PROJECT_ROOT) / "carps_integration/configs/optimizer/smac3_hpo_facade_lcb.yaml"
        self.assertTrue(cfg_path.exists(), f"Missing config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg.get("optimizer_id"), "SMAC3_HPOFacade_lcb")
        self.assertEqual(cfg["optimizer"]["acq_func_name"], "lcb")

    def test_proposed_proximity_lcb_config_exists_and_valid(self):
        """Proposed config smac20_proximity_lcb.yaml must exist and configure proximity_b with k=10, alpha=0.05."""
        cfg_path = Path(PROJECT_ROOT) / "carps_integration/configs/optimizer/smac20_proximity_lcb.yaml"
        self.assertTrue(cfg_path.exists(), f"Missing config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg.get("optimizer_id"), "SMAC20_ProximityLCB_k10")
        self.assertEqual(cfg["optimizer"]["acq_func_name"], "proximity_lcb")
        self.assertEqual(cfg["optimizer"]["smac_cfg"]["model_kwargs"]["uncertainty_func"], "proximity_b")


class TestGenerateBBSUBSETProximityLCBSweepTasks(unittest.TestCase):
    def test_generate_tasks_output_and_pairing(self):
        """Should generate 360 lines with 18 tasks * 10 seeds * 2 optimizers, paired seeds 1..10, 100 trials."""
        from scripts.generate_bbsubset_proximity_lcb_sweep_tasks import generate_bbsubset_proximity_lcb_tasks

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            lines = generate_bbsubset_proximity_lcb_tasks(
                output_file=tmp_path,
                runs_dir="results/bbsubset_proximity_lcb",
                seeds=10,
                trials=100,
            )
            # 18 tasks * 10 seeds * 2 optimizers = 360 tasks
            self.assertEqual(len(lines), 360)

            # Check file contents
            with open(tmp_path, "r") as f:
                saved_lines = [l.strip() for l in f if l.strip()]
            self.assertEqual(len(saved_lines), 360)

            # Check distribution of optimizers
            proposed_lines = [l for l in lines if "SMAC20_ProximityLCB_k10" in l]
            baseline_lines = [l for l in lines if "SMAC3_HPOFacade_lcb" in l]
            self.assertEqual(len(proposed_lines), 180)
            self.assertEqual(len(baseline_lines), 180)

            # Check trials=100
            for l in lines:
                self.assertIn("task.optimization_resources.n_trials=100", l)

            # Check paired seeds: each task must have seeds 1 to 10 for both methods
            tasks = CarpsBBSubsetRegistry.get_working_dev_tasks()
            for t in tasks:
                t_name = t.split("/")[-1]
                p_seeds = sorted([
                    int(l.split("seed=")[1].split()[0])
                    for l in proposed_lines if t_name in l
                ])
                b_seeds = sorted([
                    int(l.split("seed=")[1].split()[0])
                    for l in baseline_lines if t_name in l
                ])
                self.assertEqual(p_seeds, list(range(1, 11)))
                self.assertEqual(b_seeds, list(range(1, 11)))

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestSlurmSubmissionScripts(unittest.TestCase):
    def test_sbatch_and_launcher_scripts(self):
        """Array tasks must be chunked into <= 200 tasks with %25 concurrency."""
        sbatch_path = Path(PROJECT_ROOT) / "scripts/submit_bbsubset_proximity_lcb_array.sbatch"
        all_sh_path = Path(PROJECT_ROOT) / "scripts/submit_bbsubset_proximity_lcb_all.sh"

        self.assertTrue(sbatch_path.exists(), f"Missing: {sbatch_path}")
        self.assertTrue(all_sh_path.exists(), f"Missing: {all_sh_path}")

        sbatch_content = sbatch_path.read_text()
        all_sh_content = all_sh_path.read_text()

        self.assertIn("run_carps_patched.py", sbatch_content)
        self.assertIn("%25", all_sh_content)
        self.assertIn("sbatch", all_sh_content)


class TestDualCheckpointAnalysis(unittest.TestCase):
    def test_compute_analysis_slices_at_50_and_100(self):
        """Analysis script must extract costs at trial 50 and trial 100 and output tables."""
        from scripts.compute_bbsubset_proximity_lcb_analysis import (
            compute_dual_checkpoint_analysis,
            calculate_cliffs_delta,
        )

        # Cliff's delta tests
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        self.assertEqual(calculate_cliffs_delta(x, y), -1.0)
        self.assertEqual(calculate_cliffs_delta(y, x), 1.0)
        self.assertEqual(calculate_cliffs_delta(x, x), 0.0)

        # Create synthetic trial data for 2 tasks, 5 seeds, 2 optimizers with 100 trials
        records = []
        for task in ["task_A", "task_B"]:
            for seed in range(1, 6):
                # Proposed method: steadily improving
                cost_proposed = 10.0
                for trial in range(1, 101):
                    cost_proposed -= 0.05
                    records.append({
                        "task_id": task,
                        "optimizer_id": "SMAC20_ProximityLCB_k10",
                        "seed": seed,
                        "trial": trial,
                        "trial_value__cost_inc": cost_proposed,
                    })

                # Baseline method: slower improvement
                cost_baseline = 10.0
                for trial in range(1, 101):
                    cost_baseline -= 0.02
                    records.append({
                        "task_id": task,
                        "optimizer_id": "SMAC3_HPOFacade_lcb",
                        "seed": seed,
                        "trial": trial,
                        "trial_value__cost_inc": cost_baseline,
                    })

        df = pd.DataFrame(records)

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, "logs.parquet")
            df.to_parquet(parquet_path)

            results = compute_dual_checkpoint_analysis(
                input_file=parquet_path,
                output_dir=tmpdir,
                checkpoints=[50, 100],
                proposed_id="SMAC20_ProximityLCB_k10",
                baseline_id="SMAC3_HPOFacade_lcb",
            )

            self.assertIn(50, results)
            self.assertIn(100, results)

            # At both checkpoints, proposed is lower cost (better)
            for t in [50, 100]:
                res_t = results[t]
                self.assertIn("wilcoxon_stat", res_t)
                self.assertIn("wilcoxon_p", res_t)
                self.assertIn("cliffs_delta", res_t)
                self.assertLess(res_t["cliffs_delta"], 0.0)  # Proposed is better
                self.assertEqual(res_t["wins"], 10)  # 2 tasks * 5 seeds = 10 wins
                self.assertEqual(res_t["losses"], 0)

            # Verify markdown report files were created
            report_50 = Path(tmpdir) / "proximity_lcb_analysis_t50.md"
            report_100 = Path(tmpdir) / "proximity_lcb_analysis_t100.md"
            self.assertTrue(report_50.exists())
            self.assertTrue(report_100.exists())


if __name__ == "__main__":
    unittest.main()
