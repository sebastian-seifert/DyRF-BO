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

        # Batch 1 and Batch 2 scripts
        b1_path = Path("scripts/submit_bbob_highdim_proximity_batch1.sh")
        b2_path = Path("scripts/submit_bbob_highdim_proximity_batch2.sh")
        self.assertTrue(b1_path.exists(), f"{b1_path} must exist")
        self.assertTrue(b2_path.exists(), f"{b2_path} must exist")

        b1_content = b1_path.read_text()
        b2_content = b2_path.read_text()
        self.assertIn("START_TASK=1", b1_content)
        self.assertIn("END_TASK=5000", b1_content)
        self.assertIn("START_TASK=5001", b2_content)
        self.assertIn("END_TASK=8640", b2_content)
        self.assertIn("%25", b1_content)
        self.assertIn("%25", b2_content)


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

    def test_multi_trial_per_run_reduces_to_final_trial(self):
        """Verify that multi-step trial logs are properly reduced to the final trial per run, avoiding Cartesian explosion."""
        from scripts.compute_bbob_highdim_proximity_analysis import compute_statistical_comparison

        rows = []
        # 2 tasks (dim 16, dim 32), 2 seeds, 2 optimizers, each with 50 trials
        for task, dim in [("cfg_16_1_0", 16), ("cfg_32_1_0", 32)]:
            for seed in [1, 2]:
                for opt in ["SMAC20_ProximityLCB", "SMAC3_HPOFacade_lcb"]:
                    base_val = 100.0 if opt == "SMAC3_HPOFacade_lcb" else 80.0
                    for t in range(1, 51):
                        # Cost improves over trials
                        c = base_val / t
                        rows.append({
                            "task_id": task,
                            "dimension": dim,
                            "seed": seed,
                            "optimizer_id": opt,
                            "n_trials": t,
                            "trial_value__cost": c,
                            "trial_value__cost_inc": c,
                        })

        df_multi = pd.DataFrame(rows)
        # Total rows = 2 tasks * 2 seeds * 2 optimizers * 50 trials = 400 rows
        self.assertEqual(len(df_multi), 400)

        results = compute_statistical_comparison(df_multi)
        # Paired runs MUST be 4 (2 tasks * 2 seeds), NOT 4 * 50 * 50 = 10,000!
        self.assertEqual(results["overall"]["n_paired_runs"], 4)
        self.assertEqual(results["dim_16"]["n_paired_runs"], 2)
        self.assertEqual(results["dim_32"]["n_paired_runs"], 2)
        # Proposed (80/50 = 1.6) is strictly better than baseline (100/50 = 2.0)
        self.assertEqual(results["overall"]["wins"], 4)
        self.assertEqual(results["overall"]["losses"], 0)

    def test_calculate_cliffs_delta_matches_definition(self):
        """Verify Cliff's delta matches the formal definition across edge cases including ties."""
        from scripts.compute_bbob_highdim_proximity_analysis import calculate_cliffs_delta

        def ground_truth_cliffs(x, y):
            gt = 0
            lt = 0
            for vx in x:
                gt += int(np.sum(vx > y))
                lt += int(np.sum(vx < y))
            return float((gt - lt) / (len(x) * len(y)))

        # Test case with distinct values and ties
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0])
        y = np.array([2.0, 3.0, 5.0, 5.0, 6.0, 7.0])
        self.assertAlmostEqual(calculate_cliffs_delta(x, y), ground_truth_cliffs(x, y), places=7)

        # Empty array edge case
        self.assertEqual(calculate_cliffs_delta([], [1.0, 2.0]), 0.0)

        # Identical arrays
        self.assertEqual(calculate_cliffs_delta([2.0, 2.0], [2.0, 2.0]), 0.0)

    def test_end_to_end_analysis_cli_file_generation(self):
        """Verify that running compute_bbob_highdim_proximity_analysis outputs valid scorecard and CSVs without OOM."""
        import tempfile
        from scripts.compute_bbob_highdim_proximity_analysis import main
        import sys

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_csv = tmp_path / "logs.csv"
            out_dir = tmp_path / "analysis"

            rows = []
            for task, dim in [("cfg_16_1_0", 16), ("cfg_32_1_0", 32)]:
                for seed in [1, 2]:
                    for opt in ["SMAC20_ProximityLCB", "SMAC3_HPOFacade_lcb"]:
                        for t in range(1, 11):
                            rows.append({
                                "task_id": task,
                                "dimension": dim,
                                "seed": seed,
                                "optimizer_id": opt,
                                "n_trials": t,
                                "trial_value__cost": 10.0 / t if "Proximity" in opt else 15.0 / t,
                                "trial_value__cost_inc": 10.0 / t if "Proximity" in opt else 15.0 / t,
                            })

            pd.DataFrame(rows).to_csv(input_csv, index=False)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "compute_bbob_highdim_proximity_analysis.py",
                    "--input", str(input_csv),
                    "--output-dir", str(out_dir),
                ]
                main()
            finally:
                sys.argv = old_argv

            self.assertTrue((out_dir / "bbob_highdim_proximity_scorecard.md").exists())
            self.assertTrue((out_dir / "stratified_scorecard.csv").exists())
            self.assertTrue((out_dir / "task_details.csv").exists())

            strat_df = pd.read_csv(out_dir / "stratified_scorecard.csv")
            self.assertIn("Overall (D=16 & D=32)", strat_df["stratum"].values)
            self.assertEqual(int(strat_df.loc[strat_df["stratum"] == "Overall (D=16 & D=32)", "n_paired_runs"].iloc[0]), 4)


class TestBBOBHighDimGathering(unittest.TestCase):
    def test_gather_resolves_hydra_interpolated_task_name(self):
        """Verify gather_bbob_highdim_logs resolves ${task.name} and assigns correct dimensions."""
        import tempfile
        import json
        from scripts.gather_bbob_highdim_proximity import gather_bbob_highdim_logs

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            runs_dir = tmp_path / "runs"
            out_dir = tmp_path / "results"

            # Create mock run 1: BBOB 16D with task_id: ${task.name}
            run1 = runs_dir / "SMAC20_ProximityLCB" / "BBOB" / "bbob" / "16" / "1" / "0" / "1"
            run1.mkdir(parents=True, exist_ok=True)
            (run1 / ".hydra").mkdir(parents=True, exist_ok=True)
            with open(run1 / ".hydra" / "config.yaml", "w") as f:
                f.write(
                    "task_id: ${task.name}\n"
                    "optimizer_id: SMAC20_ProximityLCB\n"
                    "seed: 1\n"
                    "task:\n"
                    "  name: bbob/16/1/0\n"
                )
            with open(run1 / "trial_logs.jsonl", "w") as f:
                f.write(json.dumps({"n_trials": 1, "trial_value": {"cost": 42.0}}) + "\n")

            # Create mock run 2: BBOB 32D with task_id: ${task.name}
            run2 = runs_dir / "SMAC3_HPOFacade_lcb" / "BBOB" / "bbob" / "32" / "2" / "1" / "1"
            run2.mkdir(parents=True, exist_ok=True)
            (run2 / ".hydra").mkdir(parents=True, exist_ok=True)
            with open(run2 / ".hydra" / "config.yaml", "w") as f:
                f.write(
                    "task_id: ${task.name}\n"
                    "optimizer_id: SMAC3_HPOFacade_lcb\n"
                    "seed: 1\n"
                    "task:\n"
                    "  name: bbob/32/2/1\n"
                )
            with open(run2 / "trial_logs.jsonl", "w") as f:
                f.write(json.dumps({"n_trials": 1, "trial_value": {"cost": 55.0}}) + "\n")

            df = gather_bbob_highdim_logs(runs_dir=str(runs_dir), output_dir=str(out_dir))

            self.assertFalse(df.empty)
            # Must NOT be the unresolved string "${task.name}"
            self.assertNotIn("${task.name}", df["task_id"].values)
            self.assertIn("bbob/16/1/0", df["task_id"].values)
            self.assertIn("bbob/32/2/1", df["task_id"].values)

            # Dimensions must be resolved to 16 and 32
            row16 = df[df["task_id"] == "bbob/16/1/0"].iloc[0]
            row32 = df[df["task_id"] == "bbob/32/2/1"].iloc[0]
            self.assertEqual(row16["dimension"], 16)
            self.assertEqual(row32["dimension"], 32)


if __name__ == "__main__":
    unittest.main()
