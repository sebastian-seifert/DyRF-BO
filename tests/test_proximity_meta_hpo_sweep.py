"""Unit and Integration Test Suite for Proximity Lower Bound Meta-HPO Sweep.

Verifies:
1. Sobol sequence sampling:
   - Exactly N = 50 configurations generated with fixed seed for reproducibility.
   - k in [5, 30] (integer, inclusive).
   - lambda (decay_lambda) in [0.2, 2.0] (float).
   - eps in [0.02, 0.20] (float).
   - Correct JSON serialization to results/sweep_proximity_meta_hpo/sobol_configs.json.
2. Task Generator:
   - Exactly 4,500 Hydra task command lines generated (50 configs * 18 tasks * 5 seeds).
   - All 18 working dev tasks from CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True).
   - Seeds 1 to 5 per task.
   - Budget T = 100 trials per task run.
   - Correct configuration overrides (k, eps, decay_lambda, optimizer_id).
3. SLURM Scripts:
   - submit_proximity_meta_array.sbatch and submit_proximity_meta_all.sh exist and are valid.
   - Chunked to <= 200 tasks with %25 concurrency.
4. Normalized Regret Analysis:
   - y_norm_m(cfg) = (y_m(cfg) - y_min_m) / (y_max_m - y_min_m)
   - Loss(cfg) = (1 / M) * sum_m y_norm_m(cfg)
   - Safe division by zero when y_max_m == y_min_m.
   - Proper ranking of configurations from best (lowest Loss) to worst.
   - Generates CSV and Markdown report artifacts.
"""

from __future__ import annotations

import json
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

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


class TestSobolSamplingBounds(unittest.TestCase):
    def test_sobol_sampling_count_and_reproducibility(self):
        """Sobol sampling must produce exactly 50 configs and be deterministic with fixed seed."""
        from scripts.sample_proximity_meta_configs import sample_proximity_meta_configs

        configs_1 = sample_proximity_meta_configs(n_configs=50, seed=42)
        configs_2 = sample_proximity_meta_configs(n_configs=50, seed=42)

        self.assertEqual(len(configs_1), 50)
        self.assertEqual(len(configs_2), 50)
        self.assertEqual(configs_1, configs_2)

    def test_sobol_sampling_bounds(self):
        """Configurations must strictly observe parameter bounds: k in [5, 30], lambda in [0.2, 2.0], eps in [0.02, 0.20]."""
        from scripts.sample_proximity_meta_configs import sample_proximity_meta_configs

        configs = sample_proximity_meta_configs(n_configs=50, seed=42)
        k_values = set()

        for idx, cfg in enumerate(configs):
            self.assertIn("config_id", cfg)
            self.assertIn("k", cfg)
            self.assertTrue(any(k in cfg for k in ("decay_lambda", "lambda")), "Config must contain lambda parameter")
            self.assertIn("eps", cfg)

            k = cfg["k"]
            lam = cfg.get("decay_lambda", cfg.get("lambda"))
            eps = cfg["eps"]

            self.assertIsInstance(k, (int, np.integer), f"k must be an integer, got {type(k)}")
            self.assertTrue(5 <= k <= 30, f"k={k} out of bounds [5, 30]")
            k_values.add(k)

            self.assertIsInstance(lam, (float, np.floating), f"lambda must be a float, got {type(lam)}")
            self.assertTrue(0.2 <= lam <= 2.0, f"lambda={lam} out of bounds [0.2, 2.0]")

            self.assertIsInstance(eps, (float, np.floating), f"eps must be a float, got {type(eps)}")
            self.assertTrue(0.02 <= eps <= 0.20, f"eps={eps} out of bounds [0.02, 0.20]")

        # Check diversity in sampled k
        self.assertGreater(len(k_values), 10, "k values should be diverse across [5, 30]")

    def test_sobol_json_serialization(self):
        """Configs must serialize to JSON and match expected schema."""
        from scripts.sample_proximity_meta_configs import sample_proximity_meta_configs

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            configs = sample_proximity_meta_configs(n_configs=50, seed=42, output_file=tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            with open(tmp_path, "r") as f:
                loaded = json.load(f)

            self.assertEqual(len(loaded), 50)
            self.assertEqual(loaded[0]["config_id"], configs[0]["config_id"])
            self.assertEqual(loaded[0]["k"], configs[0]["k"])
            self.assertAlmostEqual(loaded[0]["eps"], configs[0]["eps"], places=4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestGenerateProximityMetaSweepTasks(unittest.TestCase):
    def test_benchmark_scope_and_scale(self):
        """Must generate exactly 4,500 tasks across 18 working tasks, 5 seeds, and 50 configurations."""
        from scripts.generate_proximity_meta_sweep_tasks import generate_proximity_meta_tasks

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tasks_lines = generate_proximity_meta_tasks(
                output_file=tmp_path,
                n_configs=50,
                seeds=5,
                trials=100,
                seed_sobol=42,
            )
            # 50 configs * 18 tasks * 5 seeds = 4,500 runs
            self.assertEqual(len(tasks_lines), 4500)

            # Check file on disk
            with open(tmp_path, "r") as f:
                saved_lines = [l.strip() for l in f if l.strip()]
            self.assertEqual(len(saved_lines), 4500)

            # Check working tasks
            working_tasks = CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)
            self.assertEqual(len(working_tasks), 18)

            for wt in working_tasks:
                wt_name = wt.split("/")[-1]
                matches = [l for l in tasks_lines if wt_name in l]
                # 50 configs * 5 seeds = 250 runs per task
                self.assertEqual(len(matches), 250)

            # Check seeds 1 to 5
            for s in range(1, 6):
                seed_matches = [l for l in tasks_lines if f"seed={s} " in l or l.endswith(f"seed={s}")]
                # 50 configs * 18 tasks = 900 runs per seed
                self.assertEqual(len(seed_matches), 900)

            # Check trials=100
            for l in tasks_lines[:50]:
                self.assertIn("task.optimization_resources.n_trials=100", l)
                self.assertIn("+optimizer=smac20_proximity_lcb", l)
                self.assertIn("++optimizer.acq_func_kwargs.k=", l)
                self.assertIn("++optimizer.acq_func_kwargs.eps=", l)
                self.assertIn("++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda=", l)
                self.assertIn("optimizer_id=SMAC20_ProximityLCB_", l)
                self.assertIn("optimizer_container_id=SMAC20_ProximityLCB", l)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestSlurmSubmissionScripts(unittest.TestCase):
    def test_sbatch_and_launcher_scripts(self):
        """Array tasks must be chunked into <= 200 tasks with %25 concurrency."""
        sbatch_path = Path(PROJECT_ROOT) / "scripts/submit_proximity_meta_array.sbatch"
        all_sh_path = Path(PROJECT_ROOT) / "scripts/submit_proximity_meta_all.sh"

        self.assertTrue(sbatch_path.exists(), f"Missing: {sbatch_path}")
        self.assertTrue(all_sh_path.exists(), f"Missing: {all_sh_path}")

        sbatch_content = sbatch_path.read_text()
        all_sh_content = all_sh_path.read_text()

        self.assertIn("run_carps_patched.py", sbatch_content)
        self.assertIn("results/sweep_proximity_meta_hpo/tasks.txt", sbatch_content)
        self.assertIn("CHUNK_SIZE=200", all_sh_content)
        self.assertIn("%25", all_sh_content)
        self.assertIn("sbatch", all_sh_content)
        self.assertIn("results/sweep_proximity_meta_hpo/tasks.txt", all_sh_content)


class TestNormalizedRegretAnalysis(unittest.TestCase):
    def test_normalized_regret_formula(self):
        """Loss(cfg) = 1/M * sum_m (y_m(cfg) - y_min_m) / (y_max_m - y_min_m)."""
        from scripts.compute_proximity_meta_analysis import compute_normalized_regret_loss

        # 3 configs across 2 tasks
        # Task 1 costs: cfg_1=10.0, cfg_2=20.0, cfg_3=30.0 -> min=10, max=30
        # normalized: cfg_1=0.0, cfg_2=0.5, cfg_3=1.0
        # Task 2 costs: cfg_1=5.0, cfg_2=3.0, cfg_3=1.0 -> min=1, max=5
        # normalized: cfg_1=1.0, cfg_2=0.5, cfg_3=0.0
        # Mean Loss: cfg_1=0.5, cfg_2=0.5, cfg_3=0.5
        task_means = {
            "task_1": {"cfg_1": 10.0, "cfg_2": 20.0, "cfg_3": 30.0},
            "task_2": {"cfg_1": 5.0, "cfg_2": 3.0, "cfg_3": 1.0},
        }
        loss_dict, norm_df = compute_normalized_regret_loss(task_means)

        self.assertAlmostEqual(loss_dict["cfg_1"], 0.5)
        self.assertAlmostEqual(loss_dict["cfg_2"], 0.5)
        self.assertAlmostEqual(loss_dict["cfg_3"], 0.5)
        self.assertAlmostEqual(norm_df.loc["cfg_1", "task_1"], 0.0)
        self.assertAlmostEqual(norm_df.loc["cfg_2", "task_1"], 0.5)
        self.assertAlmostEqual(norm_df.loc["cfg_3", "task_1"], 1.0)
        self.assertAlmostEqual(norm_df.loc["cfg_1", "task_2"], 1.0)
        self.assertAlmostEqual(norm_df.loc["cfg_2", "task_2"], 0.5)
        self.assertAlmostEqual(norm_df.loc["cfg_3", "task_2"], 0.0)

    def test_constant_task_cost_zero_division_guard(self):
        """When all configs achieve identical cost on a task (y_max == y_min), normalized regret must be 0.0."""
        from scripts.compute_proximity_meta_analysis import compute_normalized_regret_loss

        task_means = {
            "task_flat": {"cfg_1": 42.0, "cfg_2": 42.0, "cfg_3": 42.0},
            "task_diff": {"cfg_1": 1.0, "cfg_2": 2.0, "cfg_3": 3.0},
        }
        loss_dict, norm_df = compute_normalized_regret_loss(task_means)
        self.assertEqual(norm_df.loc["cfg_1", "task_flat"], 0.0)
        self.assertEqual(norm_df.loc["cfg_2", "task_flat"], 0.0)
        self.assertEqual(norm_df.loc["cfg_3", "task_flat"], 0.0)

        # task_diff: min=1.0, max=3.0 -> cfg_1=0.0, cfg_2=0.5, cfg_3=1.0
        # Average loss: cfg_1=0.0, cfg_2=0.25, cfg_3=0.50
        self.assertAlmostEqual(loss_dict["cfg_1"], 0.0)
        self.assertAlmostEqual(loss_dict["cfg_2"], 0.25)
        self.assertAlmostEqual(loss_dict["cfg_3"], 0.5)

    def test_ranking_and_report_generation(self):
        """End-to-end test on synthetic benchmark DataFrame producing ranked leaderboard and markdown."""
        from scripts.compute_proximity_meta_analysis import analyze_proximity_meta_hpo

        # Create mock data for 3 configs, 2 tasks, 5 seeds, 100 trials
        records = []
        configs_meta = [
            {"config_id": "cfg_01", "k": 10, "decay_lambda": 1.0, "eps": 0.05},
            {"config_id": "cfg_02", "k": 20, "decay_lambda": 1.5, "eps": 0.10},
            {"config_id": "cfg_03", "k": 30, "decay_lambda": 0.5, "eps": 0.15},
        ]
        # Make cfg_01 the clear winner
        cost_multipliers = {"cfg_01": 1.0, "cfg_02": 2.0, "cfg_03": 3.0}

        for cfg in configs_meta:
            cid = cfg["config_id"]
            mult = cost_multipliers[cid]
            for task in ["task_A", "task_B"]:
                for seed in range(1, 6):
                    cost = 10.0 * mult + seed * 0.1
                    records.append({
                        "optimizer_id": f"SMAC20_ProximityLCB_{cid}",
                        "task_id": task,
                        "seed": seed,
                        "trial": 100,
                        "trial_value__cost_inc": cost,
                    })

        df = pd.DataFrame(records)

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, "logs.parquet")
            configs_json = os.path.join(tmpdir, "configs.json")
            df.to_parquet(parquet_path)

            with open(configs_json, "w") as f:
                json.dump(configs_meta, f)

            leaderboard_df, norm_task_df = analyze_proximity_meta_hpo(
                input_data=parquet_path,
                configs_path=configs_json,
                output_dir=tmpdir,
                checkpoint=100,
            )

            self.assertEqual(len(leaderboard_df), 3)
            # Rank 1 must be cfg_01
            self.assertEqual(leaderboard_df.iloc[0]["config_id"], "cfg_01")
            self.assertAlmostEqual(leaderboard_df.iloc[0]["loss_normalized_regret"], 0.0)

            # Rank 3 must be cfg_03
            self.assertEqual(leaderboard_df.iloc[2]["config_id"], "cfg_03")
            self.assertAlmostEqual(leaderboard_df.iloc[2]["loss_normalized_regret"], 1.0)

            # Check exported files
            self.assertTrue((Path(tmpdir) / "proximity_meta_hpo_leaderboard.csv").exists())
            self.assertTrue((Path(tmpdir) / "proximity_meta_hpo_leaderboard.md").exists())
            self.assertTrue((Path(tmpdir) / "proximity_meta_hpo_per_task.csv").exists())

    def test_load_from_telemetry_json_directory_with_filename_metadata(self):
        """Verify that load_logs_dataframe parses optimizer, task, and seed from telemetry filenames."""
        from scripts.compute_proximity_meta_analysis import load_logs_dataframe

        with tempfile.TemporaryDirectory() as tmpdir:
            telem_dir = Path(tmpdir)
            # Create a telemetry file matching CARP-S naming
            fn = telem_dir / "telemetry_SMAC20_ProximityLCB_cfg_00_subset_yahpo_lcbench_seed2.json"
            sample_data = {
                "task_name": "subset_yahpo_lcbench",
                "trials": [
                    {"cost": 10.0, "cost_inc": 10.0},
                    {"cost": 8.0, "cost_inc": 8.0},
                ]
            }
            with open(fn, "w") as f:
                json.dump(sample_data, f)

            df = load_logs_dataframe(telem_dir)
            self.assertEqual(len(df), 2)
            self.assertEqual(df.iloc[0]["optimizer_id"], "SMAC20_ProximityLCB_cfg_00")
            self.assertEqual(df.iloc[0]["task_id"], "subset_yahpo_lcbench")
            self.assertEqual(df.iloc[0]["seed"], 2)
            self.assertEqual(df.iloc[1]["trial_value__cost_inc"], 8.0)


if __name__ == "__main__":
    unittest.main()
