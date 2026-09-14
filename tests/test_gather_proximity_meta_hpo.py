#!/usr/bin/env python3
"""Tests for CARP-S Proximity Meta-HPO Gather Script."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import pandas as pd


class TestGatherProximityMetaHPO(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.runs_dir = Path(self.tmpdir) / "runs" / "sweep_proximity_meta_hpo"
        self.out_dir = Path(self.tmpdir) / "results" / "sweep_proximity_meta_hpo"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_mock_run(self, task_name: str, opt_id: str, seed: int, n_trials: int = 5):
        run_path = self.runs_dir / task_name / opt_id / str(seed)
        hydra_dir = run_path / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)

        # Write dummy hydra config
        hydra_cfg = (
            f"task_name: {task_name}\n"
            f"task_id: {task_name}\n"
            f"optimizer_id: {opt_id}\n"
            f"optimizer_container_id: SMAC20_ProximityLCB\n"
            f"seed: {seed}\n"
        )
        (hydra_dir / "config.yaml").write_text(hydra_cfg)

        # Write trial_logs.jsonl
        log_file = run_path / "trial_logs.jsonl"
        with open(log_file, "w") as f:
            for t in range(1, n_trials + 1):
                record = {
                    "n_trials": t,
                    "n_function_calls": t,
                    "trial_info": {
                        "config": [0.1, 0.2],
                        "instance": None,
                        "seed": seed,
                        "budget": None,
                    },
                    "trial_value": {
                        "cost": 10.0 / t + (seed * 0.1),
                        "time": 0.01,
                        "virtual_time": 0.0,
                        "status": 1,
                        "starttime": 1000.0 + t,
                        "endtime": 1000.0 + t + 0.01,
                    },
                }
                f.write(json.dumps(record) + "\n")

    def test_gather_produces_parquet_and_csv(self):
        from scripts.gather_proximity_meta_hpo import gather_proximity_meta_data

        # Create two configs, two seeds
        self._create_mock_run("bbob_task_1", "SMAC20_ProximityLCB_cfg_01", seed=1)
        self._create_mock_run("bbob_task_1", "SMAC20_ProximityLCB_cfg_01", seed=2)
        self._create_mock_run("bbob_task_1", "SMAC20_ProximityLCB_cfg_02", seed=1)

        gather_proximity_meta_data(
            runs_base=str(self.runs_dir),
            outdir=str(self.out_dir),
        )

        parquet_path = self.out_dir / "logs.parquet"
        csv_path = self.out_dir / "logs.csv"

        self.assertTrue(parquet_path.exists(), "logs.parquet was not created")
        self.assertTrue(csv_path.exists(), "logs.csv was not created")

        df = pd.read_parquet(parquet_path)
        self.assertGreater(len(df), 0)
        self.assertIn("optimizer_id", df.columns)
        self.assertIn("seed", df.columns)

    def test_gather_cli_arguments(self):
        import subprocess
        import sys

        self._create_mock_run("bbob_task_2", "SMAC20_ProximityLCB_cfg_03", seed=1)

        cmd = [
            sys.executable,
            "scripts/gather_proximity_meta_hpo.py",
            "--runs-dir",
            str(self.runs_dir),
            "--outdir",
            str(self.out_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Gather CLI failed: {res.stderr}")
        self.assertTrue((self.out_dir / "logs.parquet").exists())


if __name__ == "__main__":
    unittest.main()
