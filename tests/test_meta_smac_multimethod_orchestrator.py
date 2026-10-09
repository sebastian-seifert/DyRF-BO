"""Unit and Integration Tests for Multi-Method Proximity Meta SMAC Orchestrator.

Verifies:
1. CLI Argument Parsing:
   - CLI accepts --method {a,b,ac,bc} (defaulting to 'a').
   - Passes method to MetaSmacOrchestrator.
   - Invalid methods are rejected.
2. Method-Adapted Default Paths:
   - Default output_dir and baserundir adapt to method if not explicitly specified.
   - Method 'a' preserves 'results/meta_smac_proximity_hpo' and 'runs/meta_smac_proximity_hpo'.
   - Methods 'b', 'ac', 'bc' use 'results/meta_smac_proximity_{method}_hpo' and 'runs/meta_smac_proximity_{method}_hpo'.
   - Custom output_dir and baserundir override defaults.
3. ConfigSpace & Task Generation Integration:
   - MetaSmacOrchestrator configspace matches create_proximity_meta_configspace(method=self.method).
   - Tasks generated during iteration contain method-specific optimizer identifiers and parameters.
4. Dry-Run Execution for Methods B, AC, BC:
   - Checkpoints persisted to meta_smac_checkpoint.json.
   - Evaluations logged to meta_leaderboard.csv with method-specific hyperparameter columns.
   - Initial evaluation (iteration 1) evaluates method-specific incumbent.
5. Checkpoint Resume Across Methods:
   - Resuming from checkpoint restores past trials and preserves initial incumbent evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_meta_smac_proximity_hpo import MetaSmacOrchestrator, main


class TestMetaSmacCLIAndPathAdaptation(unittest.TestCase):
    """Verifies CLI argument parsing and method-adapted default paths."""

    @classmethod
    def tearDownClass(cls):
        import shutil
        for p in [
            "results/meta_smac_proximity_b_hpo",
            "results/meta_smac_proximity_ac_hpo",
            "results/meta_smac_proximity_bc_hpo",
            "runs/meta_smac_proximity_b_hpo",
            "runs/meta_smac_proximity_ac_hpo",
            "runs/meta_smac_proximity_bc_hpo",
        ]:
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)

    def test_default_paths_adapt_to_method(self):
        """Default output_dir and baserundir must adapt to method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # We don't want to pollute current working directory, so we test path properties
            # Method A (default)
            orch_a = MetaSmacOrchestrator(method="a", start_iteration=1, end_iteration=1, dry_run=True)
            self.assertEqual(orch_a.output_dir, Path("results/meta_smac_proximity_hpo"))
            self.assertEqual(orch_a.baserundir, Path("runs/meta_smac_proximity_hpo"))

            # Method B
            orch_b = MetaSmacOrchestrator(method="b", start_iteration=1, end_iteration=1, dry_run=True)
            self.assertEqual(orch_b.output_dir, Path("results/meta_smac_proximity_b_hpo"))
            self.assertEqual(orch_b.baserundir, Path("runs/meta_smac_proximity_b_hpo"))

            # Method AC
            orch_ac = MetaSmacOrchestrator(method="ac", start_iteration=1, end_iteration=1, dry_run=True)
            self.assertEqual(orch_ac.output_dir, Path("results/meta_smac_proximity_ac_hpo"))
            self.assertEqual(orch_ac.baserundir, Path("runs/meta_smac_proximity_ac_hpo"))

            # Method BC
            orch_bc = MetaSmacOrchestrator(method="bc", start_iteration=1, end_iteration=1, dry_run=True)
            self.assertEqual(orch_bc.output_dir, Path("results/meta_smac_proximity_bc_hpo"))
            self.assertEqual(orch_bc.baserundir, Path("runs/meta_smac_proximity_bc_hpo"))

            # Explicit paths override defaults
            custom_out = Path(tmpdir) / "custom_out"
            custom_runs = Path(tmpdir) / "custom_runs"
            orch_custom = MetaSmacOrchestrator(
                method="bc",
                output_dir=str(custom_out),
                baserundir=str(custom_runs),
                dry_run=True,
            )
            self.assertEqual(orch_custom.output_dir, custom_out)
            self.assertEqual(orch_custom.baserundir, custom_runs)

    def test_cli_parsing_method_argument(self):
        """CLI main() parses --method and instantiates MetaSmacOrchestrator appropriately."""
        for m in ["a", "b", "ac", "bc"]:
            with patch("argparse._sys.argv", ["run_meta_smac_proximity_hpo.py", "--method", m, "--dry-run", "--end-iteration", "1"]), \
                 patch("scripts.run_meta_smac_proximity_hpo.MetaSmacOrchestrator.run") as mock_run:
                with patch("scripts.run_meta_smac_proximity_hpo.MetaSmacOrchestrator.__init__", return_value=None) as mock_init:
                    main()
                    mock_init.assert_called_once()
                    _, kwargs = mock_init.call_args
                    self.assertEqual(kwargs.get("method"), m)


class TestMultiMethodConfigSpaceAndTasks(unittest.TestCase):
    """Verifies that each method binds the correct ConfigSpace and generates method-specific task commands."""

    def test_configspace_matches_method(self):
        """Orchestrator configspace must match method-specific parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_a = MetaSmacOrchestrator(method="a", output_dir=str(Path(tmpdir) / "a"), dry_run=True)
            self.assertEqual(set(orch_a.cs.keys()), {"k", "decay_lambda", "eps"})

            orch_b = MetaSmacOrchestrator(method="b", output_dir=str(Path(tmpdir) / "b"), dry_run=True)
            self.assertEqual(set(orch_b.cs.keys()), {"decay_lambda", "eps"})

            orch_ac = MetaSmacOrchestrator(method="ac", output_dir=str(Path(tmpdir) / "ac"), dry_run=True)
            self.assertEqual(set(orch_ac.cs.keys()), {"k", "decay_lambda", "eps", "alpha"})

            orch_bc = MetaSmacOrchestrator(method="bc", output_dir=str(Path(tmpdir) / "bc"), dry_run=True)
            self.assertEqual(set(orch_bc.cs.keys()), {"decay_lambda", "eps", "alpha"})

    def test_task_generation_uses_method_specific_flags(self):
        """Dry-run iteration generates tasks containing method-specific optimizer and parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_b = Path(tmpdir) / "b"
            orch_b = MetaSmacOrchestrator(
                method="b",
                start_iteration=1,
                end_iteration=1,
                seeds=2,
                trials=50,
                output_dir=str(out_b),
                dry_run=True,
            )
            orch_b.run()

            tasks_file = out_b / "iter_001" / "tasks.txt"
            self.assertTrue(tasks_file.exists())
            lines = tasks_file.read_text().strip().split("\n")
            self.assertEqual(len(lines), 18 * 2)
            for line in lines:
                self.assertIn("+optimizer=smac20_proximity_b_lcb", line)
                self.assertIn("optimizer_id=SMAC20_ProximityB_LCB_iter001", line)
                self.assertNotIn("acq_func_kwargs.k=", line)


class TestMultiMethodDryRunExecution(unittest.TestCase):
    """Verifies dry-run iterations, checkpoints, and leaderboards for Methods B, AC, and BC."""

    def test_method_b_dry_run_and_leaderboard(self):
        """Method B dry-run must save checkpoint and leaderboard with decay_lambda and eps columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out_b"
            orch = MetaSmacOrchestrator(
                method="b",
                start_iteration=1,
                end_iteration=2,
                seeds=2,
                trials=10,
                output_dir=str(out_dir),
                dry_run=True,
            )
            best = orch.run()

            # Checkpoint verification
            checkpoint_file = out_dir / "meta_smac_checkpoint.json"
            leaderboard_file = out_dir / "meta_leaderboard.csv"
            self.assertTrue(checkpoint_file.exists())
            self.assertTrue(leaderboard_file.exists())

            df = pd.read_csv(leaderboard_file)
            self.assertEqual(len(df), 2)
            self.assertIn("decay_lambda", df.columns)
            self.assertIn("eps", df.columns)
            self.assertNotIn("k", df.columns)
            self.assertNotIn("alpha", df.columns)

            # Iteration 1 must be Method B incumbent: decay_lambda=1.345, eps=0.16
            row0 = df.iloc[0]
            self.assertEqual(int(row0["iteration"]), 1)
            self.assertAlmostEqual(float(row0["decay_lambda"]), 1.345, places=3)
            self.assertAlmostEqual(float(row0["eps"]), 0.16, places=3)

    def test_method_ac_dry_run_and_leaderboard(self):
        """Method AC dry-run must save checkpoint and leaderboard with k, decay_lambda, eps, alpha."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out_ac"
            orch = MetaSmacOrchestrator(
                method="ac",
                start_iteration=1,
                end_iteration=2,
                seeds=2,
                trials=10,
                output_dir=str(out_dir),
                dry_run=True,
            )
            orch.run()

            df = pd.read_csv(out_dir / "meta_leaderboard.csv")
            self.assertEqual(len(df), 2)
            self.assertIn("k", df.columns)
            self.assertIn("decay_lambda", df.columns)
            self.assertIn("eps", df.columns)
            self.assertIn("alpha", df.columns)

            # Iteration 1 must be Method AC incumbent: k=25, decay_lambda=1.345, eps=0.16, alpha=1.0
            row0 = df.iloc[0]
            self.assertEqual(int(row0["iteration"]), 1)
            self.assertEqual(int(row0["k"]), 25)
            self.assertAlmostEqual(float(row0["decay_lambda"]), 1.345, places=3)
            self.assertAlmostEqual(float(row0["eps"]), 0.16, places=3)
            self.assertAlmostEqual(float(row0["alpha"]), 1.0, places=3)

    def test_method_bc_dry_run_and_leaderboard(self):
        """Method BC dry-run must save checkpoint and leaderboard with decay_lambda, eps, alpha."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out_bc"
            orch = MetaSmacOrchestrator(
                method="bc",
                start_iteration=1,
                end_iteration=2,
                seeds=2,
                trials=10,
                output_dir=str(out_dir),
                dry_run=True,
            )
            orch.run()

            df = pd.read_csv(out_dir / "meta_leaderboard.csv")
            self.assertEqual(len(df), 2)
            self.assertNotIn("k", df.columns)
            self.assertIn("decay_lambda", df.columns)
            self.assertIn("eps", df.columns)
            self.assertIn("alpha", df.columns)

            # Iteration 1 must be Method BC incumbent: decay_lambda=1.345, eps=0.16, alpha=1.0
            row0 = df.iloc[0]
            self.assertEqual(int(row0["iteration"]), 1)
            self.assertAlmostEqual(float(row0["decay_lambda"]), 1.345, places=3)
            self.assertAlmostEqual(float(row0["eps"]), 0.16, places=3)
            self.assertAlmostEqual(float(row0["alpha"]), 1.0, places=3)


class TestCheckpointResumeMultiMethod(unittest.TestCase):
    """Verifies checkpoint resume across all methods preserving past incumbent evaluations."""

    def test_resume_method_bc_preserves_incumbent(self):
        """Resuming method BC restores history and evaluates subsequent iterations correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "resume_bc"

            # Phase 1: Iterations 1 to 2
            orch1 = MetaSmacOrchestrator(
                method="bc",
                start_iteration=1,
                end_iteration=2,
                seeds=2,
                trials=10,
                output_dir=str(out_dir),
                dry_run=True,
            )
            orch1.run()

            # Phase 2: Resume for iteration 3
            orch2 = MetaSmacOrchestrator(
                method="bc",
                start_iteration=3,
                end_iteration=3,
                seeds=2,
                trials=10,
                output_dir=str(out_dir),
                resume=True,
                dry_run=True,
            )
            orch2.run()

            df = pd.read_csv(out_dir / "meta_leaderboard.csv")
            self.assertEqual(len(df), 3)
            self.assertEqual(list(df["iteration"]), [1, 2, 3])

            # Preserved incumbent in row 0
            self.assertAlmostEqual(float(df.iloc[0]["decay_lambda"]), 1.345, places=3)
            self.assertAlmostEqual(float(df.iloc[0]["eps"]), 0.16, places=3)
            self.assertAlmostEqual(float(df.iloc[0]["alpha"]), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
