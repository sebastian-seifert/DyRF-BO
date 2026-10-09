"""Unit and Integration Tests for Multi-Method Hydra Task Command Generation.

Verifies:
1. Method B:
   - Exactly 90 tasks referencing +optimizer=smac20_proximity_b_lcb and optimizer_id=SMAC20_ProximityB_LCB_iter{iter:03d}
   - Injects eps and decay_lambda overrides
   - No k or k_warmup overrides
2. Method AC:
   - Exactly 90 tasks referencing +optimizer=smac20_proximity_ac_lcb and optimizer_id=SMAC20_ProximityAC_LCB_iter{iter:03d}
   - Invariant enforcement: k == k_warmup with overrides for k, k_warmup, decay_lambda, eps, and alpha
3. Method BC:
   - Exactly 90 tasks referencing +optimizer=smac20_proximity_bc_lcb and optimizer_id=SMAC20_ProximityBC_LCB_iter{iter:03d}
   - Injects eps, decay_lambda, and alpha overrides
   - No k or k_warmup overrides
4. Task Filtering & OOB Targeting:
   - Filters out broken HPOBench NAS benchmarks across all methods
   - Only targets OOB configurations (no CV variants)
5. Method A Backward Compatibility:
   - Exact compatibility with existing signature and output format
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry
from scripts.meta_smac_task_generator import generate_iteration_tasks


class TestMetaSmacMultiMethodGenerator(unittest.TestCase):
    """Test suite for multi-method task generation (Methods A, B, AC, BC)."""

    def setUp(self):
        self.dev_tasks = CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)
        self.assertEqual(len(self.dev_tasks), 18)

    def test_method_a_backward_compatibility(self):
        """Method A retains exact backward-compatible task commands and defaults."""
        cfg_a = {"k": 25, "decay_lambda": 1.345, "eps": 0.1678}
        tasks = generate_iteration_tasks(
            config=cfg_a,
            iteration=1,
            seeds=5,
            trials=100,
        )

        self.assertEqual(len(tasks), 90)
        for t in tasks:
            self.assertIn("+optimizer=smac20_proximity_lcb", t)
            self.assertIn("optimizer_id=SMAC20_ProximityLCB_iter001", t)
            self.assertIn("optimizer_container_id=SMAC20_ProximityLCB", t)
            self.assertIn("++optimizer.acq_func_kwargs.k=25", t)
            self.assertIn("++optimizer.acq_func_kwargs.level=0.95", t)
            self.assertIn("++optimizer.acq_func_kwargs.eps=0.1678", t)
            self.assertIn("++optimizer.smac_cfg.model_kwargs.uncertainty_func=proximity_b", t)
            self.assertIn("++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda=1.345", t)
            self.assertIn("task.optimization_resources.n_trials=100", t)
            self.assertNotIn("_cv", t)
            self.assertNotIn("tabular_nas", t)

    def test_method_b_task_generation(self):
        """Method B generates 90 tasks with smac20_proximity_b_lcb, injecting eps and decay_lambda."""
        cfg_b = {"decay_lambda": 0.85, "eps": 0.12}
        tasks = generate_iteration_tasks(
            config=cfg_b,
            iteration=3,
            seeds=5,
            trials=100,
            method="b",
        )

        self.assertEqual(len(tasks), 90)
        for t in tasks:
            self.assertIn("+optimizer=smac20_proximity_b_lcb", t)
            self.assertIn("optimizer_id=SMAC20_ProximityB_LCB_iter003", t)
            self.assertIn("optimizer_container_id=SMAC20_ProximityB_LCB", t)
            self.assertIn("++optimizer.acq_func_kwargs.eps=0.12", t)
            self.assertIn("++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda=0.85", t)
            # Invariant: Method B does not use k or alpha overrides
            self.assertNotIn("++optimizer.acq_func_kwargs.k=", t)
            self.assertNotIn("++optimizer.acq_func_kwargs.k_warmup=", t)
            self.assertNotIn("++optimizer.smac_cfg.model_kwargs.extractor_kwargs.alpha=", t)
            # OOB model only
            self.assertNotIn("_cv", t)
            self.assertNotIn("tabular_nas", t)

    def test_method_ac_task_generation_and_k_invariant(self):
        """Method AC generates 90 tasks with smac20_proximity_ac_lcb, enforcing k == k_warmup and overrides."""
        cfg_ac = {"k": 18, "decay_lambda": 1.15, "eps": 0.08, "alpha": 1.4}
        tasks = generate_iteration_tasks(
            config=cfg_ac,
            iteration=7,
            seeds=5,
            trials=100,
            method="ac",
        )

        self.assertEqual(len(tasks), 90)
        for t in tasks:
            self.assertIn("+optimizer=smac20_proximity_ac_lcb", t)
            self.assertIn("optimizer_id=SMAC20_ProximityAC_LCB_iter007", t)
            self.assertIn("optimizer_container_id=SMAC20_ProximityAC_LCB", t)
            # Invariant: k == k_warmup
            self.assertIn("++optimizer.acq_func_kwargs.k=18", t)
            self.assertIn("++optimizer.acq_func_kwargs.k_warmup=18", t)
            self.assertIn("++optimizer.acq_func_kwargs.eps=0.08", t)
            self.assertIn("++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda=1.15", t)
            self.assertIn("++optimizer.smac_cfg.model_kwargs.extractor_kwargs.alpha=1.4", t)
            self.assertIn("++optimizer.smac_cfg.model_kwargs.uncertainty_func=proximity_ac", t)
            # OOB model only
            self.assertNotIn("_cv", t)
            self.assertNotIn("tabular_nas", t)

    def test_method_bc_task_generation(self):
        """Method BC generates 90 tasks with smac20_proximity_bc_lcb, injecting eps, decay_lambda, and alpha."""
        cfg_bc = {"decay_lambda": 0.95, "eps": 0.15, "alpha": 0.75}
        tasks = generate_iteration_tasks(
            config=cfg_bc,
            iteration=12,
            seeds=5,
            trials=100,
            method="bc",
        )

        self.assertEqual(len(tasks), 90)
        for t in tasks:
            self.assertIn("+optimizer=smac20_proximity_bc_lcb", t)
            self.assertIn("optimizer_id=SMAC20_ProximityBC_LCB_iter012", t)
            self.assertIn("optimizer_container_id=SMAC20_ProximityBC_LCB", t)
            self.assertIn("++optimizer.acq_func_kwargs.eps=0.15", t)
            self.assertIn("++optimizer.smac_cfg.model_kwargs.extractor_kwargs.decay_lambda=0.95", t)
            self.assertIn("++optimizer.smac_cfg.model_kwargs.extractor_kwargs.alpha=0.75", t)
            self.assertIn("++optimizer.smac_cfg.model_kwargs.uncertainty_func=proximity_bc", t)
            # Method BC does not use k or k_warmup overrides
            self.assertNotIn("++optimizer.acq_func_kwargs.k=", t)
            self.assertNotIn("++optimizer.acq_func_kwargs.k_warmup=", t)
            # OOB model only
            self.assertNotIn("_cv", t)
            self.assertNotIn("tabular_nas", t)

    def test_all_methods_dev_task_distribution_and_nas_exclusion(self):
        """All methods must distribute across exactly 18 dev tasks (5 seeds each) and exclude NAS."""
        methods_configs = [
            ("a", {"k": 25, "decay_lambda": 1.345, "eps": 0.1678}),
            ("b", {"decay_lambda": 1.345, "eps": 0.16}),
            ("ac", {"k": 25, "decay_lambda": 1.345, "eps": 0.16, "alpha": 1.0}),
            ("bc", {"decay_lambda": 1.345, "eps": 0.16, "alpha": 1.0}),
        ]

        for method, cfg in methods_configs:
            tasks = generate_iteration_tasks(config=cfg, iteration=1, seeds=5, method=method)
            self.assertEqual(len(tasks), 90)

            # Each of the 18 working dev tasks must appear exactly 5 times
            for task_arg in self.dev_tasks:
                matching = [t for t in tasks if task_arg in t]
                self.assertEqual(
                    len(matching),
                    5,
                    f"Method {method}: task {task_arg} must appear 5 times",
                )

            # No broken NAS benchmarks
            for t in tasks:
                self.assertNotIn("tabular_nas", t)
                self.assertNotIn("cv", t)

    def test_output_file_writing(self):
        """Tasks can be written to an output file when specified."""
        cfg = {"decay_lambda": 1.0, "eps": 0.1}
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "tasks_method_b.txt"
            tasks = generate_iteration_tasks(
                config=cfg,
                iteration=2,
                seeds=5,
                method="b",
                output_file=str(out_file),
            )
            self.assertTrue(out_file.exists())
            written_lines = [line.strip() for line in out_file.read_text().splitlines() if line.strip()]
            self.assertEqual(len(written_lines), 90)
            self.assertEqual(written_lines, tasks)

    def test_invalid_method_raises_error(self):
        """Unknown method name should raise ValueError."""
        cfg = {"eps": 0.1}
        with self.assertRaises(ValueError):
            generate_iteration_tasks(config=cfg, iteration=1, method="invalid_method")


if __name__ == "__main__":
    unittest.main()
