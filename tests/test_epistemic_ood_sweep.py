import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from synthetic_functions import (
    get_all_normal_functions,
    get_special_functions,
)
from data_generator import generate_data


class TestEpistemicOODSweepBenchmark(unittest.TestCase):
    """Test suite for large-scale OOD Detection & Epistemic UQ Benchmark Sweep."""

    def test_all_51_benchmark_functions_exist_and_generate_valid_data(self):
        """Asserts all 41 normal and 10 special functions exist and return valid data with binary labels."""
        normal_funcs = get_all_normal_functions()
        special_funcs = get_special_functions()

        self.assertEqual(
            len(normal_funcs),
            41,
            f"Expected exactly 41 normal functions (1D-15D), found {len(normal_funcs)}"
        )
        self.assertEqual(
            len(special_funcs),
            10,
            f"Expected exactly 10 special functions, found {len(special_funcs)}"
        )

        expected_special_names = {
            "ackley_2d",
            "rosenbrock_2d",
            "ackley_4d",
            "rosenbrock_4d",
            "friedman_6d",
            "hartmann_6d",
            "friedman_10d",
            "branin",
            "hartmann3",
            "hartmann6",
        }
        self.assertEqual(set(special_funcs.keys()), expected_special_names)

        # Ensure no name collision between normal and special
        overlap = set(normal_funcs.keys()).intersection(set(special_funcs.keys()))
        self.assertEqual(len(overlap), 0, f"Overlapping function names found: {overlap}")

        all_funcs = {**normal_funcs, **special_funcs}
        self.assertEqual(len(all_funcs), 51, "Total benchmark functions must be 51")

        # Verify all function configurations and test generate_data
        for name, cfg in all_funcs.items():
            self.assertIn("func", cfg, f"{name} missing 'func'")
            self.assertIn("gap", cfg, f"{name} missing 'gap'")
            self.assertIn("range", cfg, f"{name} missing 'range'")
            self.assertIn("bounds", cfg, f"{name} missing 'bounds'")
            self.assertTrue(callable(cfg["func"]), f"{name} 'func' must be callable")

            gap = cfg["gap"]
            x_range = cfg["range"]
            bounds = cfg["bounds"]

            self.assertIsInstance(gap, tuple, f"{name} gap must be tuple")
            self.assertIsInstance(x_range, tuple, f"{name} range must be tuple")
            self.assertIsInstance(bounds, tuple, f"{name} bounds must be tuple")
            self.assertLess(gap[0], gap[1], f"{name} gap invalid: {gap}")
            self.assertLess(x_range[0], x_range[1], f"{name} range invalid: {x_range}")

            # Verify generate_data produces valid shapes and ground-truth binary labels
            X_train, y_train, X_test, y_test, y_true_binary = generate_data(
                func_dict=all_funcs,
                func_name=name,
                seed=42,
                gap_type="empty",
            )

            self.assertGreater(len(X_train), 0, f"{name} X_train is empty")
            self.assertGreater(len(X_test), 0, f"{name} X_test is empty")
            self.assertEqual(len(X_train), len(y_train), f"{name} X_train/y_train shape mismatch")
            self.assertEqual(len(X_test), len(y_test), f"{name} X_test/y_test shape mismatch")
            self.assertEqual(len(X_test), len(y_true_binary), f"{name} X_test/y_true_binary mismatch")

            # Ground truth binary labels should contain both ID (0) and OOD (1)
            unique_labels = set(np.unique(y_true_binary))
            self.assertTrue(unique_labels.issubset({0, 1}), f"{name} labels must be 0 or 1: {unique_labels}")
            self.assertIn(0, unique_labels, f"{name} missing ID labels (0)")
            self.assertIn(1, unique_labels, f"{name} missing OOD labels (1)")

            self.assertFalse(np.isnan(X_train).any(), f"{name} X_train has NaN")
            self.assertFalse(np.isnan(y_train).any(), f"{name} y_train has NaN")
            self.assertFalse(np.isnan(X_test).any(), f"{name} X_test has NaN")
            self.assertFalse(np.isnan(y_test).any(), f"{name} y_test has NaN")

    def test_task_generator_generates_2040_tasks(self):
        """Asserts task generator produces exactly 2040 tasks (51 x 2 x 2 x 10)."""
        from scripts.generate_epistemic_ood_sweep_tasks import (
            generate_epistemic_ood_sweep_tasks,
            TASK_FILE,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            test_task_file = os.path.join(tmpdir, "tasks.txt")
            tasks = generate_epistemic_ood_sweep_tasks(
                output_file=test_task_file,
                output_dir="results/epistemic_ood_sweep"
            )

            self.assertEqual(
                len(tasks),
                2040,
                f"Expected exactly 2040 tasks, got {len(tasks)}"
            )

            # Check file on disk
            with open(test_task_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(lines), 2040)

            # Validate coverage
            functions_found = set()
            gap_types_found = set()
            approaches_found = set()
            seeds_found = set()

            for t in tasks:
                self.assertIn("--func_name=", t)
                self.assertIn("--gap_type=", t)
                self.assertIn("--approach=", t)
                self.assertIn("--seed=", t)

                # Parse flags
                parts = t.split()
                for p in parts:
                    if p.startswith("--func_name="):
                        functions_found.add(p.split("=")[1])
                    elif p.startswith("--gap_type="):
                        gap_types_found.add(p.split("=")[1])
                    elif p.startswith("--approach="):
                        approaches_found.add(p.split("=")[1])
                    elif p.startswith("--seed="):
                        seeds_found.add(int(p.split("=")[1]))

            self.assertEqual(len(functions_found), 51)
            self.assertEqual(gap_types_found, {"empty", "sparse"})
            self.assertEqual(approaches_found, {"standard_disagreement", "distance_evidential"})
            self.assertEqual(seeds_found, set(range(1, 11)))

    def test_runner_dry_run_and_smoke(self):
        """Asserts runner works cleanly for dry-run/smoke tasks and local multiprocessing."""
        from scripts.run_epistemic_ood_sweep_local import (
            run_epistemic_ood_task,
            run_epistemic_ood_sweep_local,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Test single task dry-run
            res_dry = run_epistemic_ood_task(
                func_name="sin",
                gap_type="empty",
                approach="standard_disagreement",
                seed=1,
                output_dir=tmpdir,
                dry_run=True,
            )
            self.assertEqual(res_dry["func_name"], "sin")
            self.assertEqual(res_dry["gap_type"], "empty")
            self.assertEqual(res_dry["approach"], "standard_disagreement")
            self.assertIn("auroc", res_dry)
            self.assertIn("fpr95", res_dry)
            self.assertIn("aupr", res_dry)

            # 2. Test single task real execution (smoke) for both approaches
            for approach in ["standard_disagreement", "distance_evidential"]:
                res_real = run_epistemic_ood_task(
                    func_name="sin",
                    gap_type="empty",
                    approach=approach,
                    seed=1,
                    output_dir=tmpdir,
                    dry_run=False,
                    n_estimators=10,  # Fast smoke test
                )
                self.assertEqual(res_real["approach"], approach)
                self.assertGreaterEqual(res_real["auroc"], 0.0)
                self.assertLessEqual(res_real["auroc"], 1.0)
                self.assertIn("aurc", res_real)
                self.assertIn("spearman", res_real)

            # 3. Test multi-task local execution harness with ProcessPoolExecutor
            smoke_tasks = [
                {
                    "func_name": "sin",
                    "gap_type": "empty",
                    "approach": "standard_disagreement",
                    "seed": 1,
                    "dry_run": True,
                },
                {
                    "func_name": "poly",
                    "gap_type": "sparse",
                    "approach": "distance_evidential",
                    "seed": 2,
                    "dry_run": True,
                },
            ]
            results = run_epistemic_ood_sweep_local(
                tasks=smoke_tasks,
                output_dir=tmpdir,
                max_workers=2,
            )
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r.get("status") == "success" for r in results))

    def test_parser_separates_normal_and_special_functions(self):
        """Asserts parser correctly generates Table 1 (Normal aggregated by dim) and Table 2 (Special individually)."""
        from scripts.parse_epistemic_ood_sweep_results import parse_epistemic_ood_sweep_results

        with tempfile.TemporaryDirectory() as tmpdir:
            res_dir = os.path.join(tmpdir, "results")
            out_dir = os.path.join(tmpdir, "parsed")
            os.makedirs(res_dir, exist_ok=True)

            # Create synthetic result JSON files:
            # Normal functions:
            # 1D: sin (seeds 1, 2), cos_trend (seed 1)
            # 2D: sin_cos (seed 1)
            # Special functions:
            # ackley_2d (seed 1), branin (seed 1)
            sample_records = [
                {
                    "func_name": "sin",
                    "dim": 1,
                    "gap_type": "empty",
                    "approach": "standard_disagreement",
                    "seed": 1,
                    "auroc": 0.80,
                    "fpr95": 0.30,
                    "aupr": 0.75,
                    "spearman": 0.50,
                    "aurc": 0.20,
                    "oracle_aurc": 0.10,
                    "jsd": 0.40,
                    "mi": 0.30,
                    "nlpd": 1.20,
                    "brier": 0.15,
                },
                {
                    "func_name": "sin",
                    "dim": 1,
                    "gap_type": "empty",
                    "approach": "standard_disagreement",
                    "seed": 2,
                    "auroc": 0.82,
                    "fpr95": 0.28,
                    "aupr": 0.77,
                    "spearman": 0.52,
                    "aurc": 0.19,
                    "oracle_aurc": 0.09,
                    "jsd": 0.42,
                    "mi": 0.32,
                    "nlpd": 1.18,
                    "brier": 0.14,
                },
                {
                    "func_name": "cos_trend",
                    "dim": 1,
                    "gap_type": "empty",
                    "approach": "standard_disagreement",
                    "seed": 1,
                    "auroc": 0.78,
                    "fpr95": 0.32,
                    "aupr": 0.73,
                    "spearman": 0.48,
                    "aurc": 0.21,
                    "oracle_aurc": 0.11,
                    "jsd": 0.38,
                    "mi": 0.28,
                    "nlpd": 1.22,
                    "brier": 0.16,
                },
                {
                    "func_name": "sin_cos",
                    "dim": 2,
                    "gap_type": "empty",
                    "approach": "standard_disagreement",
                    "seed": 1,
                    "auroc": 0.85,
                    "fpr95": 0.25,
                    "aupr": 0.80,
                    "spearman": 0.55,
                    "aurc": 0.18,
                    "oracle_aurc": 0.08,
                    "jsd": 0.45,
                    "mi": 0.35,
                    "nlpd": 1.10,
                    "brier": 0.12,
                },
                # Special functions
                {
                    "func_name": "ackley_2d",
                    "dim": 2,
                    "gap_type": "empty",
                    "approach": "standard_disagreement",
                    "seed": 1,
                    "auroc": 0.70,
                    "fpr95": 0.40,
                    "aupr": 0.65,
                    "spearman": 0.40,
                    "aurc": 0.25,
                    "oracle_aurc": 0.15,
                    "jsd": 0.35,
                    "mi": 0.25,
                    "nlpd": 1.30,
                    "brier": 0.18,
                },
                {
                    "func_name": "branin",
                    "dim": 2,
                    "gap_type": "empty",
                    "approach": "distance_evidential",
                    "seed": 1,
                    "auroc": 0.95,
                    "fpr95": 0.10,
                    "aupr": 0.92,
                    "spearman": 0.70,
                    "aurc": 0.10,
                    "oracle_aurc": 0.05,
                    "jsd": 0.60,
                    "mi": 0.50,
                    "nlpd": 0.80,
                    "brier": 0.08,
                },
            ]

            for i, rec in enumerate(sample_records):
                file_path = os.path.join(res_dir, f"result_{i}.json")
                with open(file_path, "w") as f:
                    json.dump(rec, f)

            table1, table2 = parse_epistemic_ood_sweep_results(
                results_dir=res_dir,
                output_dir=out_dir
            )

            # Verify Table 1: Normal functions aggregated by dimension
            self.assertIsNotNone(table1, "Table 1 must not be None")
            self.assertIn("dim", table1.columns)
            self.assertIn("gap_type", table1.columns)
            self.assertIn("approach", table1.columns)
            # Table 1 should NOT have 'func_name' since it's aggregated by dimension
            self.assertNotIn("func_name", table1.columns)

            # Normal functions should only have dims 1 and 2
            self.assertEqual(set(table1["dim"].unique()), {1, 2})

            # Check 1D aggregated AUROC: average of (0.80, 0.82, 0.78) = 0.80
            row_1d = table1[table1["dim"] == 1].iloc[0]
            self.assertAlmostEqual(row_1d["auroc"], 0.80, places=3)

            # Verify Table 2: Special functions listed individually
            self.assertIsNotNone(table2, "Table 2 must not be None")
            self.assertIn("func_name", table2.columns)
            self.assertIn("approach", table2.columns)
            self.assertEqual(set(table2["func_name"].unique()), {"ackley_2d", "branin"})

            # Verify generated CSV and Markdown files exist
            t1_csv = os.path.join(out_dir, "table1_normal_functions_by_dim.csv")
            t1_md = os.path.join(out_dir, "table1_normal_functions_by_dim.md")
            t2_csv = os.path.join(out_dir, "table2_special_functions_individual.csv")
            t2_md = os.path.join(out_dir, "table2_special_functions_individual.md")

            self.assertTrue(os.path.isfile(t1_csv), f"Missing {t1_csv}")
            self.assertTrue(os.path.isfile(t1_md), f"Missing {t1_md}")
            self.assertTrue(os.path.isfile(t2_csv), f"Missing {t2_csv}")
            self.assertTrue(os.path.isfile(t2_md), f"Missing {t2_md}")


if __name__ == "__main__":
    unittest.main()
