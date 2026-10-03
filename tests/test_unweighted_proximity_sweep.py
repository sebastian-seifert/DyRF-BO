"""Test suite for Milestone 3: Dedicated Unweighted Proximity Sweep Infrastructure & Analysis.

Strict TDD verification of:
1. scripts/generate_unweighted_proximity_sweep_tasks.py:
   - CLI parser (--mode, --output-file, --workers, --output-dir, --summary-dir)
   - Grid configurations:
     * pilot: 8 tasks (D in [2, 16], N=112, functions ['sphere', 'ackley'], strategies ['natural', 'stratified'], seed 0)
     * comparison: 32 tasks (D in [2, 5, 16, 32], N in [112, 224], 4 functions, 2 strategies, seed 0)
     * full: 64 tasks (balanced orthogonal sweep)
   - Executable command syntax targeting scripts/run_extrapolation_experiment.py
   - File permissions (chmod +x)
2. scripts/run_unweighted_proximity_sweep_local.py:
   - CLI parser and ProcessPoolExecutor concurrency
   - Dry-run mode
   - Progress logging, execution timing, success/failure counting
   - Parallel execution of tasks
   - File permissions (chmod +x)
3. scripts/aggregate_unweighted_proximity_results.py:
   - Generation of table_unweighted_vs_weighted_scorecard.md and .csv
   - Head-to-head comparisons: prox_a_unweighted vs prox_a, plcb_unweighted vs plcb
   - Stratifications: Low-D (D <= 5) and High-D (D >= 16)
   - Evaluated metrics: Spearman rho(d_norm, U), Spearman rho(d_inf, U), Delta(rho), normalized rank monotonicity
   - File permissions (chmod +x)
4. SLURM dispatchers:
   - scripts/submit_unweighted_proximity_sweep_array.sbatch (#SBATCH -p ai, chmod +x)
   - scripts/submit_unweighted_proximity_sweep.sh (chmod +x)
"""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Test 1: Task Generator
# ---------------------------------------------------------------------------

class TestGenerateUnweightedProximitySweepTasks:
    """Tests for scripts/generate_unweighted_proximity_sweep_tasks.py."""

    SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_unweighted_proximity_sweep_tasks.py"

    def test_script_exists_and_executable(self):
        assert self.SCRIPT_PATH.exists(), f"{self.SCRIPT_PATH} does not exist"
        mode = self.SCRIPT_PATH.stat().st_mode
        assert bool(mode & stat.S_IXUSR), f"{self.SCRIPT_PATH} is not executable (chmod +x required)"

    def test_cli_parser_options(self):
        from scripts.generate_unweighted_proximity_sweep_tasks import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--mode", "pilot",
            "--output-file", "results/test_tasks.txt",
            "--workers", "4",
            "--output-dir", "results/test_raw",
            "--summary-dir", "results/test_summaries",
        ])
        assert args.mode == "pilot"
        assert args.output_file == "results/test_tasks.txt"
        assert args.workers == 4
        assert args.output_dir == "results/test_raw"
        assert args.summary_dir == "results/test_summaries"

    def test_pilot_grid_generates_8_tasks(self, tmp_path):
        from scripts.generate_unweighted_proximity_sweep_tasks import generate_tasks

        out_file = tmp_path / "pilot_tasks.txt"
        tasks = generate_tasks(mode="pilot", output_file=out_file)

        assert len(tasks) == 8, f"Expected 8 pilot tasks, got {len(tasks)}"
        assert out_file.exists()

        # Parse and verify content of all tasks
        dimensions = set()
        n_trains = set()
        functions = set()
        strategies = set()
        seeds = set()

        for task in tasks:
            assert "scripts/run_extrapolation_experiment.py" in task
            parts = task.split()
            dim = int(parts[parts.index("--dimension") + 1])
            n_train = int(parts[parts.index("--n-train") + 1])
            func = parts[parts.index("--function") + 1]
            strat = parts[parts.index("--strategy") + 1]
            seed = int(parts[parts.index("--seed") + 1])

            dimensions.add(dim)
            n_trains.add(n_train)
            functions.add(func)
            strategies.add(strat)
            seeds.add(seed)

        assert dimensions == {2, 16}
        assert n_trains == {112}
        assert functions == {"sphere", "ackley"}
        assert strategies == {"natural", "stratified"}
        assert seeds == {0}

    def test_comparison_grid_generates_32_tasks(self, tmp_path):
        from scripts.generate_unweighted_proximity_sweep_tasks import generate_tasks

        out_file = tmp_path / "comparison_tasks.txt"
        tasks = generate_tasks(mode="comparison", output_file=out_file)

        assert len(tasks) == 32, f"Expected 32 comparison tasks, got {len(tasks)}"
        assert out_file.exists()

        dimensions = set()
        n_trains = set()
        functions = set()
        strategies = set()
        seeds = set()

        for task in tasks:
            assert "scripts/run_extrapolation_experiment.py" in task
            parts = task.split()
            dimensions.add(int(parts[parts.index("--dimension") + 1]))
            n_trains.add(int(parts[parts.index("--n-train") + 1]))
            functions.add(parts[parts.index("--function") + 1])
            strategies.add(parts[parts.index("--strategy") + 1])
            seeds.add(int(parts[parts.index("--seed") + 1]))

        assert dimensions == {2, 5, 16, 32}
        assert n_trains == {112, 224}
        assert functions == {"sphere", "ackley", "rastrigin", "rosenbrock"}
        assert strategies == {"natural", "stratified"}
        assert seeds == {0}

    def test_full_grid_generates_64_tasks(self, tmp_path):
        from scripts.generate_unweighted_proximity_sweep_tasks import generate_tasks

        out_file = tmp_path / "full_tasks.txt"
        tasks = generate_tasks(mode="full", output_file=out_file)

        assert len(tasks) == 64, f"Expected 64 full tasks, got {len(tasks)}"
        assert out_file.exists()

        dimensions = set()
        n_trains = set()
        functions = set()
        strategies = set()
        seeds = set()

        for task in tasks:
            assert "scripts/run_extrapolation_experiment.py" in task
            parts = task.split()
            dimensions.add(int(parts[parts.index("--dimension") + 1]))
            n_trains.add(int(parts[parts.index("--n-train") + 1]))
            functions.add(parts[parts.index("--function") + 1])
            strategies.add(parts[parts.index("--strategy") + 1])
            seeds.add(int(parts[parts.index("--seed") + 1]))

        assert dimensions == {2, 5, 16, 32}
        assert n_trains == {112, 224}
        assert functions == {"sphere", "ackley", "rastrigin", "rosenbrock"}
        assert strategies == {"natural", "stratified"}
        assert seeds == {0}


# ---------------------------------------------------------------------------
# Test 2: Local Runner
# ---------------------------------------------------------------------------

class TestRunUnweightedProximitySweepLocal:
    """Tests for scripts/run_unweighted_proximity_sweep_local.py."""

    SCRIPT_PATH = REPO_ROOT / "scripts" / "run_unweighted_proximity_sweep_local.py"

    def test_script_exists_and_executable(self):
        assert self.SCRIPT_PATH.exists(), f"{self.SCRIPT_PATH} does not exist"
        mode = self.SCRIPT_PATH.stat().st_mode
        assert bool(mode & stat.S_IXUSR), f"{self.SCRIPT_PATH} is not executable (chmod +x required)"

    def test_cli_parser_options(self):
        from scripts.run_unweighted_proximity_sweep_local import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--task-file", "results/test_tasks.txt",
            "--max-workers", "4",
            "--dry-run",
            "--limit", "10",
        ])
        assert args.task_file == "results/test_tasks.txt"
        assert args.max_workers == 4
        assert args.dry_run is True
        assert args.limit == 10

    def test_dry_run_execution(self, tmp_path):
        from scripts.run_unweighted_proximity_sweep_local import run_local_sweep

        task_file = tmp_path / "tasks.txt"
        task_file.write_text(
            "python scripts/run_extrapolation_experiment.py --dimension 2 --n-train 112 --function sphere --strategy natural --seed 0\n"
            "python scripts/run_extrapolation_experiment.py --dimension 16 --n-train 112 --function ackley --strategy stratified --seed 0\n"
        )

        res = run_local_sweep(task_file=task_file, max_workers=2, dry_run=True)
        assert res["total_tasks"] == 2
        assert res["executed_tasks"] == 2
        assert res["succeeded_tasks"] == 2
        assert res["failed_tasks"] == 0
        assert len(res["results"]) == 2
        assert all(r["status"] == "dry_run" for r in res["results"])
        assert res["elapsed_seconds"] >= 0.0

    def test_concurrent_execution_process_pool(self, tmp_path):
        from scripts.run_unweighted_proximity_sweep_local import run_local_sweep

        task_file = tmp_path / "mock_tasks.txt"
        # Two fast commands that run via python -c
        task_file.write_text(
            f"{sys.executable} -c 'import time; time.sleep(0.05); print(\"Task 1 done\")'\n"
            f"{sys.executable} -c 'import time; time.sleep(0.05); print(\"Task 2 done\")'\n"
        )

        res = run_local_sweep(task_file=task_file, max_workers=2, dry_run=False)
        assert res["total_tasks"] == 2
        assert res["executed_tasks"] == 2
        assert res["succeeded_tasks"] == 2
        assert res["failed_tasks"] == 0
        assert all(r["status"] == "success" for r in res["results"])

    def test_handles_failing_task(self, tmp_path):
        from scripts.run_unweighted_proximity_sweep_local import run_local_sweep

        task_file = tmp_path / "fail_tasks.txt"
        task_file.write_text(
            f"{sys.executable} -c 'import sys; sys.exit(0)'\n"
            f"{sys.executable} -c 'import sys; sys.exit(42)'\n"
        )

        res = run_local_sweep(task_file=task_file, max_workers=2, dry_run=False)
        assert res["total_tasks"] == 2
        assert res["executed_tasks"] == 2
        assert res["succeeded_tasks"] == 1
        assert res["failed_tasks"] == 1
        assert res["results"][1]["exit_code"] == 42


# ---------------------------------------------------------------------------
# Test 3: Aggregator and Head-to-Head Scorecards
# ---------------------------------------------------------------------------

class TestAggregateUnweightedProximityResults:
    """Tests for scripts/aggregate_unweighted_proximity_results.py."""

    SCRIPT_PATH = REPO_ROOT / "scripts" / "aggregate_unweighted_proximity_results.py"

    def test_script_exists_and_executable(self):
        assert self.SCRIPT_PATH.exists(), f"{self.SCRIPT_PATH} does not exist"
        mode = self.SCRIPT_PATH.stat().st_mode
        assert bool(mode & stat.S_IXUSR), f"{self.SCRIPT_PATH} is not executable (chmod +x required)"

    @pytest.fixture
    def mock_results_dir(self, tmp_path):
        """Generate synthetic raw parquet files and summary JSONs."""
        raw_dir = tmp_path / "raw"
        summaries_dir = tmp_path / "summaries"
        raw_dir.mkdir(parents=True)
        summaries_dir.mkdir(parents=True)

        # Create synthetic datasets for D=2, 5 (Low-D) and D=16, 32 (High-D)
        configs = [
            (2, 112, "sphere", "natural", 0),
            (5, 112, "ackley", "stratified", 0),
            (16, 112, "rastrigin", "natural", 0),
            (32, 112, "rosenbrock", "stratified", 0),
        ]

        np.random.seed(42)
        for dim, n_train, func, strat, seed in configs:
            M = 100
            d_norm = np.linspace(0.01, 1.0, M)
            d_inf = d_norm * np.random.uniform(0.8, 1.2, M)
            # Simulated uncertainties
            # Unweighted prox: highly correlated with distance
            u_prox_a_unweighted = d_norm + 0.05 * np.random.randn(M)
            # Weighted prox: slightly corrupted correlation
            u_prox_a_weighted = d_norm + 0.25 * np.random.randn(M)
            # PLCB unweighted
            u_plcb_unweighted = u_prox_a_unweighted + 0.1
            # PLCB weighted
            u_plcb_weighted = u_prox_a_weighted + 0.1

            df = pd.DataFrame({
                "point_id": np.arange(M),
                "d_norm": d_norm,
                "d_inf": d_inf,
                "d_rel": d_norm / np.sqrt(dim),
                "stratum": (d_norm * 4).astype(int).clip(0, 3),
                "y_true": np.zeros(M),
                "y_hat": np.zeros(M),
                "abs_error": np.abs(np.random.randn(M)),
                "u_prox_a_unweighted_half": u_prox_a_unweighted,
                "u_prox_a_unweighted_lower": u_prox_a_unweighted * 0.9,
                "u_prox_a_weighted_half": u_prox_a_weighted,
                "u_prox_a_weighted_lower": u_prox_a_weighted * 0.9,
                "u_prox_a_half": u_prox_a_weighted,
                "u_prox_a_lower": u_prox_a_weighted * 0.9,
                "u_plcb_unweighted_half": u_plcb_unweighted,
                "u_plcb_unweighted_lower": u_plcb_unweighted * 0.9,
                "u_plcb_weighted_half": u_plcb_weighted,
                "u_plcb_weighted_lower": u_plcb_weighted * 0.9,
                "u_plcb_half": u_plcb_weighted,
                "u_plcb_lower": u_plcb_weighted * 0.9,
            })
            parquet_name = f"extrapolation_{func}_d{dim}_n{n_train}_{strat}_smac_default_s{seed}.parquet"
            df.to_parquet(raw_dir / parquet_name, index=False)

            summary = {
                "dimension": dim,
                "n_train": n_train,
                "function_name": func,
                "sampling_strategy": strat,
                "seed": seed,
                "surrogate": "smac_default",
                "global": {
                    "u_prox_a_unweighted_half": {"spearman_dist": 0.85, "spearman_err": 0.1},
                    "u_prox_a_weighted_half": {"spearman_dist": 0.70, "spearman_err": 0.1},
                    "u_plcb_unweighted_half": {"spearman_dist": 0.85, "spearman_err": 0.1},
                    "u_plcb_weighted_half": {"spearman_dist": 0.70, "spearman_err": 0.1},
                },
            }
            summary_name = f"summary_{func}_d{dim}_n{n_train}_{strat}_smac_default_s{seed}.json"
            with open(summaries_dir / summary_name, "w", encoding="utf-8") as f:
                json.dump(summary, f)

        return tmp_path

    def test_aggregate_scorecards_generation(self, mock_results_dir):
        from scripts.aggregate_unweighted_proximity_results import generate_unweighted_scorecard

        output_dir = mock_results_dir / "analysis"
        generate_unweighted_scorecard(
            raw_dir=mock_results_dir / "raw",
            summary_dir=mock_results_dir / "summaries",
            output_dir=output_dir,
        )

        md_path = output_dir / "table_unweighted_vs_weighted_scorecard.md"
        csv_path = output_dir / "table_unweighted_vs_weighted_scorecard.csv"

        assert md_path.exists(), f"Expected {md_path} to exist"
        assert csv_path.exists(), f"Expected {csv_path} to exist"

        # Check CSV content and schema
        df = pd.read_csv(csv_path)
        required_cols = [
            "dimension_group",
            "comparison",
            "spearman_norm_unweighted",
            "spearman_norm_weighted",
            "spearman_inf_unweighted",
            "spearman_inf_weighted",
            "delta_rho_norm",
            "delta_rho_inf",
            "norm_rank_monotonicity_unweighted",
            "norm_rank_monotonicity_weighted",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column '{col}' in scorecard CSV"

        # Check comparisons
        comps = set(df["comparison"].unique())
        assert any("prox_a" in c for c in comps), "Scorecard must compare prox_a (unweighted vs weighted)"
        assert any("plcb" in c for c in comps), "Scorecard must compare plcb (unweighted vs weighted)"

        # Check dimension stratifications
        dim_groups = set(df["dimension_group"].unique())
        assert any("Low-D" in g for g in dim_groups), "Scorecard must stratify Low-D (D <= 5)"
        assert any("High-D" in g for g in dim_groups), "Scorecard must stratify High-D (D >= 16)"

        # Check Markdown content
        md_text = md_path.read_text(encoding="utf-8")
        assert "prox_a_unweighted" in md_text
        assert "plcb_unweighted" in md_text
        assert "Low-D" in md_text or "D <= 5" in md_text
        assert "High-D" in md_text or "D >= 16" in md_text


# ---------------------------------------------------------------------------
# Test 4: SLURM Dispatchers
# ---------------------------------------------------------------------------

class TestSlurmDispatchers:
    """Tests for SLURM submission scripts."""

    ARRAY_SBATCH = REPO_ROOT / "scripts" / "submit_unweighted_proximity_sweep_array.sbatch"
    RUN_SH = REPO_ROOT / "scripts" / "submit_unweighted_proximity_sweep.sh"

    def test_array_sbatch_exists_and_executable(self):
        assert self.ARRAY_SBATCH.exists(), f"{self.ARRAY_SBATCH} does not exist"
        mode = self.ARRAY_SBATCH.stat().st_mode
        assert bool(mode & stat.S_IXUSR), f"{self.ARRAY_SBATCH} is not executable (chmod +x required)"

    def test_array_sbatch_header_and_partition(self):
        content = self.ARRAY_SBATCH.read_text(encoding="utf-8")
        assert "#SBATCH -p ai" in content or "#SBATCH --partition=ai" in content
        assert "results/unweighted_proximity_sweep" in content

    def test_submit_sh_exists_and_executable(self):
        assert self.RUN_SH.exists(), f"{self.RUN_SH} does not exist"
        mode = self.RUN_SH.stat().st_mode
        assert bool(mode & stat.S_IXUSR), f"{self.RUN_SH} is not executable (chmod +x required)"

    def test_submit_sh_invokes_task_generation(self):
        content = self.RUN_SH.read_text(encoding="utf-8")
        assert "scripts/generate_unweighted_proximity_sweep_tasks.py" in content
        assert "submit_unweighted_proximity_sweep_array.sbatch" in content
