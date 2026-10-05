"""Test suite for Milestone 3: Lower-Quantile UQ Sweep Infrastructure, Dispatchers & Downstream Aggregator.

Strict TDD verification of:
1. scripts/generate_lower_quantile_sweep_tasks.py:
   - CLI parser (--mode, --output-file, --split-parts, --output-part1, --output-part2, --workers, etc.)
   - Grid configurations:
     * pilot: 20 tasks (D in [2, 16], N=112, sphere, natural/stratified, seed 0, 5 surrogates)
     * stress: 80 tasks (16 balanced configs x 5 surrogates)
     * full: 9,600 tasks (6 D x 4 N x 4 functions x 2 strategies x 10 seeds x 5 surrogates)
   - Split parts:
     * tasks_part1.txt: exactly 5,000 tasks
     * tasks_part2.txt: exactly 4,600 tasks
   - File permissions (chmod +x)
2. scripts/run_lower_quantile_sweep_local.py:
   - CLI parser and ProcessPoolExecutor concurrency
   - Thread-pinning (OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, OMP_NUM_THREADS=1)
   - Dry-run mode
   - Progress logging, execution timing, success/failure counting
   - Parallel execution of tasks
   - File permissions (chmod +x)
3. scripts/aggregate_lower_quantile_results.py:
   - Reading parquets and summary JSONs
   - Emits progress updates at 10% steps
   - Generates table_lower_quantile_scorecard.csv and table_lower_quantile_scorecard.md
   - Comparing Proximity A, B, AC, BC, PLCB, and non-proximity baselines (SLCB / Hutter, RF-FIRE, Shaker)
   - Statistical evaluation: Spearman correlation, paired Wilcoxon tests, Cliff's delta, Win/Tie/Loss
   - Stratifications: Low-D (D <= 5), High-D (D >= 16), All Dimensions
   - File permissions (chmod +x)
4. SLURM dispatchers:
   - scripts/submit_lower_quantile_sweep_array.sbatch (#SBATCH -p ai, chunk <= 200, chmod +x)
   - scripts/submit_lower_quantile_sweep.sh (master submitter with chunking <= 200, chmod +x)
   - scripts/submit_lower_quantile_sweep_part1.sh (submits tasks 1 to 5,000, chmod +x)
   - scripts/submit_lower_quantile_sweep_part2.sh (submits tasks 5,001 to 9,600, chmod +x)
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

class TestGenerateLowerQuantileSweepTasks:
    """Tests for scripts/generate_lower_quantile_sweep_tasks.py."""

    SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_lower_quantile_sweep_tasks.py"

    def test_script_exists_and_executable(self):
        assert self.SCRIPT_PATH.exists(), f"{self.SCRIPT_PATH} does not exist"
        mode = self.SCRIPT_PATH.stat().st_mode
        assert bool(mode & stat.S_IXUSR), f"{self.SCRIPT_PATH} is not executable (chmod +x required)"

    def test_cli_parser_options(self):
        from scripts.generate_lower_quantile_sweep_tasks import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--pilot",
            "--output-file", "results/test_lower_tasks.txt",
            "--split-parts",
            "--output-part1", "results/test_part1.txt",
            "--output-part2", "results/test_part2.txt",
            "--workers", "4",
            "--output-dir", "results/test_lower_raw",
            "--summary-dir", "results/test_lower_summaries",
            "--dimensions", "2", "3",
            "--n-trains", "112",
            "--functions", "sphere",
            "--strategies", "natural",
            "--seeds", "0", "1",
            "--surrogates", "smac_default", "breiman",
        ])
        assert args.pilot is True
        assert args.output_file == "results/test_lower_tasks.txt"
        assert args.split_parts is True
        assert args.output_part1 == "results/test_part1.txt"
        assert args.output_part2 == "results/test_part2.txt"
        assert args.workers == 4
        assert args.output_dir == "results/test_lower_raw"
        assert args.summary_dir == "results/test_lower_summaries"
        assert args.dimensions == [2, 3]
        assert args.n_trains == [112]
        assert args.functions == ["sphere"]
        assert args.strategies == ["natural"]
        assert args.seeds == [0, 1]
        assert args.surrogates == ["smac_default", "breiman"]

    def test_pilot_grid_generates_20_tasks(self, tmp_path):
        from scripts.generate_lower_quantile_sweep_tasks import generate_tasks

        out_file = tmp_path / "pilot_tasks.txt"
        tasks = generate_tasks(pilot=True, output_file=out_file)

        # 2 D x 1 N x 1 func x 2 strats x 1 seed x 5 surrogates = 20 tasks
        assert len(tasks) == 20, f"Expected 20 pilot tasks, got {len(tasks)}"
        assert out_file.exists()

        dimensions = set()
        n_trains = set()
        functions = set()
        strategies = set()
        seeds = set()
        surrogates = set()

        for task in tasks:
            assert "scripts/run_extrapolation_experiment.py" in task
            assert "--skip-if-exists" in task

            parts = task.split()
            dimensions.add(int(parts[parts.index("--dimension") + 1]))
            n_trains.add(int(parts[parts.index("--n-train") + 1]))
            functions.add(parts[parts.index("--function") + 1])
            strategies.add(parts[parts.index("--strategy") + 1])
            seeds.add(int(parts[parts.index("--seed") + 1]))
            surrogates.add(parts[parts.index("--surrogate") + 1])

        assert dimensions == {2, 16}
        assert n_trains == {112}
        assert functions == {"sphere"}
        assert strategies == {"natural", "stratified"}
        assert seeds == {0}
        assert surrogates == {"smac_default", "mature", "shallow", "coarse", "breiman"}

    def test_stress_grid_generates_80_tasks(self, tmp_path):
        from scripts.generate_lower_quantile_sweep_tasks import generate_tasks

        out_file = tmp_path / "stress_tasks.txt"
        tasks = generate_tasks(stress=True, output_file=out_file)

        assert len(tasks) == 80, f"Expected 80 stress tasks, got {len(tasks)}"
        assert out_file.exists()

        surrogates = set()
        for task in tasks:
            assert "scripts/run_extrapolation_experiment.py" in task
            assert "--skip-if-exists" in task
            parts = task.split()
            surrogates.add(parts[parts.index("--surrogate") + 1])

        assert surrogates == {"smac_default", "mature", "shallow", "coarse", "breiman"}

    def test_full_grid_generates_9600_tasks(self, tmp_path):
        from scripts.generate_lower_quantile_sweep_tasks import generate_tasks

        out_file = tmp_path / "full_tasks.txt"
        tasks = generate_tasks(output_file=out_file)

        # 6 D x 4 N x 4 functions x 2 strategies x 10 seeds x 5 surrogates = 9,600 tasks
        assert len(tasks) == 9600, f"Expected 9600 full tasks, got {len(tasks)}"
        assert out_file.exists()
        assert "scripts/run_extrapolation_experiment.py" in tasks[0]
        assert "--skip-if-exists" in tasks[0]

    def test_split_parts_generates_5000_and_4600_tasks(self, tmp_path):
        from scripts.generate_lower_quantile_sweep_tasks import generate_tasks

        out_file = tmp_path / "tasks.txt"
        tasks = generate_tasks(
            output_file=out_file,
            split_parts=True,
        )
        assert len(tasks) == 9600

        part1_file = tmp_path / "tasks_part1.txt"
        part2_file = tmp_path / "tasks_part2.txt"

        assert part1_file.exists(), f"Expected {part1_file} to exist"
        assert part2_file.exists(), f"Expected {part2_file} to exist"

        lines_part1 = [l.strip() for l in part1_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        lines_part2 = [l.strip() for l in part2_file.read_text(encoding="utf-8").splitlines() if l.strip()]

        assert len(lines_part1) == 5000, f"Expected 5,000 tasks in part 1, got {len(lines_part1)}"
        assert len(lines_part2) == 4600, f"Expected 4,600 tasks in part 2, got {len(lines_part2)}"

        # Ensure no overlap and exact partition
        assert lines_part1 == tasks[:5000]
        assert lines_part2 == tasks[5000:]

    def test_separate_output_paths_part1_part2(self, tmp_path):
        from scripts.generate_lower_quantile_sweep_tasks import generate_tasks

        out_file = tmp_path / "master.txt"
        p1 = tmp_path / "custom_p1.txt"
        p2 = tmp_path / "custom_p2.txt"

        tasks = generate_tasks(
            output_file=out_file,
            split_parts=True,
            output_part1=p1,
            output_part2=p2,
        )
        assert len(tasks) == 9600
        assert p1.exists()
        assert p2.exists()
        assert len(p1.read_text().splitlines()) == 5000
        assert len(p2.read_text().splitlines()) == 4600

    def test_custom_parameters_grid(self, tmp_path):
        from scripts.generate_lower_quantile_sweep_tasks import generate_tasks

        out_file = tmp_path / "custom_tasks.txt"
        tasks = generate_tasks(
            dimensions=[2, 5],
            n_trains=[112],
            functions=["sphere"],
            strategies=["natural"],
            seeds=[0, 1],
            surrogates=["smac_default", "breiman"],
            output_file=out_file,
        )
        # 2 dims x 1 N x 1 func x 1 strat x 2 seeds x 2 surrogates = 8 tasks
        assert len(tasks) == 8


# ---------------------------------------------------------------------------
# Test 2: Local Runner
# ---------------------------------------------------------------------------

class TestRunLowerQuantileSweepLocal:
    """Tests for scripts/run_lower_quantile_sweep_local.py."""

    SCRIPT_PATH = REPO_ROOT / "scripts" / "run_lower_quantile_sweep_local.py"

    def test_script_exists_and_executable(self):
        assert self.SCRIPT_PATH.exists(), f"{self.SCRIPT_PATH} does not exist"
        mode = self.SCRIPT_PATH.stat().st_mode
        assert bool(mode & stat.S_IXUSR), f"{self.SCRIPT_PATH} is not executable (chmod +x required)"

    def test_cli_parser_options(self):
        from scripts.run_lower_quantile_sweep_local import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--task-file", "results/test_lower_tasks.txt",
            "--max-workers", "4",
            "--dry-run",
            "--limit", "10",
        ])
        assert args.task_file == "results/test_lower_tasks.txt"
        assert args.max_workers == 4
        assert args.dry_run is True
        assert args.limit == 10

    def test_dry_run_execution(self, tmp_path):
        from scripts.run_lower_quantile_sweep_local import run_local_sweep

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

    def test_concurrent_execution_process_pool(self, tmp_path):
        from scripts.run_lower_quantile_sweep_local import run_local_sweep

        task_file = tmp_path / "mock_tasks.txt"
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

    def test_thread_limits_environment_variables(self, tmp_path):
        from scripts.run_lower_quantile_sweep_local import run_local_sweep

        task_file = tmp_path / "env_tasks.txt"
        # Verify that thread pinning env vars are active in worker
        code = (
            "import os, sys; "
            "assert os.environ.get('OPENBLAS_NUM_THREADS') == '1'; "
            "assert os.environ.get('MKL_NUM_THREADS') == '1'; "
            "assert os.environ.get('OMP_NUM_THREADS') == '1'; "
            "sys.exit(0)"
        )
        task_file.write_text(f"{sys.executable} -c \"{code}\"\n")

        res = run_local_sweep(task_file=task_file, max_workers=1, dry_run=False)
        assert res["succeeded_tasks"] == 1
        assert res["failed_tasks"] == 0

    def test_handles_failing_task(self, tmp_path):
        from scripts.run_lower_quantile_sweep_local import run_local_sweep

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
# Test 3: Aggregator and Lower-Quantile Scorecard
# ---------------------------------------------------------------------------

class TestAggregateLowerQuantileResults:
    """Tests for scripts/aggregate_lower_quantile_results.py."""

    SCRIPT_PATH = REPO_ROOT / "scripts" / "aggregate_lower_quantile_results.py"

    def test_script_exists_and_executable(self):
        assert self.SCRIPT_PATH.exists(), f"{self.SCRIPT_PATH} does not exist"
        mode = self.SCRIPT_PATH.stat().st_mode
        assert bool(mode & stat.S_IXUSR), f"{self.SCRIPT_PATH} is not executable (chmod +x required)"

    @pytest.fixture
    def mock_results_dir(self, tmp_path):
        """Generate synthetic raw parquet files and summary JSONs with lower quantile fields."""
        raw_dir = tmp_path / "raw"
        summaries_dir = tmp_path / "summaries"
        raw_dir.mkdir(parents=True)
        summaries_dir.mkdir(parents=True)

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

            u_prox_a_lower = d_norm + 0.05 * np.random.randn(M)
            u_prox_b_lower = d_norm + 0.06 * np.random.randn(M)
            u_prox_ac_lower = u_prox_a_lower * 1.2
            u_prox_bc_lower = u_prox_b_lower * 1.2
            u_plcb_lower = u_prox_a_lower + 0.1
            u_hutter_total = d_norm * 0.4 + 0.2 * np.random.randn(M)
            u_slcb = u_hutter_total
            u_rf_fire_lower = d_norm * 0.5 + 0.15 * np.random.randn(M)
            u_shaker_epistemic = d_norm * 0.3 + 0.3 * np.random.randn(M)

            u_hutter_between = d_norm * 0.35 + 0.1 * np.random.randn(M)
            u_hutter_within = d_norm * 0.1 + 0.05 * np.random.randn(M)
            u_shaker_total = d_norm * 0.38 + 0.2 * np.random.randn(M)
            shaker_mi = d_norm * 0.25 + 0.1 * np.random.randn(M)
            shaker_total_entropy = d_norm * 0.4 + 0.1 * np.random.randn(M)

            df = pd.DataFrame({
                "point_id": np.arange(M),
                "d_norm": d_norm,
                "d_inf": d_inf,
                "d_rel": d_norm / np.sqrt(dim),
                "stratum": (d_norm * 4).astype(int).clip(0, 3),
                "y_true": np.zeros(M),
                "y_hat": np.zeros(M),
                "abs_error": np.abs(np.random.randn(M)),
                "u_prox_a_lower": u_prox_a_lower,
                "u_prox_b_lower": u_prox_b_lower,
                "u_prox_ac_lower": u_prox_ac_lower,
                "u_prox_bc_lower": u_prox_bc_lower,
                "u_plcb_lower": u_plcb_lower,
                "u_plcb": u_plcb_lower,
                "u_hutter_total": u_hutter_total,
                "u_hutter_between": u_hutter_between,
                "u_hutter_within": u_hutter_within,
                "u_slcb": u_slcb,
                "u_rf_fire_lower": u_rf_fire_lower,
                "u_shaker_epistemic": u_shaker_epistemic,
                "u_shaker_total": u_shaker_total,
                "shaker_mi": shaker_mi,
                "shaker_total_entropy": shaker_total_entropy,
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
                    "u_prox_a_lower": {"spearman_dist": 0.85, "spearman_err": 0.1},
                    "u_prox_b_lower": {"spearman_dist": 0.84, "spearman_err": 0.1},
                    "u_prox_ac_lower": {"spearman_dist": 0.88, "spearman_err": 0.1},
                    "u_prox_bc_lower": {"spearman_dist": 0.87, "spearman_err": 0.1},
                    "u_plcb_lower": {"spearman_dist": 0.86, "spearman_err": 0.1},
                    "u_slcb": {"spearman_dist": 0.45, "spearman_err": 0.1},
                    "u_hutter_total": {"spearman_dist": 0.45, "spearman_err": 0.1},
                    "u_hutter_between": {"spearman_dist": 0.40, "spearman_err": 0.1},
                    "u_hutter_within": {"spearman_dist": 0.20, "spearman_err": 0.1},
                    "u_rf_fire_lower": {"spearman_dist": 0.50, "spearman_err": 0.1},
                    "u_shaker_epistemic": {"spearman_dist": 0.35, "spearman_err": 0.1},
                    "u_shaker_total": {"spearman_dist": 0.38, "spearman_err": 0.1},
                    "shaker_mi": {"spearman_dist": 0.25, "spearman_err": 0.1},
                    "shaker_total_entropy": {"spearman_dist": 0.30, "spearman_err": 0.1},
                },
            }
            summary_name = f"summary_{func}_d{dim}_n{n_train}_{strat}_smac_default_s{seed}.json"
            with open(summaries_dir / summary_name, "w", encoding="utf-8") as f:
                json.dump(summary, f)

        return tmp_path

    def test_progress_logging_output_10_percent_steps(self, mock_results_dir, capsys):
        from scripts.aggregate_lower_quantile_results import generate_lower_quantile_scorecard

        output_dir = mock_results_dir / "analysis_progress"
        generate_lower_quantile_scorecard(
            raw_dir=mock_results_dir / "raw",
            output_dir=output_dir,
        )
        captured = capsys.readouterr()
        assert "[Aggregation]" in captured.out
        assert "%" in captured.out

    def test_aggregate_scorecard_generation_md_and_csv(self, mock_results_dir):
        from scripts.aggregate_lower_quantile_results import generate_lower_quantile_scorecard

        output_dir = mock_results_dir / "analysis"
        generate_lower_quantile_scorecard(
            raw_dir=mock_results_dir / "raw",
            summary_dir=mock_results_dir / "summaries",
            output_dir=output_dir,
        )

        md_path = output_dir / "table_lower_quantile_scorecard.md"
        csv_path = output_dir / "table_lower_quantile_scorecard.csv"

        assert md_path.exists(), f"Expected {md_path} to exist"
        assert csv_path.exists(), f"Expected {csv_path} to exist"

        df = pd.read_csv(csv_path)
        required_cols = [
            "dimension_group",
            "comparison",
            "estimator_test",
            "estimator_ref",
            "spearman_norm_test",
            "spearman_norm_ref",
            "delta_rho_norm",
            "wilcoxon_stat",
            "wilcoxon_pvalue",
            "cliffs_delta",
            "win_tie_loss",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column '{col}' in scorecard CSV"

        # Check comparisons include Proximity A, B, AC, BC, PLCB and non-proximity baselines
        comparisons = list(df["comparison"].unique())

        assert any("prox_a" in c for c in comparisons), "Scorecard must evaluate Proximity A"
        assert any("prox_b" in c for c in comparisons), "Scorecard must evaluate Proximity B"
        assert any("prox_ac" in c for c in comparisons), "Scorecard must evaluate Proximity AC"
        assert any("prox_bc" in c for c in comparisons), "Scorecard must evaluate Proximity BC"
        assert any("plcb" in c for c in comparisons), "Scorecard must evaluate PLCB"
        assert any("slcb" in c or "hutter" in c for c in comparisons), "Scorecard must compare vs SLCB/Hutter"

        # Stratifications: Low-D, High-D, All Dimensions
        dim_groups = set(df["dimension_group"].unique())
        assert any("Low-D" in g for g in dim_groups), "Scorecard must stratify Low-D (D <= 5)"
        assert any("High-D" in g for g in dim_groups), "Scorecard must stratify High-D (D >= 16)"
        assert any("All" in g for g in dim_groups), "Scorecard must include All Dimensions"

        # Markdown format checks
        md_text = md_path.read_text(encoding="utf-8")
        assert "table_lower_quantile_scorecard" in md_text or "Lower-Quantile" in md_text
        assert "prox_a" in md_text
        assert "prox_b" in md_text
        assert "prox_ac" in md_text
        assert "prox_bc" in md_text
        assert "plcb" in md_text

    def test_extracts_all_non_proximity_baselines(self, mock_results_dir):
        from scripts.aggregate_lower_quantile_results import collect_experiment_records

        df_records = collect_experiment_records(raw_dir=mock_results_dir / "raw")
        assert not df_records.empty

        # Check all non-proximity baselines are recorded
        for base in ["slcb", "hutter_between", "hutter_within", "shaker", "shaker_total", "shaker_mi", "shaker_total_entropy", "rf_fire"]:
            assert f"{base}_norm" in df_records.columns, f"Expected {base}_norm in extracted records"

    def test_generates_leaderboard_table_md_and_csv(self, mock_results_dir):
        from scripts.aggregate_lower_quantile_results import generate_lower_quantile_scorecard

        output_dir = mock_results_dir / "analysis_leaderboard"
        generate_lower_quantile_scorecard(
            raw_dir=mock_results_dir / "raw",
            output_dir=output_dir,
        )

        lb_md = output_dir / "table_lower_quantile_leaderboard.md"
        lb_csv = output_dir / "table_lower_quantile_leaderboard.csv"

        assert lb_md.exists(), f"Expected {lb_md} to exist"
        assert lb_csv.exists(), f"Expected {lb_csv} to exist"

        df_lb = pd.read_csv(lb_csv)
        assert "rank" in df_lb.columns
        assert "estimator" in df_lb.columns
        assert "spearman_norm" in df_lb.columns
        assert "norm_rank_monotonicity" in df_lb.columns
        assert "dimension_group" in df_lb.columns

        # Verify Proximity variants and baselines are ranked
        estimators = list(df_lb["estimator"].unique())
        assert "prox_a" in estimators
        assert "prox_b" in estimators
        assert "prox_ac" in estimators
        assert "prox_bc" in estimators
        assert "plcb" in estimators

    def test_use_summaries_flag(self, mock_results_dir):
        from scripts.aggregate_lower_quantile_results import generate_lower_quantile_scorecard

        output_dir = mock_results_dir / "analysis_summaries"
        scorecard_df, md = generate_lower_quantile_scorecard(
            summary_dir=mock_results_dir / "summaries",
            prefer_summaries=True,
            output_dir=output_dir,
        )
        assert not scorecard_df.empty
        assert len(scorecard_df) > 0

    def test_surrogate_breakdown_scorecard(self, mock_results_dir):
        from scripts.aggregate_lower_quantile_results import generate_lower_quantile_scorecard

        output_dir = mock_results_dir / "analysis_surrogate"
        generate_lower_quantile_scorecard(
            raw_dir=mock_results_dir / "raw",
            output_dir=output_dir,
        )

        surr_md = output_dir / "table_lower_quantile_by_surrogate.md"
        surr_csv = output_dir / "table_lower_quantile_by_surrogate.csv"

        assert surr_md.exists(), f"Expected {surr_md} to exist"
        assert surr_csv.exists(), f"Expected {surr_csv} to exist"

        df_surr = pd.read_csv(surr_csv)
        assert "surrogate" in df_surr.columns
        assert "estimator" in df_surr.columns
        assert "spearman_norm" in df_surr.columns


# ---------------------------------------------------------------------------
# Test 4: SLURM Dispatchers
# ---------------------------------------------------------------------------

class TestSlurmDispatchers:
    """Tests for SLURM submission scripts for lower quantile sweep."""

    ARRAY_SBATCH = REPO_ROOT / "scripts" / "submit_lower_quantile_sweep_array.sbatch"
    RUN_SH = REPO_ROOT / "scripts" / "submit_lower_quantile_sweep.sh"
    PART1_SH = REPO_ROOT / "scripts" / "submit_lower_quantile_sweep_part1.sh"
    PART2_SH = REPO_ROOT / "scripts" / "submit_lower_quantile_sweep_part2.sh"

    def test_array_sbatch_exists_and_executable(self):
        assert self.ARRAY_SBATCH.exists(), f"{self.ARRAY_SBATCH} does not exist"
        mode = self.ARRAY_SBATCH.stat().st_mode
        assert bool(mode & stat.S_IXUSR), f"{self.ARRAY_SBATCH} is not executable (chmod +x required)"

    def test_array_sbatch_header_and_partition_ai(self):
        content = self.ARRAY_SBATCH.read_text(encoding="utf-8")
        assert "#SBATCH -p ai" in content or "#SBATCH --partition=ai" in content
        assert "results/lower_quantile_sweep" in content

    def test_submit_sh_exists_executable_and_chunks_under_200(self):
        assert self.RUN_SH.exists(), f"{self.RUN_SH} does not exist"
        mode = self.RUN_SH.stat().st_mode
        assert bool(mode & stat.S_IXUSR), f"{self.RUN_SH} is not executable (chmod +x required)"

        content = self.RUN_SH.read_text(encoding="utf-8")
        assert "scripts/generate_lower_quantile_sweep_tasks.py" in content
        assert "submit_lower_quantile_sweep_array.sbatch" in content

        # Check chunk size is <= 200
        match = re.search(r"CHUNK_SIZE=(\d+)", content)
        assert match is not None, "CHUNK_SIZE must be defined in submit_lower_quantile_sweep.sh"
        chunk_size = int(match.group(1))
        assert chunk_size <= 200, f"CHUNK_SIZE must be <= 200, got {chunk_size}"

    def test_submit_part1_and_part2_scripts_exist_executable_and_configured(self):
        assert self.PART1_SH.exists(), "part1 script should exist"
        assert self.PART2_SH.exists(), "part2 script should exist"

        assert bool(self.PART1_SH.stat().st_mode & stat.S_IXUSR), "part1 script must be executable"
        assert bool(self.PART2_SH.stat().st_mode & stat.S_IXUSR), "part2 script must be executable"

        p1_content = self.PART1_SH.read_text(encoding="utf-8")
        assert "START_TASK=1" in p1_content
        assert "END_TASK=5000" in p1_content

        p2_content = self.PART2_SH.read_text(encoding="utf-8")
        assert "START_TASK=5001" in p2_content
        assert "END_TASK=9600" in p2_content
