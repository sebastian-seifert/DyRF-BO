"""Unit tests for extrapolation CLI runner and sweep task generator."""

import json
import os
import shlex
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Target modules for Milestone 5
import numpy as np

from scripts.generate_extrapolation_sweep_tasks import (
    build_parser as build_task_generator_parser,
    generate_tasks,
    main as main_generate_tasks,
)
from scripts.run_extrapolation_experiment import (
    _json_serializable,
    build_parser as build_run_experiment_parser,
    main as main_run_experiment,
    run_experiment,
)


class TestRunExperimentParser:
    """Tests for run_extrapolation_experiment argument parser."""

    def test_parser_defaults(self):
        parser = build_run_experiment_parser()
        args = parser.parse_args([
            "--dimension", "2",
            "--n-train", "112",
            "--function", "sphere",
            "--strategy", "natural",
            "--seed", "42",
        ])
        assert args.dimension == 2
        assert args.n_train == 112
        assert args.function == "sphere"
        assert args.strategy == "natural"
        assert args.seed == 42
        assert args.n_test == 10000
        assert args.k == 28
        assert pytest.approx(args.eps) == 0.080791
        assert pytest.approx(args.decay_lambda) == 0.20486
        assert args.n_trees == 10
        assert args.output_dir == "results/extrapolation_uq/raw"
        assert args.summary_dir == "results/extrapolation_uq/summaries"
        assert args.save_parquet is True
        assert args.skip_if_exists is False

    def test_parser_custom_args(self):
        parser = build_run_experiment_parser()
        args = parser.parse_args([
            "--dimension", "16",
            "--n-train", "896",
            "--function", "rastrigin",
            "--strategy", "stratified",
            "--seed", "7",
            "--n-test", "5000",
            "--k", "32",
            "--eps", "0.05",
            "--decay-lambda", "0.15",
            "--n-trees", "20",
            "--output-dir", "custom/raw",
            "--summary-dir", "custom/summaries",
            "--no-parquet",
            "--skip-if-exists",
        ])
        assert args.dimension == 16
        assert args.n_train == 896
        assert args.function == "rastrigin"
        assert args.strategy == "stratified"
        assert args.seed == 7
        assert args.n_test == 5000
        assert args.k == 32
        assert pytest.approx(args.eps) == 0.05
        assert pytest.approx(args.decay_lambda) == 0.15
        assert args.n_trees == 20
        assert args.output_dir == "custom/raw"
        assert args.summary_dir == "custom/summaries"
        assert args.save_parquet is False
        assert args.skip_if_exists is True

    def test_parser_missing_required_args(self):
        parser = build_run_experiment_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--dimension", "2"])

    def test_parser_invalid_strategy(self):
        parser = build_run_experiment_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--dimension", "2",
                "--n-train", "112",
                "--function", "sphere",
                "--strategy", "unknown_strategy",
                "--seed", "0",
            ])

    def test_parser_function_case_normalization_and_choices(self):
        parser = build_run_experiment_parser()
        args = parser.parse_args([
            "--dimension", "2",
            "--n-train", "112",
            "--function", "SPHERE",
            "--strategy", "natural",
            "--seed", "0",
        ])
        assert args.function == "sphere"

    def test_parser_invalid_function(self):
        parser = build_run_experiment_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--dimension", "2",
                "--n-train", "112",
                "--function", "invalid_func",
                "--strategy", "natural",
                "--seed", "0",
            ])


class TestRunExperimentExecution:
    """Tests for run_extrapolation_experiment execution logic."""

    def test_small_pilot_execution(self, tmp_path):
        raw_dir = tmp_path / "raw"
        sum_dir = tmp_path / "summaries"

        argv = [
            "--dimension", "2",
            "--n-train", "28",
            "--function", "sphere",
            "--strategy", "stratified",
            "--seed", "42",
            "--n-test", "100",
            "--k", "28",
            "--n-trees", "5",
            "--output-dir", str(raw_dir),
            "--summary-dir", str(sum_dir),
        ]
        ret = main_run_experiment(argv)
        assert ret == 0

        # Check parquet file
        expected_parquet = raw_dir / "extrapolation_sphere_d2_n28_stratified_s42.parquet"
        assert expected_parquet.exists()

        # Check summary file
        expected_summary = sum_dir / "summary_sphere_d2_n28_stratified_s42.json"
        assert expected_summary.exists()

        with open(expected_summary, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        assert summary_data["dimension"] == 2
        assert summary_data["n_train"] == 28
        assert summary_data["function_name"] == "sphere"
        assert summary_data["sampling_strategy"] == "stratified"
        assert summary_data["seed"] == 42
        assert summary_data["n_test"] == 100
        assert "slcb_picp" in summary_data
        assert "plcb_picp" in summary_data
        assert "global" in summary_data

    def test_execution_no_parquet(self, tmp_path):
        raw_dir = tmp_path / "raw"
        sum_dir = tmp_path / "summaries"

        argv = [
            "--dimension", "2",
            "--n-train", "28",
            "--function", "sphere",
            "--strategy", "natural",
            "--seed", "1",
            "--n-test", "50",
            "--k", "28",
            "--n-trees", "5",
            "--output-dir", str(raw_dir),
            "--summary-dir", str(sum_dir),
            "--no-parquet",
        ]
        ret = main_run_experiment(argv)
        assert ret == 0

        # Parquet should NOT exist
        expected_parquet = raw_dir / "extrapolation_sphere_d2_n28_natural_s1.parquet"
        assert not expected_parquet.exists()

        # Summary JSON should exist
        expected_summary = sum_dir / "summary_sphere_d2_n28_natural_s1.json"
        assert expected_summary.exists()

    def test_skip_if_exists_both_files_present(self, tmp_path):
        raw_dir = tmp_path / "raw"
        sum_dir = tmp_path / "summaries"
        raw_dir.mkdir(parents=True)
        sum_dir.mkdir(parents=True)

        parquet_file = raw_dir / "extrapolation_sphere_d2_n112_natural_s0.parquet"
        summary_file = sum_dir / "summary_sphere_d2_n112_natural_s0.json"
        parquet_file.write_text("dummy parquet content")
        summary_file.write_text(json.dumps({"dummy": True}))

        with patch("scripts.run_extrapolation_experiment.run_single_experiment") as mock_run:
            argv = [
                "--dimension", "2",
                "--n-train", "112",
                "--function", "sphere",
                "--strategy", "natural",
                "--seed", "0",
                "--output-dir", str(raw_dir),
                "--summary-dir", str(sum_dir),
                "--skip-if-exists",
            ]
            ret = main_run_experiment(argv)
            assert ret == 0
            mock_run.assert_not_called()

    def test_no_skip_if_parquet_missing(self, tmp_path):
        raw_dir = tmp_path / "raw"
        sum_dir = tmp_path / "summaries"
        sum_dir.mkdir(parents=True)

        # Only summary exists, parquet missing
        summary_file = sum_dir / "summary_sphere_d2_n112_natural_s0.json"
        summary_file.write_text(json.dumps({"dummy": True}))

        fake_summary = {"dimension": 2, "n_train": 112, "function_name": "sphere", "seed": 0}
        with patch(
            "scripts.run_extrapolation_experiment.run_single_experiment",
            return_value=(fake_summary, MagicMock()),
        ) as mock_run:
            argv = [
                "--dimension", "2",
                "--n-train", "112",
                "--function", "sphere",
                "--strategy", "natural",
                "--seed", "0",
                "--output-dir", str(raw_dir),
                "--summary-dir", str(sum_dir),
                "--skip-if-exists",
            ]
            ret = main_run_experiment(argv)
            assert ret == 0
            mock_run.assert_called_once()

    def test_no_skip_if_flag_not_set(self, tmp_path):
        raw_dir = tmp_path / "raw"
        sum_dir = tmp_path / "summaries"
        raw_dir.mkdir(parents=True)
        sum_dir.mkdir(parents=True)

        parquet_file = raw_dir / "extrapolation_sphere_d2_n112_natural_s0.parquet"
        summary_file = sum_dir / "summary_sphere_d2_n112_natural_s0.json"
        parquet_file.write_text("dummy")
        summary_file.write_text(json.dumps({"dummy": True}))

        fake_summary = {"dimension": 2, "n_train": 112, "function_name": "sphere", "seed": 0}
        with patch(
            "scripts.run_extrapolation_experiment.run_single_experiment",
            return_value=(fake_summary, MagicMock()),
        ) as mock_run:
            argv = [
                "--dimension", "2",
                "--n-train", "112",
                "--function", "sphere",
                "--strategy", "natural",
                "--seed", "0",
                "--output-dir", str(raw_dir),
                "--summary-dir", str(sum_dir),
            ]
            ret = main_run_experiment(argv)
            assert ret == 0
            mock_run.assert_called_once()

    def test_skip_if_exists_zero_byte_file_does_not_skip(self, tmp_path):
        raw_dir = tmp_path / "raw"
        sum_dir = tmp_path / "summaries"
        raw_dir.mkdir(parents=True)
        sum_dir.mkdir(parents=True)

        # 0-byte parquet file and 0-byte summary file
        parquet_file = raw_dir / "extrapolation_sphere_d2_n112_natural_s0.parquet"
        summary_file = sum_dir / "summary_sphere_d2_n112_natural_s0.json"
        parquet_file.touch()
        summary_file.touch()
        assert parquet_file.stat().st_size == 0
        assert summary_file.stat().st_size == 0

        fake_summary = {"dimension": 2, "n_train": 112, "function_name": "sphere", "seed": 0}
        with patch(
            "scripts.run_extrapolation_experiment.run_single_experiment",
            return_value=(fake_summary, MagicMock()),
        ) as mock_run:
            argv = [
                "--dimension", "2",
                "--n-train", "112",
                "--function", "sphere",
                "--strategy", "natural",
                "--seed", "0",
                "--output-dir", str(raw_dir),
                "--summary-dir", str(sum_dir),
                "--skip-if-exists",
            ]
            ret = main_run_experiment(argv)
            assert ret == 0
            mock_run.assert_called_once()

    def test_json_serializable_nan_and_inf_conversion(self):
        data = {
            "a": float("nan"),
            "b": np.float64(np.nan),
            "c": [1.0, np.nan, 2.0],
            "d": {"nested": np.nan},
            "pos_inf": float("inf"),
            "neg_inf": float("-inf"),
            "np_pos_inf": np.float64(np.inf),
            "np_neg_inf": np.float64(-np.inf),
        }
        res = _json_serializable(data)
        assert res["a"] is None
        assert res["b"] is None
        assert res["c"] == [1.0, None, 2.0]
        assert res["d"]["nested"] is None
        assert res["pos_inf"] is None
        assert res["neg_inf"] is None
        assert res["np_pos_inf"] is None
        assert res["np_neg_inf"] is None
        # Must be valid json without standard NaN or Infinity tokens
        encoded = json.dumps(res)
        assert "NaN" not in encoded
        assert "Infinity" not in encoded
        assert "null" in encoded


class TestTaskGenerator:
    """Tests for generate_extrapolation_sweep_tasks script."""

    def test_parser_defaults(self):
        parser = build_task_generator_parser()
        args = parser.parse_args([])
        assert args.output_file == "results/extrapolation_sweep_tasks.txt"
        assert args.pilot is False
        assert args.dimensions == [2, 3, 5, 8, 16, 32]
        assert args.n_trains == [112, 224, 448, 896]
        assert args.functions == ["sphere", "rosenbrock", "rastrigin", "ackley"]
        assert args.strategies == ["natural", "stratified"]
        assert args.seeds == list(range(10))
        assert args.python_bin == "python"
        assert args.skip_if_exists is True

    def test_pilot_generates_exactly_4_tasks(self, tmp_path):
        out_file = tmp_path / "pilot_tasks.txt"
        tasks = generate_tasks(output_file=out_file, pilot=True)
        assert len(tasks) == 4
        assert out_file.exists()

        lines = [line.strip() for line in out_file.read_text().splitlines() if line.strip()]
        assert len(lines) == 4

        # Verify pilot grid: D in [2, 16], N=112, sphere, seed=0, natural & stratified
        expected_combinations = {
            (2, 112, "sphere", "natural", 0),
            (2, 112, "sphere", "stratified", 0),
            (16, 112, "sphere", "natural", 0),
            (16, 112, "sphere", "stratified", 0),
        }

        parser = build_run_experiment_parser()
        actual_combinations = set()
        for cmd in tasks:
            tokens = shlex.split(cmd)
            # Find arguments for run_extrapolation_experiment
            # Tokens should begin with python scripts/run_extrapolation_experiment.py
            assert "run_extrapolation_experiment.py" in tokens[1]
            cli_args = parser.parse_args(tokens[2:])
            actual_combinations.add((
                cli_args.dimension,
                cli_args.n_train,
                cli_args.function,
                cli_args.strategy,
                cli_args.seed,
            ))
            assert cli_args.skip_if_exists is True

        assert actual_combinations == expected_combinations

    def test_full_grid_generates_exactly_1920_tasks(self, tmp_path):
        out_file = tmp_path / "tasks.txt"
        tasks = generate_tasks(output_file=out_file, pilot=False)
        assert len(tasks) == 1920
        assert out_file.exists()

        lines = [line.strip() for line in out_file.read_text().splitlines() if line.strip()]
        assert len(lines) == 1920

        # Check unique tasks
        assert len(set(tasks)) == 1920

        # Sample a few commands and check parseability
        parser = build_run_experiment_parser()
        for cmd in [tasks[0], tasks[100], tasks[500], tasks[1000], tasks[-1]]:
            tokens = shlex.split(cmd)
            cli_args = parser.parse_args(tokens[2:])
            assert cli_args.dimension in [2, 3, 5, 8, 16, 32]
            assert cli_args.n_train in [112, 224, 448, 896]
            assert cli_args.function in ["sphere", "rosenbrock", "rastrigin", "ackley"]
            assert cli_args.strategy in ["natural", "stratified"]
            assert cli_args.seed in list(range(10))
            assert cli_args.skip_if_exists is True

    def test_custom_subsets_and_skip_flag(self, tmp_path):
        out_file = tmp_path / "custom_tasks.txt"
        tasks = generate_tasks(
            output_file=out_file,
            dimensions=[3, 5],
            n_trains=[224],
            functions=["ackley"],
            strategies=["natural"],
            seeds=[1, 2, 3],
            python_bin=".venv/bin/python",
            skip_if_exists=False,
        )
        assert len(tasks) == 2 * 1 * 1 * 1 * 3
        for cmd in tasks:
            assert cmd.startswith(".venv/bin/python scripts/run_extrapolation_experiment.py")
            assert "--skip-if-exists" not in cmd

    def test_main_cli_execution(self, tmp_path):
        out_file = tmp_path / "cli_tasks.txt"
        argv = [
            "--output-file", str(out_file),
            "--pilot",
            "--python-bin", "python3",
        ]
        ret = main_generate_tasks(argv)
        assert ret == 0
        assert out_file.exists()
        lines = [line.strip() for line in out_file.read_text().splitlines() if line.strip()]
        assert len(lines) == 4
        assert lines[0].startswith("python3 scripts/run_extrapolation_experiment.py")

    def test_generator_quotes_python_bin_with_spaces(self, tmp_path):
        out_file = tmp_path / "quoted_tasks.txt"
        tasks = generate_tasks(
            output_file=out_file,
            pilot=True,
            python_bin="/path with spaces/bin/python",
        )
        assert len(tasks) == 4
        # First token when split by shlex must match the unquoted path
        tokens = shlex.split(tasks[0])
        assert tokens[0] == "/path with spaces/bin/python"
        assert tasks[0].startswith("'/path with spaces/bin/python' scripts/run_extrapolation_experiment.py")
