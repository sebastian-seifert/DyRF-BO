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
        assert args.surrogate == "smac_default"
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
            "--surrogate", "mature",
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
        assert args.surrogate == "mature"
        assert args.output_dir == "custom/raw"
        assert args.summary_dir == "custom/summaries"
        assert args.save_parquet is False
        assert args.skip_if_exists is True

    def test_parser_surrogate_choices_and_aliases(self):
        parser = build_run_experiment_parser()
        base_cmd = [
            "--dimension", "2",
            "--n-train", "112",
            "--function", "sphere",
            "--strategy", "natural",
            "--seed", "0",
        ]
        for surr in ["smac_default", "mature", "shallow", "coarse", "breiman"]:
            # Test --surrogate
            args1 = parser.parse_args(base_cmd + ["--surrogate", surr])
            assert args1.surrogate == surr
            # Test --surrogate-type alias
            args2 = parser.parse_args(base_cmd + ["--surrogate-type", surr])
            assert args2.surrogate == surr

        with pytest.raises(SystemExit):
            parser.parse_args(base_cmd + ["--surrogate", "invalid_surrogate"])

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

        with open(sum_dir / "summary_sphere_d2_n28_stratified_smac_default_s42.json", "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        assert summary_data["dimension"] == 2
        assert summary_data["n_train"] == 28
        assert summary_data["function_name"] == "sphere"
        assert summary_data["sampling_strategy"] == "stratified"
        assert summary_data["seed"] == 42
        assert summary_data["surrogate_type"] == "smac_default"
        assert summary_data["n_test"] == 100
        assert "slcb_picp" in summary_data
        assert "plcb_picp" in summary_data
        assert "global" in summary_data

        # Check parquet file
        expected_parquet = raw_dir / "extrapolation_sphere_d2_n28_stratified_smac_default_s42.parquet"
        assert expected_parquet.exists()

        # Check summary file
        expected_summary = sum_dir / "summary_sphere_d2_n28_stratified_smac_default_s42.json"
        assert expected_summary.exists()

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
        expected_parquet = raw_dir / "extrapolation_sphere_d2_n28_natural_smac_default_s1.parquet"
        assert not expected_parquet.exists()

        # Summary JSON should exist
        expected_summary = sum_dir / "summary_sphere_d2_n28_natural_smac_default_s1.json"
        assert expected_summary.exists()

    def test_skip_if_exists_both_files_present(self, tmp_path):
        raw_dir = tmp_path / "raw"
        sum_dir = tmp_path / "summaries"
        raw_dir.mkdir(parents=True)
        sum_dir.mkdir(parents=True)

        parquet_file = raw_dir / "extrapolation_sphere_d2_n112_natural_smac_default_s0.parquet"
        summary_file = sum_dir / "summary_sphere_d2_n112_natural_smac_default_s0.json"
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
        summary_file = sum_dir / "summary_sphere_d2_n112_natural_smac_default_s0.json"
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

        parquet_file = raw_dir / "extrapolation_sphere_d2_n112_natural_smac_default_s0.parquet"
        summary_file = sum_dir / "summary_sphere_d2_n112_natural_smac_default_s0.json"
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
        parquet_file = raw_dir / "extrapolation_sphere_d2_n112_natural_smac_default_s0.parquet"
        summary_file = sum_dir / "summary_sphere_d2_n112_natural_smac_default_s0.json"
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

    def test_all_15_candidate_estimators_in_parquet_and_summary_json(self, tmp_path):
        import pandas as pd
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

        parquet_path = raw_dir / "extrapolation_sphere_d2_n28_stratified_smac_default_s42.parquet"
        summary_path = sum_dir / "summary_sphere_d2_n28_stratified_smac_default_s42.json"
        assert parquet_path.exists() and parquet_path.stat().st_size > 0
        assert summary_path.exists() and summary_path.stat().st_size > 0

        # Lossless Parquet roundtrip
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        assert len(df) == 100
        assert "surrogate" in df.columns
        assert (df["surrogate"] == "smac_default").all()

        expected_estimators = [
            "u_hutter_total",
            "u_hutter_between",
            "u_hutter_within",
            "u_shaker_epistemic",
            "u_shaker_total",
            "u_rf_fire_half",
            "u_rf_fire_lower",
            "u_prox_a_half",
            "u_prox_a_lower",
            "u_prox_b_half",
            "u_prox_b_lower",
            "u_prox_bc_half",
            "u_prox_bc_lower",
            "u_plcb_half",
            "u_plcb_lower",
        ]

        for col in expected_estimators:
            assert col in df.columns, f"Column '{col}' missing in Parquet."
            assert not df[col].isna().any(), f"Column '{col}' contains NaNs in Parquet."
            assert (df[col] >= 0.0).all(), f"Estimator '{col}' contains negative values in Parquet."

        # Verify summary JSON validity and structure
        raw_json_str = summary_path.read_text(encoding="utf-8")
        assert "NaN" not in raw_json_str
        assert "Infinity" not in raw_json_str

        summary_data = json.loads(raw_json_str)
        assert "global" in summary_data
        assert "strata" in summary_data

        expected_metrics = ["picp", "mpiw", "winkler", "spearman_dist", "spearman_err", "auroc", "auprc"]
        for est in expected_estimators:
            assert est in summary_data["global"], f"Estimator '{est}' missing from global summary."
            for m in expected_metrics:
                assert m in summary_data["global"][est], f"Metric '{m}' missing for '{est}' in global summary."

        # Verify strata 0, 1, 2, 3 in summary
        for s_key in ["0", "1", "2", "3"]:
            assert s_key in summary_data["strata"], f"Stratum '{s_key}' missing from summary strata."
            for est in expected_estimators:
                assert est in summary_data["strata"][s_key], f"Estimator '{est}' missing from stratum {s_key}."

    def test_custom_surrogate_cli_execution(self, tmp_path):
        raw_dir = tmp_path / "raw"
        sum_dir = tmp_path / "summaries"

        argv = [
            "--dimension", "2",
            "--n-train", "28",
            "--function", "sphere",
            "--strategy", "stratified",
            "--seed", "42",
            "--n-test", "60",
            "--k", "28",
            "--n-trees", "5",
            "--surrogate", "breiman",
            "--output-dir", str(raw_dir),
            "--summary-dir", str(sum_dir),
        ]
        ret = main_run_experiment(argv)
        assert ret == 0

        expected_parquet = raw_dir / "extrapolation_sphere_d2_n28_stratified_breiman_s42.parquet"
        expected_summary = sum_dir / "summary_sphere_d2_n28_stratified_breiman_s42.json"
        assert expected_parquet.exists()
        assert expected_summary.exists()

        with open(expected_summary, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        assert summary_data["surrogate_type"] == "breiman"


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
        assert args.surrogates == ["smac_default", "mature", "shallow", "coarse", "breiman"]
        assert args.python_bin == "python"
        assert args.skip_if_exists is True

    def test_pilot_generates_multi_surrogate_tasks(self, tmp_path):
        out_file = tmp_path / "pilot_tasks.txt"
        tasks = generate_tasks(output_file=out_file, pilot=True)
        assert len(tasks) in (10, 20)
        assert out_file.exists()

        lines = [line.strip() for line in out_file.read_text().splitlines() if line.strip()]
        assert len(lines) in (10, 20)

        # Verify pilot grid: D in [2, 16], N=112, sphere, seed=0, natural & stratified, 5 surrogates
        expected_combinations = {
            (dim, 112, "sphere", strat, 0, surr)
            for dim in [2, 16]
            for strat in ["natural", "stratified"]
            for surr in ["smac_default", "mature", "shallow", "coarse", "breiman"]
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
                cli_args.surrogate,
            ))
            assert cli_args.skip_if_exists is True

        assert actual_combinations == expected_combinations

    def test_full_grid_generates_exactly_9600_tasks(self, tmp_path):
        out_file = tmp_path / "tasks.txt"
        tasks = generate_tasks(output_file=out_file, pilot=False)
        assert len(tasks) == 9600
        assert out_file.exists()

        lines = [line.strip() for line in out_file.read_text().splitlines() if line.strip()]
        assert len(lines) == 9600

        # Check unique tasks
        assert len(set(tasks)) == 9600

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
            assert cli_args.surrogate in ["smac_default", "mature", "shallow", "coarse", "breiman"]
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
            surrogates=["shallow", "breiman"],
            python_bin=".venv/bin/python",
            skip_if_exists=False,
        )
        assert len(tasks) == 2 * 1 * 1 * 1 * 3 * 2
        for cmd in tasks:
            assert cmd.startswith(".venv/bin/python scripts/run_extrapolation_experiment.py")
            assert "--surrogate" in cmd
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
        assert len(lines) in (10, 20)
        assert lines[0].startswith("python3 scripts/run_extrapolation_experiment.py")

    def test_generator_quotes_python_bin_with_spaces(self, tmp_path):
        out_file = tmp_path / "quoted_tasks.txt"
        tasks = generate_tasks(
            output_file=out_file,
            pilot=True,
            python_bin="/path with spaces/bin/python",
        )
        assert len(tasks) in (10, 20)
        # First token when split by shlex must match the unquoted path
        tokens = shlex.split(tasks[0])
        assert tokens[0] == "/path with spaces/bin/python"
        assert tasks[0].startswith("'/path with spaces/bin/python' scripts/run_extrapolation_experiment.py")

    def test_stress_generates_exactly_16_tasks(self, tmp_path):
        out_file = tmp_path / "stress_tasks.txt"
        tasks = generate_tasks(
            output_file=out_file,
            stress=True,
            output_dir="results/extrapolation_uq/stress_raw",
            summary_dir="results/extrapolation_uq/stress_summaries",
        )
        assert len(tasks) == 16
        assert out_file.exists()

        lines = [line.strip() for line in out_file.read_text().splitlines() if line.strip()]
        assert len(lines) == 16

        parser = build_run_experiment_parser()
        dims = set()
        funcs = set()
        trains = set()
        strats = set()

        for cmd in tasks:
            tokens = shlex.split(cmd)
            cli_args = parser.parse_args(tokens[2:])
            dims.add(cli_args.dimension)
            funcs.add(cli_args.function)
            trains.add(cli_args.n_train)
            strats.add(cli_args.strategy)
            assert cli_args.output_dir == "results/extrapolation_uq/stress_raw"
            assert cli_args.summary_dir == "results/extrapolation_uq/stress_summaries"
            assert cli_args.skip_if_exists is True

        assert dims == {2, 5, 16, 32}
        assert funcs == {"sphere", "rosenbrock", "rastrigin", "ackley"}
        assert trains == {112, 224}
        assert strats == {"natural", "stratified"}
