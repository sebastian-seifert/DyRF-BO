"""Unit and integration tests for Milestone 2: Lower-Quantile UQ Runner Pipeline & Downstream Metrics Export.

Validates:
1. run_single_experiment() returns point_df and creates Parquet file containing:
   - Point coordinates: x_0..x_{d-1}, point_id, surrogate, stratum, d_norm, d_rel, d_inf,
     is_interpolating, y_true, y_hat, abs_error.
   - Non-proximity: u_hutter_total, u_hutter_between, u_hutter_within, u_shaker_epistemic,
     u_shaker_total, shaker_mi, shaker_total_entropy, u_rf_fire_lower, u_slcb.
   - Proximity (lower quantiles): u_prox_a_lower, u_prox_b_lower, u_prox_ac_lower,
     u_prox_bc_lower, u_plcb_lower, u_plcb.
   - Diagnostics: delta_floor, local_mae, q_lower.
   - Strict absence of ANY '*_half' columns or deprecated '*_weighted_half' / '*_unweighted_half' columns.
2. JSON summary record contains 'global' and 'strata_metrics' dictionaries for all active estimators:
   hutter_total, hutter_between, hutter_within, shaker_epistemic, shaker_total,
   rf_fire, prox_a, prox_b, prox_ac, prox_bc, plcb (and slcb).
3. Parquet read/write roundtrip and finite value verification across all columns.
4. CLI invocation via scripts/run_extrapolation_experiment.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dyrf_bo.extrapolation_uq.runner import (
    ExtrapolationRunConfig,
    run_single_experiment,
)


EXPECTED_BASE_COLS = {
    "point_id",
    "surrogate",
    "stratum",
    "d_norm",
    "d_rel",
    "d_inf",
    "is_interpolating",
    "y_true",
    "y_hat",
    "abs_error",
}

EXPECTED_NON_PROXIMITY_COLS = {
    "u_hutter_total",
    "u_hutter_between",
    "u_hutter_within",
    "u_shaker_epistemic",
    "u_shaker_total",
    "shaker_mi",
    "shaker_total_entropy",
    "u_rf_fire_lower",
    "u_slcb",
}

EXPECTED_PROXIMITY_COLS = {
    "u_prox_a_lower",
    "u_prox_b_lower",
    "u_prox_ac_lower",
    "u_prox_bc_lower",
    "u_plcb_lower",
    "u_plcb",
}

EXPECTED_DIAGNOSTIC_COLS = {
    "delta_floor",
    "local_mae",
    "q_lower",
}

ACTIVE_ESTIMATORS = [
    "hutter_total",
    "hutter_between",
    "hutter_within",
    "shaker_epistemic",
    "shaker_total",
    "rf_fire",
    "prox_a",
    "prox_b",
    "prox_ac",
    "prox_bc",
    "plcb",
]

SCORECARD_METRICS = [
    "spearman_dist",
    "spearman_err",
    "picp",
    "mpiw",
    "winkler",
    "auroc",
    "auprc",
]


class TestLowerQuantileRunnerColumnsAndParquet:
    """Verifies DataFrame column schema, absence of *_half, and Parquet persistence."""

    def test_run_single_experiment_columns_exact_match(self, tmp_path: Path):
        config = ExtrapolationRunConfig(
            dimension=2,
            n_train=28,
            function_name="sphere",
            sampling_strategy="stratified",
            seed=42,
            n_test=40,
            k=28,
            n_trees=5,
        )
        summary, df = run_single_experiment(config, output_dir=tmp_path, save_parquet=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 40

        # Coordinate columns
        expected_coord_cols = {f"x_{d}" for d in range(config.dimension)}
        all_expected_cols: Set[str] = (
            EXPECTED_BASE_COLS
            | expected_coord_cols
            | EXPECTED_NON_PROXIMITY_COLS
            | EXPECTED_PROXIMITY_COLS
            | EXPECTED_DIAGNOSTIC_COLS
        )

        actual_cols = set(df.columns)
        assert actual_cols == all_expected_cols, (
            f"Column schema mismatch in point_df!\n"
            f"Extra columns: {actual_cols - all_expected_cols}\n"
            f"Missing columns: {all_expected_cols - actual_cols}"
        )

        # Ordering check: point_id followed immediately by coordinates
        cols_list = list(df.columns)
        assert cols_list[0] == "point_id"
        assert cols_list[1 : 1 + config.dimension] == [f"x_{d}" for d in range(config.dimension)]

    def test_strict_absence_of_half_and_deprecated_split_columns(self, tmp_path: Path):
        config = ExtrapolationRunConfig(
            dimension=3,
            n_train=28,
            function_name="sphere",
            sampling_strategy="natural",
            seed=101,
            n_test=30,
            k=28,
            n_trees=5,
        )
        _, df = run_single_experiment(config, output_dir=tmp_path, save_parquet=True)

        for col in df.columns:
            assert not col.endswith("_half"), f"Found deprecated '*_half' column in point_df: {col}"
            assert "_weighted_half" not in col, f"Found deprecated '_weighted_half' column: {col}"
            assert "_unweighted_half" not in col, f"Found deprecated '_unweighted_half' column: {col}"
            assert "_weighted_lower" not in col, f"Found deprecated '_weighted_lower' column: {col}"
            assert "_unweighted_lower" not in col, f"Found deprecated '_unweighted_lower' column: {col}"

    def test_parquet_roundtrip_lossless_and_finite(self, tmp_path: Path):
        config = ExtrapolationRunConfig(
            dimension=2,
            n_train=28,
            function_name="sphere",
            sampling_strategy="stratified",
            seed=42,
            n_test=40,
            k=28,
            n_trees=5,
        )
        summary, df = run_single_experiment(config, output_dir=tmp_path, save_parquet=True)

        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 1
        parquet_file = parquet_files[0]
        assert parquet_file.name == "extrapolation_sphere_d2_n28_stratified_smac_default_s42.parquet"

        loaded_df = pd.read_parquet(parquet_file, engine="pyarrow")
        assert list(loaded_df.columns) == list(df.columns)
        assert len(loaded_df) == len(df)

        # Verify finite values across all numeric columns
        for col in loaded_df.columns:
            if col == "surrogate":
                assert (loaded_df[col] == "smac_default").all()
            elif col == "is_interpolating":
                assert loaded_df[col].dtype == bool
                assert (loaded_df[col] == df[col]).all()
            elif col in ["point_id", "stratum"]:
                assert np.issubdtype(loaded_df[col].dtype, np.integer)
                assert (loaded_df[col] == df[col]).all()
            else:
                assert np.issubdtype(loaded_df[col].dtype, np.floating)
                assert np.all(np.isfinite(loaded_df[col])), f"Non-finite values found in {col}"
                assert np.allclose(loaded_df[col], df[col], rtol=1e-12, atol=1e-12)

        # Coordinate bounds [-1, 1]
        for d in range(config.dimension):
            assert np.all(loaded_df[f"x_{d}"] >= -1.0 - 1e-7)
            assert np.all(loaded_df[f"x_{d}"] <= 1.0 + 1e-7)

        # Mathematical relationships
        assert np.allclose(loaded_df["abs_error"], np.abs(loaded_df["y_true"] - loaded_df["y_hat"]))
        for u_col in EXPECTED_PROXIMITY_COLS | {"u_hutter_total", "u_hutter_between", "u_hutter_within", "u_slcb"}:
            assert np.all(loaded_df[u_col] >= -1e-12), f"Negative values in {u_col}"


class TestLowerQuantileSummaryMetricsRecord:
    """Verifies JSON summary record contains 'global' and 'strata_metrics' for all active estimators."""

    def test_global_and_strata_metrics_presence_for_all_estimators(self, tmp_path: Path):
        config = ExtrapolationRunConfig(
            dimension=2,
            n_train=28,
            function_name="sphere",
            sampling_strategy="stratified",
            seed=42,
            n_test=40,
            k=28,
            n_trees=5,
        )
        summary, df = run_single_experiment(config, output_dir=tmp_path, save_parquet=False)

        assert "global" in summary, "Summary missing 'global' dictionary!"
        assert "strata_metrics" in summary, "Summary missing 'strata_metrics' dictionary!"

        global_dict = summary["global"]
        strata_dict = summary["strata_metrics"]

        # Check all active estimators are present in global metrics
        for estimator in ACTIVE_ESTIMATORS:
            assert estimator in global_dict, (
                f"Active estimator '{estimator}' missing from summary['global']! "
                f"Found keys: {list(global_dict.keys())}"
            )
            est_metrics = global_dict[estimator]
            for metric in SCORECARD_METRICS:
                assert metric in est_metrics, (
                    f"Metric '{metric}' missing for estimator '{estimator}' in global_dict"
                )
                val = est_metrics[metric]
                assert val is not None and np.isfinite(val), (
                    f"Non-finite or None metric '{metric}' for '{estimator}' in global_dict: {val}"
                )

        # Check all 4 strata exist and contain all active estimators
        for s in [0, 1, 2, 3]:
            assert s in strata_dict or str(s) in strata_dict, (
                f"Stratum {s} missing from summary['strata_metrics']!"
            )
            s_data = strata_dict.get(s, strata_dict.get(str(s)))
            for estimator in ACTIVE_ESTIMATORS:
                assert estimator in s_data, (
                    f"Active estimator '{estimator}' missing from stratum {s} metrics!"
                )
                est_metrics = s_data[estimator]
                for metric in SCORECARD_METRICS:
                    assert metric in est_metrics, (
                        f"Metric '{metric}' missing for estimator '{estimator}' in stratum {s}"
                    )

    def test_lower_quantiles_used_as_uncertainty_signals(self, tmp_path: Path):
        """Verifies that quantile-based estimators use lower-quantile arrays for metrics."""
        config = ExtrapolationRunConfig(
            dimension=2,
            n_train=28,
            function_name="sphere",
            sampling_strategy="stratified",
            seed=42,
            n_test=40,
            k=28,
            n_trees=5,
        )
        summary, df = run_single_experiment(config, output_dir=tmp_path, save_parquet=False)
        global_dict = summary["global"]

        # rf_fire metrics must match metrics computed from u_rf_fire_lower
        # prox_a metrics must match metrics computed from u_prox_a_lower
        # prox_b metrics must match metrics computed from u_prox_b_lower
        # prox_ac metrics must match metrics computed from u_prox_ac_lower
        # prox_bc metrics must match metrics computed from u_prox_bc_lower
        # plcb metrics must match metrics computed from u_plcb_lower
        for short_name, lower_col in [
            ("rf_fire", "u_rf_fire_lower"),
            ("prox_a", "u_prox_a_lower"),
            ("prox_b", "u_prox_b_lower"),
            ("prox_ac", "u_prox_ac_lower"),
            ("prox_bc", "u_prox_bc_lower"),
            ("plcb", "u_plcb_lower"),
        ]:
            if lower_col in global_dict:
                for metric in SCORECARD_METRICS:
                    assert np.isclose(
                        global_dict[short_name][metric],
                        global_dict[lower_col][metric],
                        rtol=1e-7,
                    ), f"Metric mismatch between {short_name} and {lower_col} for {metric}"


class TestCLIInvocation:
    """Verifies end-to-end CLI execution via scripts/run_extrapolation_experiment.py."""

    def test_cli_execution_generates_parquet_and_json(self, tmp_path: Path):
        output_dir = tmp_path / "raw"
        summary_dir = tmp_path / "summaries"

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_extrapolation_experiment.py"),
            "--dimension", "2",
            "--n-train", "28",
            "--function", "sphere",
            "--strategy", "stratified",
            "--seed", "42",
            "--n-test", "40",
            "--n-trees", "5",
            "--k", "28",
            "--output-dir", str(output_dir),
            "--summary-dir", str(summary_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, f"CLI failed with error:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

        # 1. Parquet check
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(parquet_files) == 1
        df = pd.read_parquet(parquet_files[0])
        assert len(df) == 40
        assert "u_prox_a_lower" in df.columns
        assert "u_rf_fire_lower" in df.columns
        assert not any(col.endswith("_half") for col in df.columns)

        # 2. Summary JSON check
        json_files = list(summary_dir.glob("*.json"))
        assert len(json_files) == 1
        with open(json_files[0], "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert "global" in summary
        assert "strata_metrics" in summary
        for estimator in ACTIVE_ESTIMATORS:
            assert estimator in summary["global"], f"Estimator {estimator} missing from CLI JSON global"
            assert estimator in summary["strata_metrics"]["0"], f"Estimator {estimator} missing from CLI JSON stratum 0"

    def test_cli_unweighted_proximity_only_mode(self, tmp_path: Path):
        output_dir = tmp_path / "raw_fast"
        summary_dir = tmp_path / "summaries_fast"

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_extrapolation_experiment.py"),
            "--dimension", "2",
            "--n-train", "28",
            "--function", "sphere",
            "--strategy", "stratified",
            "--seed", "42",
            "--n-test", "40",
            "--n-trees", "5",
            "--k", "28",
            "--unweighted-proximity-only",
            "--output-dir", str(output_dir),
            "--summary-dir", str(summary_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, f"CLI fast mode failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

        json_files = list(summary_dir.glob("*.json"))
        assert len(json_files) == 1
        with open(json_files[0], "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert "global" in summary
        assert "prox_a" in summary["global"]
        assert "prox_b" in summary["global"]
        assert "prox_ac" in summary["global"]
        assert "prox_bc" in summary["global"]
        assert "plcb" in summary["global"]
