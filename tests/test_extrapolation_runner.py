"""Unit tests for extrapolation runner pipeline."""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq.runner import (
    ExtrapolationRunConfig,
    run_single_experiment,
)


REQUIRED_DATAFRAME_COLUMNS = [
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
    # Candidate estimators (15 signals + mi + entropy)
    "u_hutter_total",
    "u_hutter_between",
    "u_hutter_within",
    "u_shaker_epistemic",
    "u_shaker_total",
    "shaker_mi",
    "shaker_total_entropy",
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
    # Legacy aliases
    "u_slcb",
    "u_plcb",
    # Diagnostics
    "delta_floor",
    "local_mae",
    "q_lower",
]


class TestExtrapolationRunConfig:
    """Tests for ExtrapolationRunConfig dataclass."""

    def test_default_values(self):
        config = ExtrapolationRunConfig(
            dimension=2,
            n_train=28,
            function_name="sphere",
            sampling_strategy="stratified",
            seed=42,
        )
        assert config.dimension == 2
        assert config.n_train == 28
        assert config.function_name == "sphere"
        assert config.sampling_strategy == "stratified"
        assert config.seed == 42
        assert config.n_test == 10000
        assert config.k == 28
        assert np.isclose(config.eps, 0.080791)
        assert np.isclose(config.decay_lambda, 0.20486)
        assert config.n_trees == 10
        assert config.surrogate_type == "smac_default"
        assert config.surrogate == "smac_default"

    def test_custom_surrogate_types(self):
        for s_type in ["mature", "shallow", "coarse", "breiman"]:
            cfg1 = ExtrapolationRunConfig(
                dimension=2,
                n_train=28,
                function_name="sphere",
                sampling_strategy="stratified",
                seed=42,
                surrogate_type=s_type,
            )
            assert cfg1.surrogate_type == s_type
            assert cfg1.surrogate == s_type

            cfg2 = ExtrapolationRunConfig(
                dimension=2,
                n_train=28,
                function_name="sphere",
                sampling_strategy="stratified",
                seed=42,
                surrogate=s_type,
            )
            assert cfg2.surrogate_type == s_type
            assert cfg2.surrogate == s_type


class TestRunSingleExperiment:
    """Tests for run_single_experiment pipeline."""

    def test_stratified_pilot_experiment(self, tmp_path):
        config = ExtrapolationRunConfig(
            dimension=2,
            n_train=28,
            function_name="sphere",
            sampling_strategy="stratified",
            seed=42,
            n_test=100,
            k=28,
            n_trees=5,
        )

        summary, df = run_single_experiment(config, output_dir=tmp_path, save_parquet=True)

        # 1. Verify returned DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        for col in REQUIRED_DATAFRAME_COLUMNS:
            assert col in df.columns, f"Missing required column: {col}"
        for d in range(config.dimension):
            assert f"x_{d}" in df.columns, f"Missing coordinate column: x_{d}"
            assert np.issubdtype(df[f"x_{d}"].dtype, np.floating)
            assert np.all(df[f"x_{d}"] >= -1.0 - 1e-7)
            assert np.all(df[f"x_{d}"] <= 1.0 + 1e-7)
        assert list(df.columns)[: 1 + config.dimension] == ["point_id", "x_0", "x_1"]

        # 2. Check no null values
        assert df.isna().sum().sum() == 0

        # 3. Check data types and value sanity
        assert np.issubdtype(df["point_id"].dtype, np.integer)
        assert np.issubdtype(df["stratum"].dtype, np.integer)
        assert np.issubdtype(df["is_interpolating"].dtype, np.bool_)
        assert np.all(df["d_norm"] >= 0.0)
        assert np.all(df["u_slcb"] > 0.0)
        assert np.all(df["u_plcb"] > 0.0)
        assert np.all(df["u_hutter_total"] > 0.0)
        assert np.all(df["u_shaker_epistemic"] >= 0.0)
        assert np.all(df["u_rf_fire_half"] >= 0.0)
        assert np.allclose(df["abs_error"], np.abs(df["y_true"] - df["y_hat"]))

        # 4. Stratified specific checks: 4 strata evenly split
        stratum_counts = df["stratum"].value_counts().to_dict()
        for s in [0, 1, 2, 3]:
            assert stratum_counts.get(s, 0) == 25

        # 5. Verify Parquet output
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 1
        assert parquet_files[0].name == "extrapolation_sphere_d2_n28_stratified_smac_default_s42.parquet"
        loaded_df = pd.read_parquet(parquet_files[0])
        assert len(loaded_df) == 100
        assert list(loaded_df.columns) == list(df.columns)
        assert "surrogate" in loaded_df.columns
        assert (loaded_df["surrogate"] == "smac_default").all()
        assert np.allclose(loaded_df["y_true"], df["y_true"])
        for d in range(config.dimension):
            assert np.allclose(loaded_df[f"x_{d}"], df[f"x_{d}"])

        # 6. Verify summary metrics
        assert isinstance(summary, dict)
        assert summary.get("surrogate_type") == "smac_default"
        for method in ["slcb", "plcb", "u_hutter_total", "u_shaker_epistemic"]:
            for metric in ["picp", "mpiw", "winkler", "spearman_dist", "spearman_err"]:
                assert f"{method}_{metric}" in summary

        for s in [0, 1, 2, 3]:
            assert f"stratum_{s}_slcb_picp" in summary
            assert f"stratum_{s}_plcb_picp" in summary
            assert f"stratum_{s}_u_hutter_total_picp" in summary

    def test_natural_pilot_experiment(self, tmp_path):
        config = ExtrapolationRunConfig(
            dimension=2,
            n_train=28,
            function_name="sphere",
            sampling_strategy="natural",
            seed=123,
            n_test=100,
            k=28,
            n_trees=5,
        )

        summary, df = run_single_experiment(config, output_dir=tmp_path, save_parquet=False)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        for col in REQUIRED_DATAFRAME_COLUMNS:
            assert col in df.columns
        assert (df["surrogate"] == "smac_default").all()
        assert summary.get("surrogate_type") == "smac_default"
        for d in range(config.dimension):
            assert f"x_{d}" in df.columns, f"Missing coordinate column: x_{d}"
            assert np.issubdtype(df[f"x_{d}"].dtype, np.floating)
            assert np.all(df[f"x_{d}"] >= -1.0 - 1e-7)
            assert np.all(df[f"x_{d}"] <= 1.0 + 1e-7)
        assert list(df.columns)[: 1 + config.dimension] == ["point_id", "x_0", "x_1"]

        # Verify no parquet was saved when save_parquet=False
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 0

        # Summary checks
        assert "slcb_picp" in summary
        assert "plcb_picp" in summary

    def test_point_coordinates_stored_and_bounded(self, tmp_path):
        config = ExtrapolationRunConfig(
            dimension=3,
            n_train=28,
            function_name="sphere",
            sampling_strategy="stratified",
            seed=42,
            n_test=60,
            k=28,
            n_trees=5,
        )
        summary, df = run_single_experiment(config, output_dir=tmp_path, save_parquet=True)
        assert len(df) == 60
        coord_cols = [f"x_{d}" for d in range(3)]
        for col in coord_cols:
            assert col in df.columns, f"Missing coordinate column: {col}"
            assert np.issubdtype(df[col].dtype, np.floating)
            assert np.all(df[col] >= -1.0 - 1e-7)
            assert np.all(df[col] <= 1.0 + 1e-7)

        # Check column ordering: point_id followed immediately by x_0, x_1, x_2
        cols = list(df.columns)
        assert cols[:4] == ["point_id", "x_0", "x_1", "x_2"]

        # Check Parquet roundtrip
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 1
        assert parquet_files[0].name == "extrapolation_sphere_d3_n28_stratified_smac_default_s42.parquet"
        loaded_df = pd.read_parquet(parquet_files[0])
        assert list(loaded_df.columns) == list(df.columns)
        for col in coord_cols:
            assert np.allclose(loaded_df[col], df[col])

    def test_higher_dimensional_coordinates(self, tmp_path):
        config = ExtrapolationRunConfig(
            dimension=5,
            n_train=28,
            function_name="sphere",
            sampling_strategy="natural",
            seed=999,
            n_test=40,
            k=28,
            n_trees=5,
        )
        summary, df = run_single_experiment(config, output_dir=tmp_path, save_parquet=False)
        assert len(df) == 40
        coord_cols = [f"x_{d}" for d in range(5)]
        for col in coord_cols:
            assert col in df.columns, f"Missing coordinate column: {col}"
            assert np.issubdtype(df[col].dtype, np.floating)
            assert np.all(df[col] >= -1.0 - 1e-7)
            assert np.all(df[col] <= 1.0 + 1e-7)
        cols = list(df.columns)
        assert cols[:6] == ["point_id", "x_0", "x_1", "x_2", "x_3", "x_4"]

    def test_invalid_sampling_strategy(self):
        config = ExtrapolationRunConfig(
            dimension=2,
            n_train=28,
            function_name="sphere",
            sampling_strategy="invalid_strategy",
            seed=42,
            n_test=100,
        )
        with pytest.raises(ValueError, match="strategy"):
            run_single_experiment(config)

    def test_invalid_objective_function(self):
        config = ExtrapolationRunConfig(
            dimension=2,
            n_train=28,
            function_name="non_existent_func",
            sampling_strategy="stratified",
            seed=42,
            n_test=100,
        )
        with pytest.raises(KeyError):
            run_single_experiment(config)

    def test_lossless_pyarrow_parquet_roundtrip_all_candidate_dtypes(self, tmp_path):
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
        parquet_file = list(tmp_path.glob("*.parquet"))[0]
        assert parquet_file.name == "extrapolation_sphere_d2_n28_stratified_smac_default_s42.parquet"
        reloaded = pd.read_parquet(parquet_file, engine="pyarrow")

        # 1. Exact column list and length
        assert list(reloaded.columns) == list(df.columns)
        assert len(reloaded) == len(df)

        # 2. Strict dtypes
        assert reloaded["point_id"].dtype == np.int64
        assert reloaded["stratum"].dtype == np.int64
        assert reloaded["is_interpolating"].dtype == bool
        for col in reloaded.columns:
            if col in ["point_id", "stratum"]:
                assert reloaded[col].dtype == np.int64
            elif col == "is_interpolating":
                assert reloaded[col].dtype == bool
            elif col == "surrogate":
                assert reloaded[col].dtype == object or "string" in str(reloaded[col].dtype)
            else:
                assert reloaded[col].dtype == np.float64

        # 3. Value equality with zero precision loss
        for col in reloaded.columns:
            if col == "is_interpolating":
                assert (reloaded[col] == df[col]).all()
            elif col in ["point_id", "stratum", "surrogate"]:
                assert (reloaded[col] == df[col]).all()
            else:
                assert np.array_equal(reloaded[col].to_numpy(), df[col].to_numpy())

    @pytest.mark.parametrize(
        "surrogate_type",
        ["smac_default", "mature", "shallow", "coarse", "breiman"],
    )
    def test_all_five_surrogate_types_run_single_experiment(self, tmp_path, surrogate_type):
        config = ExtrapolationRunConfig(
            dimension=2,
            n_train=20,
            function_name="sphere",
            sampling_strategy="stratified",
            seed=42,
            n_test=20,
            k=5,
            n_trees=5,
            surrogate_type=surrogate_type,
        )
        summary, df = run_single_experiment(config, output_dir=tmp_path, save_parquet=True)
        assert df["surrogate"].iloc[0] == surrogate_type
        assert (df["surrogate"] == surrogate_type).all()
        assert summary["surrogate_type"] == surrogate_type
        expected_parquet = tmp_path / f"extrapolation_sphere_d2_n20_stratified_{surrogate_type}_s42.parquet"
        assert expected_parquet.exists()


