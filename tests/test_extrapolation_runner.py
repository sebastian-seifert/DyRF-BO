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
    "stratum",
    "d_norm",
    "d_rel",
    "d_inf",
    "is_interpolating",
    "y_true",
    "y_hat",
    "abs_error",
    "u_slcb",
    "u_plcb",
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

        # 2. Check no null values
        assert df.isna().sum().sum() == 0

        # 3. Check data types and value sanity
        assert np.issubdtype(df["point_id"].dtype, np.integer)
        assert np.issubdtype(df["stratum"].dtype, np.integer)
        assert np.issubdtype(df["is_interpolating"].dtype, np.bool_)
        assert np.all(df["d_norm"] >= 0.0)
        assert np.all(df["u_slcb"] > 0.0)
        assert np.all(df["u_plcb"] > 0.0)
        assert np.allclose(df["abs_error"], np.abs(df["y_true"] - df["y_hat"]))

        # 4. Stratified specific checks: 4 strata evenly split
        stratum_counts = df["stratum"].value_counts().to_dict()
        for s in [0, 1, 2, 3]:
            assert stratum_counts.get(s, 0) == 25

        # 5. Verify Parquet output
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 1
        loaded_df = pd.read_parquet(parquet_files[0])
        assert len(loaded_df) == 100
        assert list(loaded_df.columns) == list(df.columns)
        assert np.allclose(loaded_df["y_true"], df["y_true"])

        # 6. Verify summary metrics
        assert isinstance(summary, dict)
        for method in ["slcb", "plcb"]:
            for metric in ["picp", "mpiw", "winkler", "spearman_dist", "spearman_err"]:
                assert f"{method}_{metric}" in summary

        for s in [0, 1, 2, 3]:
            assert f"stratum_{s}_slcb_picp" in summary
            assert f"stratum_{s}_plcb_picp" in summary

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

        # Verify no parquet was saved when save_parquet=False
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 0

        # Summary checks
        assert "slcb_picp" in summary
        assert "plcb_picp" in summary

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
