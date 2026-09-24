"""Unit and integration tests for the Extrapolation Distance Metric Ablation Study."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dyrf_bo.extrapolation_uq.distance_ablation import (
    aggregate_distance_ablation,
    compute_run_distance_ablation,
    parse_parquet_metadata,
)


class TestParseParquetMetadata:
    """Tests for parsing run metadata from standard Parquet filenames."""

    def test_valid_filename_simple(self):
        filename = "extrapolation_sphere_d2_n112_natural_s0.parquet"
        meta = parse_parquet_metadata(filename)
        assert meta["function_name"] == "sphere"
        assert meta["dimension"] == 2
        assert meta["n_train"] == 112
        assert meta["sampling_strategy"] == "natural"
        assert meta["seed"] == 0
        assert isinstance(meta["dimension"], int)
        assert isinstance(meta["n_train"], int)
        assert isinstance(meta["seed"], int)

    def test_valid_filepath_with_path(self):
        fpath = Path("/workspace/results/extrapolation_uq/raw/extrapolation_rosenbrock_d16_n448_stratified_s42.parquet")
        meta = parse_parquet_metadata(fpath)
        assert meta["function_name"] == "rosenbrock"
        assert meta["dimension"] == 16
        assert meta["n_train"] == 448
        assert meta["sampling_strategy"] == "stratified"
        assert meta["seed"] == 42

    def test_valid_filename_multidigit(self):
        filename = "extrapolation_rastrigin_d32_n896_stratified_s1234.parquet"
        meta = parse_parquet_metadata(filename)
        assert meta["function_name"] == "rastrigin"
        assert meta["dimension"] == 32
        assert meta["n_train"] == 896
        assert meta["sampling_strategy"] == "stratified"
        assert meta["seed"] == 1234

    def test_invalid_filename_raises(self):
        with pytest.raises(ValueError, match="Invalid filename format"):
            parse_parquet_metadata("summary_sphere_d2.json")

    def test_invalid_extension_raises(self):
        with pytest.raises(ValueError, match="Invalid filename format"):
            parse_parquet_metadata("extrapolation_sphere_d2_n112_natural_s0.csv")


class TestComputeRunDistanceAblation:
    """Tests for computing distance ablation metrics per run."""

    @pytest.fixture
    def mock_eval_df(self) -> pd.DataFrame:
        """Create synthetic point evaluations dataframe."""
        rng = np.random.default_rng(42)
        n_points = 200

        # Create synthetic distances
        d_norm = np.linspace(0.0, 2.0, n_points)
        # Chebyshev distance roughly proportional to Euclidean distance
        d_inf = d_norm * 0.7 + rng.normal(0, 0.05, n_points)

        # Uncertainty signals:
        # SLCB has low/collapsed correlation with distance
        u_slcb = 1.0 + rng.normal(0, 0.2, n_points)
        # PLCB has strong positive correlation with d_norm
        u_plcb = 0.5 + 1.5 * d_norm + rng.normal(0, 0.1, n_points)

        # Strata: 0 (<=0.1), 1 (0.1-0.5), 2 (0.5-1.0), 3 (>1.0)
        stratum = np.zeros(n_points, dtype=int)
        stratum[d_norm > 0.1] = 1
        stratum[d_norm > 0.5] = 2
        stratum[d_norm > 1.0] = 3

        return pd.DataFrame({
            "point_id": np.arange(n_points),
            "stratum": stratum,
            "d_norm": d_norm,
            "d_rel": d_norm * 0.5,
            "d_inf": d_inf,
            "is_interpolating": d_norm <= 0.0,
            "y_true": rng.normal(0, 1, n_points),
            "y_hat": rng.normal(0, 1, n_points),
            "abs_error": rng.exponential(1, n_points),
            "u_slcb": u_slcb,
            "u_plcb": u_plcb,
        })

    def test_compute_metrics_from_dataframe(self, mock_eval_df: pd.DataFrame):
        metrics = compute_run_distance_ablation(mock_eval_df)

        # Check required global metrics
        assert "spearman_dist_norm_slcb" in metrics
        assert "spearman_dist_inf_slcb" in metrics
        assert "diff_slcb" in metrics
        assert "spearman_dist_norm_plcb" in metrics
        assert "spearman_dist_inf_plcb" in metrics
        assert "diff_plcb" in metrics

        # Verify difference definition: diff = inf - norm
        assert pytest.approx(metrics["diff_slcb"]) == metrics["spearman_dist_inf_slcb"] - metrics["spearman_dist_norm_slcb"]
        assert pytest.approx(metrics["diff_plcb"]) == metrics["spearman_dist_inf_plcb"] - metrics["spearman_dist_norm_plcb"]

        # PLCB should have high correlation with d_norm
        assert metrics["spearman_dist_norm_plcb"] > 0.8

        # Check per-stratum metrics
        for s in range(4):
            assert f"stratum_{s}_spearman_dist_norm_slcb" in metrics
            assert f"stratum_{s}_spearman_dist_inf_slcb" in metrics
            assert f"stratum_{s}_diff_slcb" in metrics
            assert f"stratum_{s}_spearman_dist_norm_plcb" in metrics
            assert f"stratum_{s}_spearman_dist_inf_plcb" in metrics
            assert f"stratum_{s}_diff_plcb" in metrics

    def test_compute_metrics_from_parquet_file(self, mock_eval_df: pd.DataFrame, tmp_path: Path):
        parquet_path = tmp_path / "extrapolation_sphere_d5_n224_stratified_s7.parquet"
        mock_eval_df.to_parquet(parquet_path, index=False)

        metrics = compute_run_distance_ablation(parquet_path)

        # Check that metadata was parsed and merged
        assert metrics["function_name"] == "sphere"
        assert metrics["dimension"] == 5
        assert metrics["n_train"] == 224
        assert metrics["sampling_strategy"] == "stratified"
        assert metrics["seed"] == 7
        assert "spearman_dist_norm_plcb" in metrics

    def test_edge_case_constant_uncertainty(self):
        n_points = 50
        df = pd.DataFrame({
            "d_norm": np.linspace(0.1, 1.0, n_points),
            "d_inf": np.linspace(0.1, 0.8, n_points),
            "u_slcb": np.full(n_points, 1.5),  # Constant
            "u_plcb": np.full(n_points, 2.0),  # Constant
            "stratum": np.zeros(n_points, dtype=int),
        })
        metrics = compute_run_distance_ablation(df)
        assert metrics["spearman_dist_norm_slcb"] == 0.0
        assert metrics["spearman_dist_inf_slcb"] == 0.0
        assert metrics["diff_slcb"] == 0.0
        assert metrics["spearman_dist_norm_plcb"] == 0.0
        assert metrics["spearman_dist_inf_plcb"] == 0.0
        assert metrics["diff_plcb"] == 0.0


class TestAggregateDistanceAblation:
    """Tests for aggregating distance ablation records across dimensions and strategies."""

    @pytest.fixture
    def mock_records(self) -> List[Dict[str, Any]]:
        """Create mock records spanning multiple dimensions, strategies, and seeds."""
        records: List[Dict[str, Any]] = []
        dimensions = [2, 16, 32]
        strategies = ["natural", "stratified"]
        seeds = [0, 1]

        for d in dimensions:
            for strat in strategies:
                for s in seeds:
                    # In higher dimensions, Chebyshev might correlate slightly better with LCB
                    norm_slcb = 0.15 - 0.005 * d
                    inf_slcb = 0.20 - 0.003 * d
                    norm_plcb = 0.85 - 0.001 * d
                    inf_plcb = 0.80 - 0.001 * d

                    rec: Dict[str, Any] = {
                        "function_name": "sphere",
                        "dimension": d,
                        "n_train": 112 if d == 2 else 224,
                        "sampling_strategy": strat,
                        "seed": s,
                        "spearman_dist_norm_slcb": norm_slcb,
                        "spearman_dist_inf_slcb": inf_slcb,
                        "diff_slcb": inf_slcb - norm_slcb,
                        "spearman_dist_norm_plcb": norm_plcb,
                        "spearman_dist_inf_plcb": inf_plcb,
                        "diff_plcb": inf_plcb - norm_plcb,
                    }

                    # Add per-stratum data
                    for st in range(4):
                        rec[f"stratum_{st}_spearman_dist_norm_slcb"] = norm_slcb * (1 - 0.1 * st)
                        rec[f"stratum_{st}_spearman_dist_inf_slcb"] = inf_slcb * (1 - 0.1 * st)
                        rec[f"stratum_{st}_diff_slcb"] = (inf_slcb - norm_slcb) * (1 - 0.1 * st)
                        rec[f"stratum_{st}_spearman_dist_norm_plcb"] = norm_plcb * (1 - 0.05 * st)
                        rec[f"stratum_{st}_spearman_dist_inf_plcb"] = inf_plcb * (1 - 0.05 * st)
                        rec[f"stratum_{st}_diff_plcb"] = (inf_plcb - norm_plcb) * (1 - 0.05 * st)

                    records.append(rec)
        return records

    def test_aggregate_distance_ablation_slices(self, mock_records: List[Dict[str, Any]]):
        df_slices, df_strata = aggregate_distance_ablation(mock_records)

        assert isinstance(df_slices, pd.DataFrame)
        assert not df_slices.empty

        # Check required columns for df_slices
        expected_cols = [
            "dimension",
            "sampling_strategy",
            "n_experiments",
            # SLCB
            "dist_norm_slcb_mean", "dist_norm_slcb_sem",
            "dist_inf_slcb_mean", "dist_inf_slcb_sem",
            "diff_slcb_mean", "pvalue_slcb", "cliffs_delta_slcb",
            "wins_slcb", "ties_slcb", "losses_slcb",
            # PLCB
            "dist_norm_plcb_mean", "dist_norm_plcb_sem",
            "dist_inf_plcb_mean", "dist_inf_plcb_sem",
            "diff_plcb_mean", "pvalue_plcb", "cliffs_delta_plcb",
            "wins_plcb", "ties_plcb", "losses_plcb",
        ]
        for col in expected_cols:
            assert col in df_slices.columns, f"Missing column {col} in df_slices"

        # Check win/tie/loss sum equals n_experiments
        for _, row in df_slices.iterrows():
            n = row["n_experiments"]
            assert row["wins_slcb"] + row["ties_slcb"] + row["losses_slcb"] == n
            assert row["wins_plcb"] + row["ties_plcb"] + row["losses_plcb"] == n

        # Check presence of dimension marginals ('All') and grand total ('All', 'All')
        grand_total = df_slices[(df_slices["dimension"].astype(str) == "All") & (df_slices["sampling_strategy"] == "All")]
        assert len(grand_total) == 1
        assert grand_total.iloc[0]["n_experiments"] == len(mock_records)

        # Check dimension 32 slice
        d32_slice = df_slices[df_slices["dimension"] == 32]
        assert not d32_slice.empty

    def test_aggregate_distance_ablation_strata(self, mock_records: List[Dict[str, Any]]):
        df_slices, df_strata = aggregate_distance_ablation(mock_records)

        assert isinstance(df_strata, pd.DataFrame)
        assert not df_strata.empty

        # Strata 0, 1, 2, 3 should all be represented
        assert set(df_strata["stratum"].unique()) == {0, 1, 2, 3}

        # Check required columns for df_strata
        expected_strata_cols = [
            "stratum",
            "dist_norm_slcb_mean", "dist_norm_slcb_sem",
            "dist_inf_slcb_mean", "dist_inf_slcb_sem",
            "diff_slcb_mean", "diff_slcb_sem",
            "dist_norm_plcb_mean", "dist_norm_plcb_sem",
            "dist_inf_plcb_mean", "dist_inf_plcb_sem",
            "diff_plcb_mean", "diff_plcb_sem",
        ]
        for col in expected_strata_cols:
            assert col in df_strata.columns, f"Missing column {col} in df_strata"

    def test_aggregate_empty_records(self):
        df_slices, df_strata = aggregate_distance_ablation([])
        assert isinstance(df_slices, pd.DataFrame)
        assert isinstance(df_strata, pd.DataFrame)
        assert df_slices.empty
        assert df_strata.empty
        assert "dist_norm_slcb_mean" in df_slices.columns
        assert "dist_norm_slcb_mean" in df_strata.columns
