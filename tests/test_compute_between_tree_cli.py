"""Tests for compute_between_tree_and_update_csvs CLI."""

import os
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.compute_between_tree_and_update_csvs import update_slice_csv, run_between_tree_update


@pytest.fixture
def mock_experiment_env(tmp_path):
    """Creates mock raw parquet files and mock slice CSV files."""
    input_dir = tmp_path / "raw"
    csv_dir = tmp_path / "analysis"
    input_dir.mkdir(parents=True)
    csv_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)

    # 1. Create 2 synthetic parquet runs
    for seed, dim, surrogate in [(0, 2, "smac_default"), (1, 5, "mature")]:
        n_per_stratum = 15
        n_total = n_per_stratum * 4
        strata = np.repeat([0, 1, 2, 3], n_per_stratum)
        y_true = rng.normal(0, 1, size=n_total)
        y_hat = y_true + rng.normal(0, 0.3, size=n_total)
        abs_error = np.abs(y_true - y_hat)
        u_between = np.maximum(0.05, abs_error + 0.2)

        df = pd.DataFrame({
            "point_id": np.arange(n_total, dtype=np.int64),
            "dimension": np.full(n_total, dim, dtype=np.int64),
            "n_train": np.full(n_total, 34, dtype=np.int64),
            "function_name": np.full(n_total, "sphere", dtype=object),
            "seed": np.full(n_total, seed, dtype=np.int64),
            "surrogate": np.full(n_total, surrogate, dtype=object),
            "stratum": strata,
            "d_norm": strata * 0.25,
            "d_rel": strata * 0.25,
            "d_inf": strata * 0.25,
            "is_interpolating": strata == 0,
            "y_true": y_true,
            "y_hat": y_hat,
            "abs_error": abs_error,
            "u_hutter_between": u_between,
        })
        df.to_parquet(input_dir / f"run_d{dim}_{surrogate}_s{seed}.parquet")

    # 2. Create mock slice CSVs with a baseline row
    mock_main_global = pd.DataFrame([{
        "approach": "prox_bc_oob",
        "ece_mean": 0.42, "ece_std": 0.1, "ece_sem": 0.01, "ece_median": 0.41, "ece_iqr": 0.05,
        "winkler_mean_mean": 80.0, "winkler_mean_std": 10.0, "winkler_mean_sem": 1.0, "winkler_mean_median": 78.0, "winkler_mean_iqr": 5.0,
        "winkler_95_mean": 170.0, "winkler_95_std": 20.0, "winkler_95_sem": 2.0, "winkler_95_median": 165.0, "winkler_95_iqr": 10.0,
        "winkler_90_mean": 90.0, "winkler_90_std": 10.0, "winkler_90_sem": 1.0, "winkler_90_median": 88.0, "winkler_90_iqr": 5.0,
        "winkler_80_mean": 50.0, "winkler_80_std": 5.0, "winkler_80_sem": 0.5, "winkler_80_median": 49.0, "winkler_80_iqr": 3.0,
        "winkler_50_mean": 20.0, "winkler_50_std": 2.0, "winkler_50_sem": 0.2, "winkler_50_median": 19.0, "winkler_50_iqr": 1.0,
        "e_aurc_mean": 3.5, "e_aurc_std": 0.5, "e_aurc_sem": 0.05, "e_aurc_median": 3.4, "e_aurc_iqr": 0.3,
        "aurc_mean": 5.0, "aurc_std": 0.5, "aurc_sem": 0.05, "aurc_median": 4.9, "aurc_iqr": 0.3,
        "oracle_aurc_mean": 1.5, "oracle_aurc_std": 0.2, "oracle_aurc_sem": 0.02, "oracle_aurc_median": 1.4, "oracle_aurc_iqr": 0.1,
        "random_aurc_mean": 5.5, "random_aurc_std": 0.5, "random_aurc_sem": 0.05, "random_aurc_median": 5.4, "random_aurc_iqr": 0.3,
        "naurc_mean": 0.88, "naurc_std": 0.05, "naurc_sem": 0.005, "naurc_median": 0.87, "naurc_iqr": 0.02,
        "n_runs": 2,
    }])
    mock_main_global.to_csv(csv_dir / "main_global.csv", index=False)

    return input_dir, csv_dir


def test_update_slice_csv_appends_and_deduplicates(tmp_path):
    csv_path = tmp_path / "test_slice.csv"
    initial_df = pd.DataFrame([
        {"approach": "approach_1", "metric_mean": 1.0, "n_runs": 10},
        {"approach": "hutter_between", "metric_mean": 99.0, "n_runs": 5},  # stale row
    ])
    initial_df.to_csv(csv_path, index=False)

    new_hutter_df = pd.DataFrame([
        {"approach": "hutter_between", "metric_mean": 2.5, "n_runs": 20},
    ])

    update_slice_csv(csv_path, new_hutter_df, key_cols=["approach"], backup=True)

    updated = pd.read_csv(csv_path)
    assert len(updated) == 2
    assert "approach_1" in updated["approach"].values
    assert "hutter_between" in updated["approach"].values
    hutter_row = updated[updated["approach"] == "hutter_between"].iloc[0]
    assert hutter_row["metric_mean"] == 2.5
    assert hutter_row["n_runs"] == 20
    assert (tmp_path / "test_slice.csv.bak").exists()


def test_run_between_tree_update_end_to_end(mock_experiment_env):
    input_dir, csv_dir = mock_experiment_env

    run_between_tree_update(
        input_dir=input_dir,
        csv_dir=csv_dir,
        n_jobs=1,
        backup=True,
    )

    main_global_file = csv_dir / "main_global.csv"
    assert main_global_file.exists()
    df = pd.read_csv(main_global_file)
    assert len(df) == 2
    assert "prox_bc_oob" in df["approach"].values
    assert "hutter_between" in df["approach"].values

    hutter_val = df[df["approach"] == "hutter_between"].iloc[0]
    assert hutter_val["n_runs"] == 2
    assert np.isfinite(hutter_val["ece_mean"])
    assert np.isfinite(hutter_val["winkler_mean_mean"])
    assert np.isfinite(hutter_val["e_aurc_mean"])
