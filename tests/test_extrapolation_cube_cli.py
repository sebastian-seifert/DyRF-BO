import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq.uq_evaluator import APPROACH_NAMES


def create_mock_parquet_file(filepath: Path, surrogate: str, dim: int, seed: int):
    n_points = 100
    y_true = np.ones(n_points) * 2.0
    y_pred = np.ones(n_points) * 2.1
    strata = np.array([0, 1, 2, 3] * 25)

    data = {
        "surrogate": [surrogate] * n_points,
        "dimension": [dim] * n_points,
        "seed": [seed] * n_points,
        "function_name": ["Sphere"] * n_points,
        "n_train": [50] * n_points,
        "y_true": y_true,
        "y_hat": y_pred,
        "stratum": strata,
    }

    for app in APPROACH_NAMES:
        data[f"u_{app}"] = np.ones(n_points) * 0.5
        for cov in [50, 80, 90, 95]:
            data[f"int_lwr_{app}_{cov}"] = np.ones(n_points) * 1.5
            data[f"int_upr_{app}_{cov}"] = np.ones(n_points) * 2.5

    df = pd.DataFrame(data)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)


def test_cli_execution_and_artifacts(tmp_path: Path):
    input_dir = tmp_path / "raw_parquet"
    output_dir = tmp_path / "cube_output"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Create 2 mock runs
    create_mock_parquet_file(input_dir / "run1.parquet", surrogate="mature", dim=2, seed=1)
    create_mock_parquet_file(input_dir / "run2.parquet", surrogate="breiman", dim=5, seed=2)

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "analyze_extrapolation_uq_cube.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--n-jobs", "2",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed with error:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Verify per_run_evaluations was written
    per_run_parquet = output_dir / "per_run_evaluations.parquet"
    assert per_run_parquet.exists()
    per_run_df = pd.read_parquet(per_run_parquet)
    assert len(per_run_df) == 2 * len(APPROACH_NAMES) * 5  # 2 files * 10 approaches * 5 strata

    # Verify all 8 slice CSV files exist
    expected_slices = [
        "main_global.csv",
        "stratum_global.csv",
        "main_global_by_dimension.csv",
        "main_global_by_surrogate.csv",
        "stratum_by_dimension.csv",
        "stratum_by_surrogate.csv",
        "main_global_rf_by_dimension.csv",
        "full_cube.csv",
    ]
    for slice_file in expected_slices:
        target = output_dir / slice_file
        assert target.exists(), f"Missing expected slice file: {slice_file}"
        df_slice = pd.read_csv(target)
        assert not df_slice.empty
        assert "approach" in df_slice.columns
        assert "ece_mean" in df_slice.columns
        assert "e_aurc_mean" in df_slice.columns
        assert "winkler_mean_mean" in df_slice.columns
