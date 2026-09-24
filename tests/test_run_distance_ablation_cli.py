"""Unit and integration tests for the Distance Metric Ablation CLI and pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_distance_metric_ablation import (
    build_parser,
    generate_ablation_markdown_report,
    main,
    run_distance_ablation_pipeline,
)


def _create_synthetic_parquet(
    target_path: Path,
    dimension: int,
    strategy: str,
    seed: int,
    n_points: int = 100,
) -> None:
    """Helper to create a realistic synthetic extrapolation parquet file."""
    rng = np.random.default_rng(seed)
    d_norm = np.linspace(0.0, 2.0, n_points)
    # Chebyshev distance
    d_inf = d_norm * 0.75 + rng.normal(0, 0.02, n_points)

    # Uncertainty signals
    u_slcb = 1.0 + rng.normal(0, 0.15, n_points)
    u_plcb = 0.5 + 1.2 * d_norm + rng.normal(0, 0.05, n_points)

    stratum = np.zeros(n_points, dtype=int)
    stratum[d_norm > 0.1] = 1
    stratum[d_norm > 0.5] = 2
    stratum[d_norm > 1.0] = 3

    df = pd.DataFrame({
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
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target_path, index=False)


@pytest.fixture
def mock_parquet_dir(tmp_path: Path) -> Path:
    """Fixture creating a directory of synthetic parquet files."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dimensions = [2, 16, 32]
    strategies = ["natural", "stratified"]
    seeds = [0, 1]

    for d in dimensions:
        for strat in strategies:
            for s in seeds:
                filename = f"extrapolation_sphere_d{d}_n{112 if d == 2 else 224}_{strat}_s{s}.parquet"
                _create_synthetic_parquet(raw_dir / filename, dimension=d, strategy=strat, seed=s)

    return raw_dir


class TestCLIParser:
    """Tests for argument parsing."""

    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.input_dir == "results/extrapolation_uq/raw"
        assert args.output_dir == "results/extrapolation_uq/analysis"
        assert args.scorecard_filename == "distance_ablation_scorecard.csv"
        assert args.report_filename == "DISTANCE_ABLATION_REPORT.md"
        assert args.max_files is None

    def test_parser_custom_overrides(self):
        parser = build_parser()
        args = parser.parse_args([
            "--input-dir", "custom/input",
            "--output-dir", "custom/output",
            "--scorecard-filename", "custom_scorecard.csv",
            "--report-filename", "custom_report.md",
            "--max-files", "10",
        ])
        assert args.input_dir == "custom/input"
        assert args.output_dir == "custom/output"
        assert args.scorecard_filename == "custom_scorecard.csv"
        assert args.report_filename == "custom_report.md"
        assert args.max_files == 10


class TestPipelineExecution:
    """Tests for running the ablation analysis pipeline."""

    def test_empty_directory_handling(self, tmp_path: Path):
        empty_in = tmp_path / "empty_raw"
        empty_in.mkdir()
        out_dir = tmp_path / "analysis"

        exit_code = run_distance_ablation_pipeline(
            input_dir=empty_in,
            output_dir=out_dir,
            scorecard_filename="scorecard.csv",
            report_filename="report.md",
        )
        assert exit_code == 0
        assert out_dir.is_dir()
        # Verify placeholder or empty outputs created
        assert (out_dir / "scorecard.csv").is_file()
        assert (out_dir / "report.md").is_file()

    def test_nonexistent_directory_handling(self, tmp_path: Path):
        nonexistent = tmp_path / "does_not_exist"
        out_dir = tmp_path / "analysis"

        exit_code = run_distance_ablation_pipeline(
            input_dir=nonexistent,
            output_dir=out_dir,
        )
        assert exit_code == 0
        assert out_dir.is_dir()

    def test_max_files_limit(self, mock_parquet_dir: Path, tmp_path: Path):
        out_dir = tmp_path / "analysis_max_files"
        scorecard_file = "scorecard.csv"

        exit_code = run_distance_ablation_pipeline(
            input_dir=mock_parquet_dir,
            output_dir=out_dir,
            scorecard_filename=scorecard_file,
            max_files=4,
        )
        assert exit_code == 0

        df_slices = pd.read_csv(out_dir / scorecard_file)
        # Grand total should have 4 experiments
        grand_total = df_slices[
            (df_slices["dimension"].astype(str) == "All") &
            (df_slices["sampling_strategy"] == "All")
        ]
        assert len(grand_total) == 1
        assert grand_total.iloc[0]["n_experiments"] == 4

    def test_end_to_end_pipeline(self, mock_parquet_dir: Path, tmp_path: Path):
        out_dir = tmp_path / "analysis_e2e"
        scorecard_name = "distance_ablation_scorecard.csv"
        report_name = "DISTANCE_ABLATION_REPORT.md"

        exit_code = run_distance_ablation_pipeline(
            input_dir=mock_parquet_dir,
            output_dir=out_dir,
            scorecard_filename=scorecard_name,
            report_filename=report_name,
        )
        assert exit_code == 0

        csv_path = out_dir / scorecard_name
        report_path = out_dir / report_name

        assert csv_path.is_file()
        assert report_path.is_file()

        # Verify CSV content
        df_slices = pd.read_csv(csv_path)
        assert not df_slices.empty
        assert "dimension" in df_slices.columns
        assert "sampling_strategy" in df_slices.columns
        assert "dist_norm_slcb_mean" in df_slices.columns
        assert "dist_inf_slcb_mean" in df_slices.columns
        assert "dist_norm_plcb_mean" in df_slices.columns
        assert "dist_inf_plcb_mean" in df_slices.columns

        # Verify Report content
        report_text = report_path.read_text(encoding="utf-8")
        assert len(report_text) > 500
        assert "Distance Metric Ablation Report" in report_text or "Chebyshev" in report_text
        assert "Executive Summary" in report_text
        assert "Dimension-Wise" in report_text
        assert "Strategy" in report_text
        assert "Stratum-Wise" in report_text
        assert "Chebyshev" in report_text
        assert "Euclidean" in report_text

    def test_cli_main_invocation(self, mock_parquet_dir: Path, tmp_path: Path):
        out_dir = tmp_path / "cli_out"
        argv = [
            "--input-dir", str(mock_parquet_dir),
            "--output-dir", str(out_dir),
            "--scorecard-filename", "test_scorecard.csv",
            "--report-filename", "test_report.md",
        ]
        exit_code = main(argv)
        assert exit_code == 0
        assert (out_dir / "test_scorecard.csv").is_file()
        assert (out_dir / "test_report.md").is_file()
