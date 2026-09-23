"""Unit and integration tests for extrapolation results aggregation and scorecard pipeline."""

from __future__ import annotations

import json
import os
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

from scripts.aggregate_extrapolation_results import (
    build_parser,
    build_scorecard_dataframe,
    compute_cliffs_delta,
    compute_paired_comparison,
    compute_paired_wilcoxon,
    generate_markdown_report,
    generate_notion_scorecard,
    load_summary_records,
    main,
    run_aggregation,
)


def _make_mock_summary(
    dimension: int,
    n_train: int,
    function_name: str,
    strategy: str,
    seed: int,
    slcb_spearman_dist: float,
    plcb_spearman_dist: float,
    slcb_spearman_err: float,
    plcb_spearman_err: float,
    slcb_picp: float,
    plcb_picp: float,
    slcb_winkler: float,
    plcb_winkler: float,
    slcb_auroc: float,
    plcb_auroc: float,
) -> Dict[str, Any]:
    """Create a mock summary dictionary matching run_single_experiment output schema."""
    summary: Dict[str, Any] = {
        "dimension": dimension,
        "n_train": n_train,
        "function_name": function_name,
        "sampling_strategy": strategy,
        "seed": seed,
        "n_test": 1000,
        "elapsed_seconds": 1.5,
        # Global metrics
        "spearman_dist_slcb": slcb_spearman_dist,
        "spearman_dist_plcb": plcb_spearman_dist,
        "spearman_err_slcb": slcb_spearman_err,
        "spearman_err_plcb": plcb_spearman_err,
        "picp_slcb": slcb_picp,
        "picp_plcb": plcb_picp,
        "mpiw_slcb": 1.2,
        "mpiw_plcb": 1.8,
        "winkler_slcb": slcb_winkler,
        "winkler_plcb": plcb_winkler,
        "auroc_slcb": slcb_auroc,
        "auroc_plcb": plcb_auroc,
        "outlier_auroc_slcb": slcb_auroc,
        "outlier_auroc_plcb": plcb_auroc,
        "auprc_slcb": 0.4,
        "auprc_plcb": 0.7,
        "global": {
            "slcb": {
                "spearman_dist": slcb_spearman_dist,
                "spearman_err": slcb_spearman_err,
                "picp": slcb_picp,
                "mpiw": 1.2,
                "winkler": slcb_winkler,
                "auroc": slcb_auroc,
                "auprc": 0.4,
            },
            "plcb": {
                "spearman_dist": plcb_spearman_dist,
                "spearman_err": plcb_spearman_err,
                "picp": plcb_picp,
                "mpiw": 1.8,
                "winkler": plcb_winkler,
                "auroc": plcb_auroc,
                "auprc": 0.7,
            },
        },
    }

    # Strata breakdown
    strata_data = {}
    for s in range(4):
        strata_data[str(s)] = {
            "slcb": {
                "spearman_dist": max(-0.2, slcb_spearman_dist - 0.05 * s),
                "spearman_err": slcb_spearman_err,
                "picp": max(0.2, slcb_picp - 0.1 * s),
                "mpiw": 1.2,
                "winkler": slcb_winkler * (1.0 + 0.2 * s),
                "auroc": slcb_auroc,
            },
            "plcb": {
                "spearman_dist": max(0.4, plcb_spearman_dist - 0.02 * s),
                "spearman_err": plcb_spearman_err,
                "picp": plcb_picp,
                "mpiw": 1.8 * (1.0 + 0.1 * s),
                "winkler": plcb_winkler * (1.0 + 0.05 * s),
                "auroc": plcb_auroc,
            },
        }
    summary["strata"] = strata_data
    summary["strata_metrics"] = strata_data
    return summary


@pytest.fixture
def mock_summaries_dir(tmp_path: Path) -> Path:
    """Fixture creating a directory of realistic mock extrapolation summaries."""
    summaries_dir = tmp_path / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    dimensions = [2, 16, 32]
    strategies = ["natural", "stratified"]
    seeds = [0, 1]
    functions = ["sphere", "rastrigin"]

    for d in dimensions:
        for strat in strategies:
            for s in seeds:
                for fn in functions:
                    # In higher dimensions, SLCB collapses (near 0 or negative correlation)
                    # while PLCB stays strong (> 0.75)
                    if d == 2:
                        slcb_dist = 0.35 + 0.02 * s
                        plcb_dist = 0.88 + 0.01 * s
                    elif d == 16:
                        slcb_dist = 0.05 - 0.02 * s
                        plcb_dist = 0.82 + 0.01 * s
                    else:  # 32
                        slcb_dist = -0.08 + 0.01 * s
                        plcb_dist = 0.79 + 0.02 * s

                    summary_data = _make_mock_summary(
                        dimension=d,
                        n_train=112 if d == 2 else 224,
                        function_name=fn,
                        strategy=strat,
                        seed=s,
                        slcb_spearman_dist=slcb_dist,
                        plcb_spearman_dist=plcb_dist,
                        slcb_spearman_err=0.25,
                        plcb_spearman_err=0.65,
                        slcb_picp=0.62,
                        plcb_picp=0.94,
                        slcb_winkler=45.0,
                        plcb_winkler=15.0,
                        slcb_auroc=0.55,
                        plcb_auroc=0.88,
                    )

                    filename = f"summary_{fn}_d{d}_n{summary_data['n_train']}_{strat}_s{s}.json"
                    with open(summaries_dir / filename, "w", encoding="utf-8") as f:
                        json.dump(summary_data, f, indent=2)

    return summaries_dir


class TestStatisticalHelpers:
    """Unit tests for statistical helpers: Cliff's Delta, Wilcoxon test, paired comparisons."""

    def test_cliffs_delta_all_greater(self):
        x = [10.0, 20.0, 30.0]
        y = [1.0, 2.0, 3.0]
        delta = compute_cliffs_delta(x, y)
        assert pytest.approx(delta) == 1.0

    def test_cliffs_delta_all_smaller(self):
        x = [1.0, 2.0, 3.0]
        y = [10.0, 20.0, 30.0]
        delta = compute_cliffs_delta(x, y)
        assert pytest.approx(delta) == -1.0

    def test_cliffs_delta_identical(self):
        x = [5.0, 5.0, 5.0]
        y = [5.0, 5.0, 5.0]
        delta = compute_cliffs_delta(x, y)
        assert pytest.approx(delta) == 0.0

    def test_cliffs_delta_empty(self):
        delta = compute_cliffs_delta([], [])
        assert np.isnan(delta) or delta == 0.0

    def test_paired_wilcoxon_valid(self):
        x = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        stat, pval = compute_paired_wilcoxon(x, y)
        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0
        assert pval < 0.05  # Strongly significant

    def test_paired_wilcoxon_identical_zero_diff(self):
        x = [5.0, 5.0, 5.0, 5.0]
        y = [5.0, 5.0, 5.0, 5.0]
        stat, pval = compute_paired_wilcoxon(x, y)
        assert pytest.approx(stat) == 0.0
        assert pytest.approx(pval) == 1.0

    def test_paired_wilcoxon_single_element(self):
        x = [10.0]
        y = [5.0]
        stat, pval = compute_paired_wilcoxon(x, y)
        # Should not raise exception; return fallback or valid statistic
        assert not np.isnan(stat)
        assert not np.isnan(pval)

    def test_paired_comparison_higher_is_better(self):
        df = pd.DataFrame({
            "plcb": [0.8, 0.9, 0.7, 0.85],
            "slcb": [0.2, 0.3, 0.25, 0.3],
        })
        comp = compute_paired_comparison(df, plcb_col="plcb", slcb_col="slcb", higher_is_better=True)
        assert comp["wins"] == 4
        assert comp["losses"] == 0
        assert comp["ties"] == 0
        assert comp["plcb_mean"] > comp["slcb_mean"]
        assert comp["diff_mean"] > 0.0
        assert comp["cliffs_delta"] > 0.9

    def test_paired_comparison_lower_is_better(self):
        # E.g. Winkler score or PICP error
        df = pd.DataFrame({
            "plcb": [10.0, 12.0, 11.0],
            "slcb": [30.0, 28.0, 35.0],
        })
        comp = compute_paired_comparison(df, plcb_col="plcb", slcb_col="slcb", higher_is_better=False)
        assert comp["wins"] == 3
        assert comp["losses"] == 0
        assert comp["ties"] == 0
        assert comp["diff_mean"] < 0.0  # plcb - slcb is negative, which is desirable


class TestLoadSummaryRecords:
    """Tests for loading and parsing summary JSON files into structured DataFrames."""

    def test_load_nonexistent_directory(self, tmp_path: Path):
        nonexistent = tmp_path / "does_not_exist"
        df = load_summary_records(nonexistent)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_load_empty_directory(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        df = load_summary_records(empty_dir)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_load_valid_summaries(self, mock_summaries_dir: Path):
        df = load_summary_records(mock_summaries_dir)
        assert isinstance(df, pd.DataFrame)
        # 3 dimensions * 2 strategies * 2 seeds * 2 functions = 24 records
        assert len(df) == 24

        # Check essential columns
        expected_cols = [
            "dimension",
            "n_train",
            "function_name",
            "sampling_strategy",
            "seed",
            "spearman_dist_plcb",
            "spearman_dist_slcb",
            "spearman_err_plcb",
            "spearman_err_slcb",
            "picp_plcb",
            "picp_slcb",
            "picp_error_plcb",
            "picp_error_slcb",
            "winkler_plcb",
            "winkler_slcb",
            "outlier_auroc_plcb",
            "outlier_auroc_slcb",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing expected column: {col}"

        # Check derived picp_error = abs(picp - 0.95)
        for _, row in df.iterrows():
            assert pytest.approx(row["picp_error_plcb"]) == abs(row["picp_plcb"] - 0.95)
            assert pytest.approx(row["picp_error_slcb"]) == abs(row["picp_slcb"] - 0.95)

    def test_strata_metrics_extraction(self, mock_summaries_dir: Path):
        df = load_summary_records(mock_summaries_dir)
        # Strata 0, 1, 2, 3 should have columns in the DataFrame
        for s in range(4):
            assert f"stratum_{s}_spearman_dist_plcb" in df.columns
            assert f"stratum_{s}_spearman_dist_slcb" in df.columns
            assert f"stratum_{s}_winkler_plcb" in df.columns


class TestScorecardGeneration:
    """Tests for generating the comprehensive calibration scorecard DataFrame."""

    def test_build_scorecard_dataframe(self, mock_summaries_dir: Path):
        df = load_summary_records(mock_summaries_dir)
        scorecard = build_scorecard_dataframe(df)

        assert isinstance(scorecard, pd.DataFrame)
        assert not scorecard.empty

        # Must group by dimension and sampling_strategy, plus an 'All' overall row
        assert "dimension" in scorecard.columns
        assert "sampling_strategy" in scorecard.columns

        # Verify target metric summary columns
        metrics = ["spearman_dist", "spearman_err", "picp_error", "winkler", "outlier_auroc"]
        for m in metrics:
            assert f"{m}_plcb_mean" in scorecard.columns
            assert f"{m}_plcb_sem" in scorecard.columns
            assert f"{m}_slcb_mean" in scorecard.columns
            assert f"{m}_slcb_sem" in scorecard.columns
            assert f"{m}_diff_mean" in scorecard.columns
            assert f"{m}_pvalue" in scorecard.columns
            assert f"{m}_cliffs_delta" in scorecard.columns
            assert f"{m}_wins" in scorecard.columns
            assert f"{m}_ties" in scorecard.columns
            assert f"{m}_losses" in scorecard.columns

        # Check values in high dimension (e.g. D=32)
        d32_rows = scorecard[scorecard["dimension"] == 32]
        assert not d32_rows.empty
        for _, row in d32_rows.iterrows():
            # PLCB spearman_dist should significantly exceed SLCB
            assert row["spearman_dist_plcb_mean"] > 0.7
            assert row["spearman_dist_slcb_mean"] < 0.1
            # Note: For N=4 (sub-strategy groups), minimum possible two-sided Wilcoxon p-value is 2/16 = 0.125.
            # For the combined group (sampling_strategy='All', N=8), p-value is < 0.05.
            if row["sampling_strategy"] == "All":
                assert row["spearman_dist_pvalue"] < 0.05
            else:
                assert row["spearman_dist_pvalue"] <= 0.125
            assert row["spearman_dist_wins"] > row["spearman_dist_losses"]


class TestReportGeneration:
    """Tests for Markdown report and Notion text generation."""

    def test_generate_markdown_report(self, mock_summaries_dir: Path):
        df = load_summary_records(mock_summaries_dir)
        scorecard = build_scorecard_dataframe(df)
        report = generate_markdown_report(df, scorecard)

        assert isinstance(report, str)
        assert len(report) > 500

        # Verify key hypotheses sections are present
        assert "Hypothesis 1" in report
        assert "SLCB" in report
        assert "Monotonicity Collapse" in report or "monotonicity" in report.lower()

        assert "Hypothesis 2" in report
        assert "PLCB" in report
        assert "Distance" in report

        assert "Hypothesis 3" in report
        assert "Coverage" in report or "Winkler" in report

        # Verify markdown table syntax
        assert "| Dimension |" in report or "| Stratum |" in report or "|" in report

    def test_generate_notion_scorecard(self, mock_summaries_dir: Path):
        df = load_summary_records(mock_summaries_dir)
        scorecard = build_scorecard_dataframe(df)
        notion_text = generate_notion_scorecard(df, scorecard)

        assert isinstance(notion_text, str)
        assert len(notion_text) > 300
        # Check Notion formatting markers
        assert "# " in notion_text
        assert "## " in notion_text
        assert "|" in notion_text


class TestCLIExecution:
    """Tests for CLI arguments, execution, and output file persistence."""

    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.summaries_dir == "results/extrapolation_uq/summaries"
        assert args.output_csv == "results/extrapolation_uq/analysis/extrapolation_calibration_scorecard.csv"
        assert args.output_report == "results/extrapolation_uq/analysis/HYPOTHESIS_EVALUATION_REPORT.md"
        assert args.output_notion == "bachelorthesis/extrapolation_uq_scorecard_notion.txt"

    def test_cli_empty_directory_graceful(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_summaries"
        empty_dir.mkdir()
        out_csv = tmp_path / "scorecard.csv"
        out_report = tmp_path / "report.md"
        out_notion = tmp_path / "notion.txt"

        exit_code = main([
            "--summaries-dir", str(empty_dir),
            "--output-csv", str(out_csv),
            "--output-report", str(out_report),
            "--output-notion", str(out_notion),
        ])
        assert exit_code == 0
        # Output files should be created (even if containing empty placeholders / headers)
        assert out_csv.is_file()
        assert out_report.is_file()
        assert out_notion.is_file()

    def test_cli_full_execution(self, mock_summaries_dir: Path, tmp_path: Path):
        out_csv = tmp_path / "analysis" / "extrapolation_calibration_scorecard.csv"
        out_report = tmp_path / "analysis" / "HYPOTHESIS_EVALUATION_REPORT.md"
        out_notion = tmp_path / "bachelorthesis" / "extrapolation_uq_scorecard_notion.txt"

        exit_code = main([
            "--summaries-dir", str(mock_summaries_dir),
            "--output-csv", str(out_csv),
            "--output-report", str(out_report),
            "--output-notion", str(out_notion),
        ])
        assert exit_code == 0

        assert out_csv.is_file()
        assert out_csv.stat().st_size > 0
        assert out_report.is_file()
        assert out_report.stat().st_size > 0
        assert out_notion.is_file()
        assert out_notion.stat().st_size > 0

        # Verify CSV can be parsed by pandas
        scorecard_df = pd.read_csv(out_csv)
        assert not scorecard_df.empty
        assert "spearman_dist_plcb_mean" in scorecard_df.columns
