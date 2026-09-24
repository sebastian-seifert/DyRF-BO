"""Unit and integration tests for thesis tables and figures generation pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_thesis_tables_and_figures import (
    build_parser,
    generate_all_artifacts,
    generate_table_1_executive_scorecard,
    generate_table_2_dimension_scaling,
    generate_table_3_strata_breakdown,
    generate_table_4_objective_breakdown,
    generate_table_5_dimension_strata_matrix,
    main,
    plot_figure_1_dimension_monotonicity_scaling,
    plot_figure_2_chebyshev_norm_advantage_bar,
    plot_figure_3_strata_calibration_heatmap,
    plot_figure_4_highdim_bbob_paradox_dual_panel,
    plot_figure_5_orthogonal_horizon_concept,
)


@pytest.fixture
def mock_analysis_dir(tmp_path: Path) -> Path:
    """Fixture creating realistic synthetic scorecards for table and figure testing."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    dims = [2, 3, 5, 8, 16, 32]
    strats = ["natural", "stratified"]

    # 1. Calibration scorecard
    cal_rows = []
    for d in dims:
        for strat in strats:
            cal_rows.append({
                "dimension": d,
                "sampling_strategy": strat,
                "n_experiments": 160,
                "spearman_dist_plcb_mean": 0.85 - 0.002 * d,
                "spearman_dist_plcb_sem": 0.01,
                "spearman_dist_slcb_mean": 0.30 - 0.015 * d,
                "spearman_dist_slcb_sem": 0.02,
                "spearman_dist_diff_mean": (0.85 - 0.002 * d) - (0.30 - 0.015 * d),
                "spearman_dist_pvalue": 1e-15,
                "spearman_dist_cliffs_delta": 0.88,
                "spearman_dist_wins": 150,
                "spearman_dist_ties": 0,
                "spearman_dist_losses": 10,
                "spearman_err_plcb_mean": 0.65,
                "spearman_err_plcb_sem": 0.01,
                "spearman_err_slcb_mean": 0.25,
                "spearman_err_slcb_sem": 0.02,
                "spearman_err_diff_mean": 0.40,
                "spearman_err_pvalue": 1e-12,
                "spearman_err_cliffs_delta": 0.75,
                "spearman_err_wins": 140,
                "spearman_err_ties": 0,
                "spearman_err_losses": 20,
                "picp_error_plcb_mean": 0.03,
                "picp_error_plcb_sem": 0.005,
                "picp_error_slcb_mean": 0.35,
                "picp_error_slcb_sem": 0.015,
                "picp_error_diff_mean": -0.32,
                "picp_error_pvalue": 1e-20,
                "picp_error_cliffs_delta": -0.85,
                "picp_error_wins": 155,
                "picp_error_ties": 0,
                "picp_error_losses": 5,
                "winkler_plcb_mean": 25.0 * (1 + 0.1 * d),
                "winkler_plcb_sem": 2.0,
                "winkler_slcb_mean": 75.0 * (1 + 0.15 * d),
                "winkler_slcb_sem": 5.0,
                "winkler_diff_mean": -50.0,
                "winkler_pvalue": 1e-25,
                "winkler_cliffs_delta": -0.90,
                "winkler_wins": 158,
                "winkler_ties": 0,
                "winkler_losses": 2,
                "outlier_auroc_plcb_mean": 0.88,
                "outlier_auroc_plcb_sem": 0.01,
                "outlier_auroc_slcb_mean": 0.55,
                "outlier_auroc_slcb_sem": 0.02,
                "outlier_auroc_diff_mean": 0.33,
                "outlier_auroc_pvalue": 1e-18,
                "outlier_auroc_cliffs_delta": 0.80,
                "outlier_auroc_wins": 145,
                "outlier_auroc_ties": 0,
                "outlier_auroc_losses": 15,
            })
    # Add marginal and All rows
    cal_rows.append({
        "dimension": "All",
        "sampling_strategy": "All",
        "n_experiments": 1920,
        "spearman_dist_plcb_mean": 0.82,
        "spearman_dist_plcb_sem": 0.005,
        "spearman_dist_slcb_mean": 0.12,
        "spearman_dist_slcb_sem": 0.008,
        "spearman_dist_diff_mean": 0.70,
        "spearman_dist_pvalue": 1e-200,
        "spearman_dist_cliffs_delta": 0.85,
        "spearman_dist_wins": 1800,
        "spearman_dist_ties": 0,
        "spearman_dist_losses": 120,
        "spearman_err_plcb_mean": 0.60,
        "spearman_err_plcb_sem": 0.006,
        "spearman_err_slcb_mean": 0.20,
        "spearman_err_slcb_sem": 0.007,
        "spearman_err_diff_mean": 0.40,
        "spearman_err_pvalue": 1e-150,
        "spearman_err_cliffs_delta": 0.72,
        "spearman_err_wins": 1700,
        "spearman_err_ties": 0,
        "spearman_err_losses": 220,
        "picp_error_plcb_mean": 0.04,
        "picp_error_plcb_sem": 0.002,
        "picp_error_slcb_mean": 0.38,
        "picp_error_slcb_sem": 0.005,
        "picp_error_diff_mean": -0.34,
        "picp_error_pvalue": 1e-250,
        "picp_error_cliffs_delta": -0.88,
        "picp_error_wins": 1850,
        "picp_error_ties": 0,
        "picp_error_losses": 70,
        "winkler_plcb_mean": 45.0,
        "winkler_plcb_sem": 1.5,
        "winkler_slcb_mean": 120.0,
        "winkler_slcb_sem": 3.0,
        "winkler_diff_mean": -75.0,
        "winkler_pvalue": 1e-220,
        "winkler_cliffs_delta": -0.87,
        "winkler_wins": 1880,
        "winkler_ties": 0,
        "winkler_losses": 40,
        "outlier_auroc_plcb_mean": 0.85,
        "outlier_auroc_plcb_sem": 0.004,
        "outlier_auroc_slcb_mean": 0.52,
        "outlier_auroc_slcb_sem": 0.006,
        "outlier_auroc_diff_mean": 0.33,
        "outlier_auroc_pvalue": 1e-180,
        "outlier_auroc_cliffs_delta": 0.78,
        "outlier_auroc_wins": 1750,
        "outlier_auroc_ties": 0,
        "outlier_auroc_losses": 170,
    })
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(analysis_dir / "extrapolation_calibration_scorecard.csv", index=False)

    # 2. Distance ablation scorecard
    dist_rows = []
    for d in dims:
        for strat in strats:
            norm_slcb = 0.30 - 0.015 * d
            inf_slcb = norm_slcb + 0.05
            norm_plcb = 0.85 - 0.002 * d
            inf_plcb = norm_plcb + 0.03
            dist_rows.append({
                "dimension": d,
                "sampling_strategy": strat,
                "n_experiments": 160,
                "dist_norm_slcb_mean": norm_slcb,
                "dist_norm_slcb_sem": 0.02,
                "dist_inf_slcb_mean": inf_slcb,
                "dist_inf_slcb_sem": 0.02,
                "diff_slcb_mean": inf_slcb - norm_slcb,
                "pvalue_slcb": 1e-5,
                "cliffs_delta_slcb": 0.25,
                "wins_slcb": 110,
                "ties_slcb": 0,
                "losses_slcb": 50,
                "dist_norm_plcb_mean": norm_plcb,
                "dist_norm_plcb_sem": 0.01,
                "dist_inf_plcb_mean": inf_plcb,
                "dist_inf_plcb_sem": 0.01,
                "diff_plcb_mean": inf_plcb - norm_plcb,
                "pvalue_plcb": 1e-8,
                "cliffs_delta_plcb": 0.30,
                "wins_plcb": 125,
                "ties_plcb": 0,
                "losses_plcb": 35,
            })
    for d in dims:
        dist_rows.append({
            "dimension": d,
            "sampling_strategy": "All",
            "n_experiments": 320,
            "dist_norm_slcb_mean": 0.25 - 0.01 * d,
            "dist_norm_slcb_sem": 0.015,
            "dist_inf_slcb_mean": 0.30 - 0.01 * d,
            "dist_inf_slcb_sem": 0.015,
            "diff_slcb_mean": 0.05,
            "pvalue_slcb": 1e-10,
            "cliffs_delta_slcb": 0.28,
            "wins_slcb": 220,
            "ties_slcb": 0,
            "losses_slcb": 100,
            "dist_norm_plcb_mean": 0.84 - 0.002 * d,
            "dist_norm_plcb_sem": 0.008,
            "dist_inf_plcb_mean": 0.87 - 0.002 * d,
            "dist_inf_plcb_sem": 0.008,
            "diff_plcb_mean": 0.03,
            "pvalue_plcb": 1e-15,
            "cliffs_delta_plcb": 0.32,
            "wins_plcb": 250,
            "ties_plcb": 0,
            "losses_plcb": 70,
        })
    dist_rows.append({
        "dimension": "All",
        "sampling_strategy": "All",
        "n_experiments": 1920,
        "dist_norm_slcb_mean": 0.12,
        "dist_norm_slcb_sem": 0.008,
        "dist_inf_slcb_mean": 0.16,
        "dist_inf_slcb_sem": 0.008,
        "diff_slcb_mean": 0.04,
        "pvalue_slcb": 1e-25,
        "cliffs_delta_slcb": 0.26,
        "wins_slcb": 1300,
        "ties_slcb": 0,
        "losses_slcb": 620,
        "dist_norm_plcb_mean": 0.82,
        "dist_norm_plcb_sem": 0.005,
        "dist_inf_plcb_mean": 0.85,
        "dist_inf_plcb_sem": 0.005,
        "diff_plcb_mean": 0.03,
        "pvalue_plcb": 1e-40,
        "cliffs_delta_plcb": 0.31,
        "wins_plcb": 1500,
        "ties_plcb": 0,
        "losses_plcb": 420,
    })
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(analysis_dir / "distance_ablation_scorecard.csv", index=False)

    # 3. Objective scorecard
    obj_rows = []
    funcs = ["sphere", "rosenbrock", "rastrigin", "ackley"]
    for fn in funcs:
        obj_rows.append({
            "function_name": fn,
            "n_experiments": 480,
            "spearman_dist_plcb_mean": 0.81,
            "spearman_dist_plcb_sem": 0.01,
            "spearman_dist_slcb_mean": 0.14,
            "spearman_dist_slcb_sem": 0.015,
            "spearman_dist_diff_mean": 0.67,
            "spearman_dist_pvalue": 1e-50,
            "spearman_dist_cliffs_delta": 0.82,
            "spearman_dist_wins": 450,
            "spearman_dist_ties": 0,
            "spearman_dist_losses": 30,
            "spearman_err_plcb_mean": 0.58,
            "spearman_err_plcb_sem": 0.01,
            "spearman_err_slcb_mean": 0.22,
            "spearman_err_slcb_sem": 0.01,
            "spearman_err_diff_mean": 0.36,
            "spearman_err_pvalue": 1e-40,
            "spearman_err_cliffs_delta": 0.70,
            "spearman_err_wins": 420,
            "spearman_err_ties": 0,
            "spearman_err_losses": 60,
            "picp_error_plcb_mean": 0.04,
            "picp_error_plcb_sem": 0.003,
            "picp_error_slcb_mean": 0.36,
            "picp_error_slcb_sem": 0.008,
            "picp_error_diff_mean": -0.32,
            "picp_error_pvalue": 1e-60,
            "picp_error_cliffs_delta": -0.86,
            "picp_error_wins": 460,
            "picp_error_ties": 0,
            "picp_error_losses": 20,
            "winkler_plcb_mean": 50.0,
            "winkler_plcb_sem": 2.5,
            "winkler_slcb_mean": 130.0,
            "winkler_slcb_sem": 5.0,
            "winkler_diff_mean": -80.0,
            "winkler_pvalue": 1e-55,
            "winkler_cliffs_delta": -0.85,
            "winkler_wins": 470,
            "winkler_ties": 0,
            "winkler_losses": 10,
            "outlier_auroc_plcb_mean": 0.86,
            "outlier_auroc_plcb_sem": 0.008,
            "outlier_auroc_slcb_mean": 0.53,
            "outlier_auroc_slcb_sem": 0.012,
            "outlier_auroc_diff_mean": 0.33,
            "outlier_auroc_pvalue": 1e-45,
            "outlier_auroc_cliffs_delta": 0.79,
            "outlier_auroc_wins": 440,
            "outlier_auroc_ties": 0,
            "outlier_auroc_losses": 40,
        })
    obj_rows.append({
        "function_name": "All",
        "n_experiments": 1920,
        "spearman_dist_plcb_mean": 0.82,
        "spearman_dist_plcb_sem": 0.005,
        "spearman_dist_slcb_mean": 0.12,
        "spearman_dist_slcb_sem": 0.008,
        "spearman_dist_diff_mean": 0.70,
        "spearman_dist_pvalue": 1e-200,
        "spearman_dist_cliffs_delta": 0.85,
        "spearman_dist_wins": 1800,
        "spearman_dist_ties": 0,
        "spearman_dist_losses": 120,
        "spearman_err_plcb_mean": 0.60,
        "spearman_err_plcb_sem": 0.006,
        "spearman_err_slcb_mean": 0.20,
        "spearman_err_slcb_sem": 0.007,
        "spearman_err_diff_mean": 0.40,
        "spearman_err_pvalue": 1e-150,
        "spearman_err_cliffs_delta": 0.72,
        "spearman_err_wins": 1700,
        "spearman_err_ties": 0,
        "spearman_err_losses": 220,
        "picp_error_plcb_mean": 0.04,
        "picp_error_plcb_sem": 0.002,
        "picp_error_slcb_mean": 0.38,
        "picp_error_slcb_sem": 0.005,
        "picp_error_diff_mean": -0.34,
        "picp_error_pvalue": 1e-250,
        "picp_error_cliffs_delta": -0.88,
        "picp_error_wins": 1850,
        "picp_error_ties": 0,
        "picp_error_losses": 70,
        "winkler_plcb_mean": 45.0,
        "winkler_plcb_sem": 1.5,
        "winkler_slcb_mean": 120.0,
        "winkler_slcb_sem": 3.0,
        "winkler_diff_mean": -75.0,
        "winkler_pvalue": 1e-220,
        "winkler_cliffs_delta": -0.87,
        "winkler_wins": 1880,
        "winkler_ties": 0,
        "winkler_losses": 40,
        "outlier_auroc_plcb_mean": 0.85,
        "outlier_auroc_plcb_sem": 0.004,
        "outlier_auroc_slcb_mean": 0.52,
        "outlier_auroc_slcb_sem": 0.006,
        "outlier_auroc_diff_mean": 0.33,
        "outlier_auroc_pvalue": 1e-180,
        "outlier_auroc_cliffs_delta": 0.78,
        "outlier_auroc_wins": 1750,
        "outlier_auroc_ties": 0,
        "outlier_auroc_losses": 170,
    })
    obj_df = pd.DataFrame(obj_rows)
    obj_df.to_csv(analysis_dir / "extrapolation_objective_scorecard.csv", index=False)

    # 4. Dimension x Strata matrix
    mat_rows = []
    all_dims = dims + ["All"]
    all_strata = [0, 1, 2, 3, "All"]

    for d in all_dims:
        for s in all_strata:
            w_p = 5.0 * (s + 1 if isinstance(s, int) else 2.5) * (1 + 0.1 * (d if isinstance(d, int) else 8))
            w_s = 15.0 * (s + 1 if isinstance(s, int) else 2.5) * (1 + 0.15 * (d if isinstance(d, int) else 8))
            w_rat = np.log(w_p / w_s)
            mat_rows.append({
                "dimension": d,
                "stratum": s,
                "n_experiments": 320 if d != "All" else 1920,
                "winkler_plcb_mean": w_p,
                "winkler_plcb_sem": 0.5,
                "winkler_slcb_mean": w_s,
                "winkler_slcb_sem": 1.2,
                "winkler_ratio": w_rat,
                "picp_plcb_mean": 0.94,
                "picp_plcb_sem": 0.005,
                "picp_slcb_mean": 0.60,
                "picp_slcb_sem": 0.01,
                "auroc_plcb_mean": 0.85,
                "auroc_plcb_sem": 0.008,
                "auroc_slcb_mean": 0.52,
                "auroc_slcb_sem": 0.01,
            })
    mat_df = pd.DataFrame(mat_rows)
    mat_df.to_csv(analysis_dir / "extrapolation_dimension_strata_matrix.csv", index=False)

    return analysis_dir


class TestTableGeneration:
    """Tests for generating publication-grade Markdown and LaTeX tables."""

    def test_table_1_executive_scorecard(self, mock_analysis_dir: Path):
        cal_df = pd.read_csv(mock_analysis_dir / "extrapolation_calibration_scorecard.csv")
        md_text, tex_text = generate_table_1_executive_scorecard(cal_df)

        assert isinstance(md_text, str)
        assert isinstance(tex_text, str)
        assert "| Core Metric |" in md_text
        assert "Distance Monotonicity" in md_text
        assert "Winkler" in md_text

        # LaTeX assertions
        assert "\\toprule" in tex_text
        assert "\\bottomrule" in tex_text
        assert "\\begin{table}" in tex_text or "\\begin{tabular}" in tex_text
        assert "\\end{tabular}" in tex_text

    def test_table_2_dimension_scaling(self, mock_analysis_dir: Path):
        dist_df = pd.read_csv(mock_analysis_dir / "distance_ablation_scorecard.csv")
        cal_df = pd.read_csv(mock_analysis_dir / "extrapolation_calibration_scorecard.csv")
        md_text, tex_text = generate_table_2_dimension_scaling(dist_df, cal_df)

        assert "| D=" in md_text or "| 2 |" in md_text
        assert "Chebyshev" in md_text or "L_\\infty" in md_text or "L_inf" in md_text
        assert "\\toprule" in tex_text
        assert "\\bottomrule" in tex_text

    def test_table_3_strata_breakdown(self, mock_analysis_dir: Path):
        mat_df = pd.read_csv(mock_analysis_dir / "extrapolation_dimension_strata_matrix.csv")
        md_text, tex_text = generate_table_3_strata_breakdown(mat_df)

        assert "Stratum 0" in md_text or "Stratum" in md_text
        assert "\\toprule" in tex_text
        assert "\\bottomrule" in tex_text

    def test_table_4_objective_breakdown(self, mock_analysis_dir: Path):
        obj_df = pd.read_csv(mock_analysis_dir / "extrapolation_objective_scorecard.csv")
        md_text, tex_text = generate_table_4_objective_breakdown(obj_df)

        assert "Sphere" in md_text or "sphere" in md_text
        assert "Rosenbrock" in md_text or "rosenbrock" in md_text
        assert "\\toprule" in tex_text
        assert "\\bottomrule" in tex_text

    def test_table_5_dimension_strata_matrix(self, mock_analysis_dir: Path):
        mat_df = pd.read_csv(mock_analysis_dir / "extrapolation_dimension_strata_matrix.csv")
        md_text, tex_text = generate_table_5_dimension_strata_matrix(mat_df)

        assert "Winkler" in md_text
        assert "\\toprule" in tex_text
        assert "\\bottomrule" in tex_text


class TestFigureGeneration:
    """Tests for generating publication-grade PNG and PDF figures."""

    def test_figure_1_dimension_monotonicity_scaling(self, mock_analysis_dir: Path, tmp_path: Path):
        dist_df = pd.read_csv(mock_analysis_dir / "distance_ablation_scorecard.csv")
        out_prefix = tmp_path / "figure_1_test"
        plot_figure_1_dimension_monotonicity_scaling(dist_df, out_prefix)

        png_file = Path(f"{out_prefix}.png")
        pdf_file = Path(f"{out_prefix}.pdf")
        assert png_file.is_file() and png_file.stat().st_size > 0
        assert pdf_file.is_file() and pdf_file.stat().st_size > 0

    def test_figure_2_chebyshev_norm_advantage_bar(self, mock_analysis_dir: Path, tmp_path: Path):
        dist_df = pd.read_csv(mock_analysis_dir / "distance_ablation_scorecard.csv")
        out_prefix = tmp_path / "figure_2_test"
        plot_figure_2_chebyshev_norm_advantage_bar(dist_df, out_prefix)

        png_file = Path(f"{out_prefix}.png")
        pdf_file = Path(f"{out_prefix}.pdf")
        assert png_file.is_file() and png_file.stat().st_size > 0
        assert pdf_file.is_file() and pdf_file.stat().st_size > 0

    def test_figure_3_strata_calibration_heatmap(self, mock_analysis_dir: Path, tmp_path: Path):
        mat_df = pd.read_csv(mock_analysis_dir / "extrapolation_dimension_strata_matrix.csv")
        out_prefix = tmp_path / "figure_3_test"
        plot_figure_3_strata_calibration_heatmap(mat_df, out_prefix)

        png_file = Path(f"{out_prefix}.png")
        pdf_file = Path(f"{out_prefix}.pdf")
        assert png_file.is_file() and png_file.stat().st_size > 0
        assert pdf_file.is_file() and pdf_file.stat().st_size > 0

    def test_figure_4_highdim_bbob_paradox_dual_panel(self, mock_analysis_dir: Path, tmp_path: Path):
        cal_df = pd.read_csv(mock_analysis_dir / "extrapolation_calibration_scorecard.csv")
        out_prefix = tmp_path / "figure_4_test"
        plot_figure_4_highdim_bbob_paradox_dual_panel(cal_df, out_prefix)

        png_file = Path(f"{out_prefix}.png")
        pdf_file = Path(f"{out_prefix}.pdf")
        assert png_file.is_file() and png_file.stat().st_size > 0
        assert pdf_file.is_file() and pdf_file.stat().st_size > 0

    def test_figure_5_orthogonal_horizon_concept(self, tmp_path: Path):
        out_prefix = tmp_path / "figure_5_test"
        plot_figure_5_orthogonal_horizon_concept(out_prefix)

        png_file = Path(f"{out_prefix}.png")
        pdf_file = Path(f"{out_prefix}.pdf")
        assert png_file.is_file() and png_file.stat().st_size > 0
        assert pdf_file.is_file() and pdf_file.stat().st_size > 0


class TestPipelineAndCLI:
    """Tests for pipeline execution and CLI entrypoint."""

    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.analysis_dir == "results/extrapolation_uq/analysis"
        assert args.output_dir == "results/extrapolation_uq/thesis_artifacts"

    def test_generate_all_artifacts_end_to_end(self, mock_analysis_dir: Path, tmp_path: Path):
        out_artifacts = tmp_path / "thesis_artifacts"

        exit_code = generate_all_artifacts(
            analysis_dir=mock_analysis_dir,
            output_dir=out_artifacts,
        )
        assert exit_code == 0

        tables_dir = out_artifacts / "tables"
        figures_dir = out_artifacts / "figures"
        assert tables_dir.is_dir()
        assert figures_dir.is_dir()

        # Check all 5 tables exist in both .md and .tex (10 files)
        for i in range(1, 6):
            md_files = list(tables_dir.glob(f"table_{i}_*.md"))
            tex_files = list(tables_dir.glob(f"table_{i}_*.tex"))
            assert len(md_files) == 1, f"Missing table_{i} .md file"
            assert len(tex_files) == 1, f"Missing table_{i} .tex file"
            assert md_files[0].stat().st_size > 0
            assert tex_files[0].stat().st_size > 0

        # Check all 5 figures exist in both .png and .pdf (10 files)
        for i in range(1, 6):
            png_files = list(figures_dir.glob(f"figure_{i}_*.png"))
            pdf_files = list(figures_dir.glob(f"figure_{i}_*.pdf"))
            assert len(png_files) == 1, f"Missing figure_{i} .png file"
            assert len(pdf_files) == 1, f"Missing figure_{i} .pdf file"
            assert png_files[0].stat().st_size > 0
            assert pdf_files[0].stat().st_size > 0

    def test_cli_invocation(self, mock_analysis_dir: Path, tmp_path: Path):
        out_artifacts = tmp_path / "cli_artifacts"
        argv = [
            "--analysis-dir", str(mock_analysis_dir),
            "--output-dir", str(out_artifacts),
        ]
        exit_code = main(argv)
        assert exit_code == 0
        assert (out_artifacts / "tables").is_dir()
        assert (out_artifacts / "figures").is_dir()
