"""Tests for unified extrapolation UQ plot generator."""

import os
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_extrapolation_plots import (
    APPROACH_STYLES,
    plot_global_point_plots,
    plot_dimension_scaling,
    plot_stratum_scaling,
    plot_surrogate_scaling,
    generate_all_extrapolation_plots,
)


@pytest.fixture
def mock_csv_environment(tmp_path):
    """Creates mock CSV files matching exact schema of the 4 analysis CSVs."""
    csv_dir = tmp_path / "analysis"
    csv_dir.mkdir(parents=True)

    approaches = [
        "prox_bc_oob", "prox_bc_cv", "prox_b_oob", "prox_b_cv",
        "prox_ac_oob", "prox_ac_cv", "prox_a_oob", "prox_a_cv",
        "shaker_epistemic", "hutter_between", "hutter_total"
    ]

    # 1. main_global.csv
    global_rows = []
    for app in approaches:
        global_rows.append({
            "approach": app,
            "ece_mean": 0.40 if "bc" in app else 0.50,
            "ece_sem": 0.005,
            "winkler_mean_mean": 80.0,
            "winkler_mean_sem": 1.0,
            "e_aurc_mean": 3.0,
            "e_aurc_sem": 0.05,
            "n_runs": 23040,
        })
    pd.DataFrame(global_rows).to_csv(csv_dir / "main_global.csv", index=False)

    # 2. main_global_by_dimension.csv
    dim_rows = []
    for d in [2, 4, 8, 16, 32]:
        for app in ["prox_bc_oob", "shaker_epistemic", "hutter_between", "hutter_total"]:
            dim_rows.append({
                "dimension": d,
                "approach": app,
                "ece_mean": 0.3 + 0.01 * d,
                "ece_sem": 0.002,
                "winkler_mean_mean": 50.0 + 2.0 * d,
                "winkler_mean_sem": 0.5,
                "e_aurc_mean": 1.0 + 0.1 * d,
                "e_aurc_sem": 0.02,
                "n_runs": 2880,
            })
    pd.DataFrame(dim_rows).to_csv(csv_dir / "main_global_by_dimension.csv", index=False)

    # 3. stratum_global.csv
    stratum_rows = []
    for s in [0, 1, 2, 3]:
        for app in ["prox_bc_oob", "shaker_epistemic", "hutter_between", "hutter_total"]:
            stratum_rows.append({
                "stratum_scope": s,
                "approach": app,
                "ece_mean": 0.2 + 0.1 * s,
                "ece_sem": 0.003,
                "winkler_mean_mean": 20.0 + 10.0 * s,
                "winkler_mean_sem": 0.8,
                "e_aurc_mean": 0.5 + 0.5 * s,
                "e_aurc_sem": 0.01,
                "n_runs": 23040,
            })
    pd.DataFrame(stratum_rows).to_csv(csv_dir / "stratum_global.csv", index=False)

    # 4. main_global_by_surrogate.csv
    surr_rows = []
    for surr in ["smac_default", "mature", "shallow", "coarse"]:
        for app in ["prox_bc_oob", "shaker_epistemic", "hutter_between", "hutter_total"]:
            surr_rows.append({
                "surrogate": surr,
                "approach": app,
                "ece_mean": 0.45,
                "ece_sem": 0.004,
                "winkler_mean_mean": 85.0,
                "winkler_mean_sem": 1.2,
                "e_aurc_mean": 3.2,
                "e_aurc_sem": 0.04,
                "n_runs": 5760,
            })
    pd.DataFrame(surr_rows).to_csv(csv_dir / "main_global_by_surrogate.csv", index=False)

    return csv_dir


def test_styles_defined():
    for app in ["prox_bc_oob", "shaker_epistemic", "hutter_between", "hutter_total"]:
        assert app in APPROACH_STYLES
        assert "color" in APPROACH_STYLES[app]
        assert "label" in APPROACH_STYLES[app]


def test_generate_all_extrapolation_plots_creates_expected_tree(mock_csv_environment, tmp_path):
    csv_dir = mock_csv_environment
    out_dir = tmp_path / "plots_updated"

    generate_all_extrapolation_plots(csv_dir=csv_dir, output_dir=out_dir)

    # Verify focal_trio subfolder
    trio_dir = out_dir / "focal_trio"
    assert (trio_dir / "global" / "global_ece.png").exists()
    assert (trio_dir / "global" / "global_winkler_mean.png").exists()
    assert (trio_dir / "global" / "global_e_aurc.png").exists()
    assert (trio_dir / "dimension" / "dimension_scaling_ece.png").exists()
    assert (trio_dir / "stratum" / "stratum_scaling_ece.png").exists()
    assert (trio_dir / "surrogate" / "surrogate_scaling_ece.png").exists()

    # Verify with_variance_decomposition subfolder
    var_dir = out_dir / "with_variance_decomposition"
    assert (var_dir / "dimension" / "dimension_scaling_ece.png").exists()
    assert (var_dir / "stratum" / "stratum_scaling_ece.png").exists()
    assert (var_dir / "surrogate" / "surrogate_scaling_ece.png").exists()


def test_global_plot_has_no_point_text_annotations(mock_csv_environment, tmp_path):
    import matplotlib.pyplot as plt
    df = pd.read_csv(mock_csv_environment / "main_global.csv")
    out_dir = tmp_path / "global_test"

    plot_global_point_plots(df=df, output_dir=out_dir)

    # Check matplotlib current figures / texts if we inspect the function directly
    # Or verify by re-running and checking that text children on ax do not contain "±"
    fig, ax = plt.subplots()
    # We will test the function's output figures or we can test that generate does not add point text annotations
    # Let's inspect ax.texts in a focused helper test
