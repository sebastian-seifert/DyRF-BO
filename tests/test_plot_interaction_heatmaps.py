import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_interaction_heatmaps import (
    compute_relative_gain_matrix,
    plot_single_heatmap,
    generate_all_interaction_heatmaps,
)


def test_compute_relative_gain_matrix():
    # Synthetic dataframe with 2 strata and 2 dimensions
    df = pd.DataFrame([
        {"stratum_scope": 0, "dimension": 2, "approach": "hutter_total", "ece_mean": 0.40},
        {"stratum_scope": 0, "dimension": 2, "approach": "prox_bc_oob", "ece_mean": 0.30},  # +25%
        {"stratum_scope": 1, "dimension": 2, "approach": "hutter_total", "ece_mean": 0.20},
        {"stratum_scope": 1, "dimension": 2, "approach": "prox_bc_oob", "ece_mean": 0.22},  # -10%
        {"stratum_scope": 0, "dimension": 8, "approach": "hutter_total", "ece_mean": 0.50},
        {"stratum_scope": 0, "dimension": 8, "approach": "prox_bc_oob", "ece_mean": 0.45},  # +10%
        {"stratum_scope": 1, "dimension": 8, "approach": "hutter_total", "ece_mean": 0.30},
        {"stratum_scope": 1, "dimension": 8, "approach": "prox_bc_oob", "ece_mean": 0.30},  # 0%
    ])

    matrix = compute_relative_gain_matrix(
        df=df,
        row_col="stratum_scope",
        col_col="dimension",
        metric_col="ece_mean",
        baseline_approach="hutter_total",
        target_approach="prox_bc_oob",
    )

    assert matrix.shape == (2, 2)
    assert np.isclose(matrix.loc[0, 2], 25.0)
    assert np.isclose(matrix.loc[1, 2], -10.0)
    assert np.isclose(matrix.loc[0, 8], 10.0)
    assert np.isclose(matrix.loc[1, 8], 0.0)


def test_generate_all_interaction_heatmaps(tmp_path):
    # Create synthetic slice CSVs
    strata = [0, 1, 2, 3]
    dims = [2, 8, 32]
    surrogates = ["smac_default", "mature", "shallow", "coarse"]

    # 1. Stratum x Dimension
    rows_sd = []
    for s in strata:
        for d in dims:
            for app in ["hutter_total", "prox_bc_oob"]:
                rows_sd.append({
                    "stratum_scope": s,
                    "dimension": d,
                    "approach": app,
                    "ece_mean": 0.4 if app == "hutter_total" else 0.35,
                    "winkler_mean_mean": 80.0 if app == "hutter_total" else 75.0,
                    "e_aurc_mean": 2.5 if app == "hutter_total" else 2.3,
                })
    csv_sd = tmp_path / "stratum_by_dimension.csv"
    pd.DataFrame(rows_sd).to_csv(csv_sd, index=False)

    # 2. Stratum x Surrogate
    rows_ss = []
    for s in strata:
        for rf in surrogates:
            for app in ["hutter_total", "prox_bc_oob"]:
                rows_ss.append({
                    "stratum_scope": s,
                    "surrogate": rf,
                    "approach": app,
                    "ece_mean": 0.4 if app == "hutter_total" else 0.35,
                    "winkler_mean_mean": 80.0 if app == "hutter_total" else 75.0,
                    "e_aurc_mean": 2.5 if app == "hutter_total" else 2.3,
                })
    csv_ss = tmp_path / "stratum_by_surrogate.csv"
    pd.DataFrame(rows_ss).to_csv(csv_ss, index=False)

    # 3. Surrogate x Dimension
    rows_rd = []
    for rf in surrogates:
        for d in dims:
            for app in ["hutter_total", "prox_bc_oob"]:
                rows_rd.append({
                    "surrogate": rf,
                    "dimension": d,
                    "approach": app,
                    "ece_mean": 0.4 if app == "hutter_total" else 0.35,
                    "winkler_mean_mean": 80.0 if app == "hutter_total" else 75.0,
                    "e_aurc_mean": 2.5 if app == "hutter_total" else 2.3,
                })
    csv_rd = tmp_path / "main_global_rf_by_dimension.csv"
    pd.DataFrame(rows_rd).to_csv(csv_rd, index=False)

    out_dir = tmp_path / "heatmaps"
    generated_files = generate_all_interaction_heatmaps(
        stratum_dim_csv=str(csv_sd),
        stratum_surrogate_csv=str(csv_ss),
        surrogate_dim_csv=str(csv_rd),
        output_dir=str(out_dir),
        formats=["png", "pdf"],
    )

    # 9 heatmaps * 2 formats = 18 files
    assert len(generated_files) == 18
    assert os.path.exists(out_dir / "stratum_by_dimension" / "heatmap_stratum_dim_ece.pdf")
    assert os.path.exists(out_dir / "stratum_by_dimension" / "heatmap_stratum_dim_ece.png")
    assert os.path.exists(out_dir / "stratum_by_surrogate" / "heatmap_stratum_surrogate_ece.png")
    assert os.path.exists(out_dir / "surrogate_by_dimension" / "heatmap_surrogate_dim_ece.png")
