import os
import sys
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_dimension_scaling import (
    generate_dimension_scaling_plots,
    TARGET_APPROACHES,
    APPROACH_DISPLAY_NAMES,
)


def test_target_approaches_and_names():
    assert "prox_bc_oob" in TARGET_APPROACHES
    assert "hutter_total" in TARGET_APPROACHES
    assert len(TARGET_APPROACHES) == 2
    assert APPROACH_DISPLAY_NAMES["prox_bc_oob"] == "Proximity BC (OOB)"
    assert APPROACH_DISPLAY_NAMES["hutter_total"] == "Variance Decomposition"


def test_generate_dimension_scaling_plots(tmp_path):
    dims = [2, 3, 5, 8, 16, 32, 64, 128]
    rows = []
    for d in dims:
        for app in ["hutter_total", "prox_bc_oob", "prox_a_cv"]:
            rows.append({
                "dimension": d,
                "approach": app,
                "ece_mean": 0.40 + 0.01 * (d % 5),
                "ece_sem": 0.002,
                "winkler_mean_mean": 80.0 + d,
                "winkler_mean_sem": 0.5,
                "e_aurc_mean": 2.5 + 0.1 * (d % 4),
                "e_aurc_sem": 0.03,
                "n_runs": 2880,
            })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "main_global_by_dimension.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "dimension_scaling"
    generated_files = generate_dimension_scaling_plots(
        input_csv=str(csv_path),
        output_dir=str(out_dir),
        formats=["png", "pdf"],
    )

    assert len(generated_files) >= 3
    assert os.path.exists(out_dir / "dim_scaling_ece.pdf")
    assert os.path.exists(out_dir / "dim_scaling_ece.png")
    assert os.path.exists(out_dir / "dim_scaling_winkler_mean.pdf")
    assert os.path.exists(out_dir / "dim_scaling_winkler_mean.png")
    assert os.path.exists(out_dir / "dim_scaling_e_aurc.pdf")
    assert os.path.exists(out_dir / "dim_scaling_e_aurc.png")
