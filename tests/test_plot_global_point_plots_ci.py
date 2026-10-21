import os
import sys
from pathlib import Path
import tempfile
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_global_point_plots_ci import (
    clean_approach_name,
    generate_point_plots_ci,
    APPROACH_DISPLAY_NAMES,
)


def test_clean_approach_name():
    assert clean_approach_name("prox_a_cv") == "Proximity A (CV)"
    assert clean_approach_name("prox_bc_oob") == "Proximity BC (OOB)"
    assert clean_approach_name("hutter_total") == "Variance Decomposition"
    assert clean_approach_name("hutter_between") == "Between-Tree Disagreement"
    assert clean_approach_name("shaker_epistemic") == "Shaker Epistemic"


def test_generate_point_plots_ci(tmp_path):
    data = {
        "approach": [
            "hutter_total", "shaker_epistemic",
            "prox_a_cv", "prox_a_oob",
            "prox_ac_cv", "prox_ac_oob",
            "prox_b_cv", "prox_b_oob",
            "prox_bc_cv", "prox_bc_oob"
        ],
        "ece_mean": [0.44, 0.56, 0.46, 0.45, 0.44, 0.43, 0.43, 0.42, 0.41, 0.40],
        "ece_sem": [0.001] * 10,
        "winkler_mean_mean": [86.0, 91.0, 85.0, 84.0, 83.0, 82.0, 82.5, 81.5, 81.0, 80.0],
        "winkler_mean_sem": [0.5] * 10,
        "e_aurc_mean": [2.8, 3.2, 4.6, 4.4, 4.4, 4.2, 4.2, 4.0, 3.7, 3.6],
        "e_aurc_sem": [0.02] * 10,
        "n_runs": [23040] * 10
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "main_global.csv"
    df.to_csv(csv_path, index=False)
    
    out_dir = tmp_path / "point_plots"
    generated_files = generate_point_plots_ci(
        input_csv=str(csv_path),
        output_dir=str(out_dir),
        formats=["png", "pdf"],
    )
    
    assert len(generated_files) >= 3
    assert os.path.exists(out_dir / "point_ci_ece.pdf")
    assert os.path.exists(out_dir / "point_ci_ece.png")
    assert os.path.exists(out_dir / "point_ci_winkler_mean.pdf")
    assert os.path.exists(out_dir / "point_ci_winkler_mean.png")
    assert os.path.exists(out_dir / "point_ci_e_aurc.pdf")
    assert os.path.exists(out_dir / "point_ci_e_aurc.png")
