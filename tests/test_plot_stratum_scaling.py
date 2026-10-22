import os
import sys
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_stratum_scaling import (
    generate_stratum_scaling_plots,
    TARGET_APPROACHES,
    APPROACHES_WITH_ENTROPY,
    APPROACH_DISPLAY_NAMES,
    STRATUM_LABELS,
)


def test_target_approaches_and_stratum_labels():
    assert "prox_bc_oob" in TARGET_APPROACHES
    assert "hutter_between" in TARGET_APPROACHES
    assert len(TARGET_APPROACHES) == 2
    assert "shaker_epistemic" in APPROACHES_WITH_ENTROPY
    assert len(APPROACHES_WITH_ENTROPY) == 3
    assert APPROACH_DISPLAY_NAMES["prox_bc_oob"] == "Proximity BC (OOB)"
    assert APPROACH_DISPLAY_NAMES["hutter_between"] == "Between-Tree Disagreement"
    assert APPROACH_DISPLAY_NAMES["shaker_epistemic"] == "Shaker Entropy"
    assert 0 in STRATUM_LABELS
    assert 1 in STRATUM_LABELS
    assert 2 in STRATUM_LABELS
    assert 3 in STRATUM_LABELS



def test_generate_stratum_scaling_plots(tmp_path):
    strata = [0, 1, 2, 3]
    rows = []
    for s in strata:
        for app in ["hutter_total", "prox_bc_oob", "shaker_epistemic"]:
            rows.append({
                "stratum_scope": s,
                "approach": app,
                "ece_mean": 0.35 + 0.05 * s,
                "ece_sem": 0.002,
                "winkler_mean_mean": 50.0 + 10.0 * s,
                "winkler_mean_sem": 0.4,
                "e_aurc_mean": 1.2 + 0.3 * s,
                "e_aurc_sem": 0.02,
                "n_runs": 23040,
            })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "stratum_global.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "stratum_scaling"
    generated_files = generate_stratum_scaling_plots(
        input_csv=str(csv_path),
        output_dir=str(out_dir),
        approaches=APPROACHES_WITH_ENTROPY,
        formats=["png", "pdf"],
    )

    assert len(generated_files) >= 6
    assert os.path.exists(out_dir / "stratum_scaling_ece.pdf")
    assert os.path.exists(out_dir / "stratum_scaling_ece.png")
    assert os.path.exists(out_dir / "stratum_scaling_winkler_mean.pdf")
    assert os.path.exists(out_dir / "stratum_scaling_winkler_mean.png")
    assert os.path.exists(out_dir / "stratum_scaling_e_aurc.pdf")
    assert os.path.exists(out_dir / "stratum_scaling_e_aurc.png")


def test_generate_stratum_scaling_plots_with_surrogate(tmp_path):
    strata = [0, 1, 2, 3]
    surrogates = ["smac_default", "coarse"]
    rows = []
    for s in surrogates:
        for st in strata:
            for app in ["hutter_total", "prox_bc_oob", "shaker_epistemic"]:
                rows.append({
                    "surrogate": s,
                    "stratum_scope": st,
                    "approach": app,
                    "ece_mean": 0.30 if s == "smac_default" else 0.45,
                    "ece_sem": 0.002,
                    "winkler_mean_mean": 60.0 if s == "smac_default" else 80.0,
                    "winkler_mean_sem": 0.5,
                    "e_aurc_mean": 1.5 if s == "smac_default" else 2.5,
                    "e_aurc_sem": 0.03,
                    "n_runs": 5760,
                })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "stratum_by_surrogate.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "stratum_scaling_smac3"
    generated_files = generate_stratum_scaling_plots(
        input_csv=str(csv_path),
        output_dir=str(out_dir),
        surrogate="smac_default",
        approaches=APPROACHES_WITH_ENTROPY,
        formats=["png", "pdf"],
    )

    assert len(generated_files) == 6
    assert os.path.exists(out_dir / "stratum_scaling_ece.pdf")
    assert os.path.exists(out_dir / "stratum_scaling_ece.png")

