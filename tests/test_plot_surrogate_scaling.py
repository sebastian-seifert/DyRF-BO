import os
import sys
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_surrogate_scaling import (
    generate_surrogate_scaling_plots,
    TARGET_APPROACHES,
    APPROACHES_WITH_ENTROPY,
    APPROACH_DISPLAY_NAMES,
    SURROGATE_DISPLAY_NAMES,
)


def test_target_approaches_and_surrogate_names():
    assert "prox_bc_oob" in TARGET_APPROACHES
    assert "hutter_between" in TARGET_APPROACHES
    assert "shaker_epistemic" in TARGET_APPROACHES
    assert len(TARGET_APPROACHES) == 3
    assert "shaker_epistemic" in APPROACHES_WITH_ENTROPY
    assert len(APPROACHES_WITH_ENTROPY) == 3
    assert APPROACH_DISPLAY_NAMES["prox_bc_oob"] == "Proximity BC (OOB)"
    assert APPROACH_DISPLAY_NAMES["hutter_between"] == "Between-Tree Disagreement"
    assert APPROACH_DISPLAY_NAMES["shaker_epistemic"] == "Shaker Entropy"
    assert "smac_default" in SURROGATE_DISPLAY_NAMES
    assert "mature" in SURROGATE_DISPLAY_NAMES
    assert "shallow" in SURROGATE_DISPLAY_NAMES
    assert "coarse" in SURROGATE_DISPLAY_NAMES


def test_generate_surrogate_scaling_plots(tmp_path):
    surrogates = ["smac_default", "mature", "shallow", "coarse"]
    rows = []
    for s in surrogates:
        for app in ["hutter_total", "prox_bc_oob", "shaker_epistemic"]:
            rows.append({
                "surrogate": s,
                "approach": app,
                "ece_mean": 0.40,
                "ece_sem": 0.003,
                "winkler_mean_mean": 85.0,
                "winkler_mean_sem": 1.5,
                "e_aurc_mean": 3.0,
                "e_aurc_sem": 0.05,
                "n_runs": 5760,
            })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "main_global_by_surrogate.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "surrogate_scaling"
    generated_files = generate_surrogate_scaling_plots(
        input_csv=str(csv_path),
        output_dir=str(out_dir),
        approaches=APPROACHES_WITH_ENTROPY,
        formats=["png", "pdf"],
    )

    assert len(generated_files) >= 6
    assert os.path.exists(out_dir / "surrogate_ece.pdf")
    assert os.path.exists(out_dir / "surrogate_ece.png")
    assert os.path.exists(out_dir / "surrogate_winkler_mean.pdf")
    assert os.path.exists(out_dir / "surrogate_winkler_mean.png")
    assert os.path.exists(out_dir / "surrogate_e_aurc.pdf")
    assert os.path.exists(out_dir / "surrogate_e_aurc.png")

