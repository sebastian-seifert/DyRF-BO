#!/usr/bin/env python3
"""Tests for NeurIPS-grade BBOB timing scaling visualization generator."""

import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.plot_bbob_timing_scaling import (
    APPROACH_STYLES,
    FAMILY_ORDER,
    generate_timing_scaling_figures,
)


def _make_dummy_timing_df() -> pd.DataFrame:
    rows = []
    dims = [2, 4, 8, 16, 32]
    for app in APPROACH_STYLES.keys():
        for d in dims:
            base = 200.0 + d * 4.0 if "Entropy" in app else (30.0 + d * 2.0)
            rows.append(
                {
                    "approach": app,
                    "dimension": d,
                    "count": 20.0,
                    "mean_wallclock": base,
                    "std_wallclock": 5.0,
                    "median_wallclock": base - 1.0,
                    "iqr_wallclock": 2.0,
                    "relative_slowdown": 1.0 if app == "SMAC3_HPOFacade_lcb" else base / (20.0 + d * 1.5),
                }
            )
    return pd.DataFrame(rows)


def test_generate_timing_scaling_figures_synthetic(tmp_path):
    """Verify generate_timing_scaling_figures creates all 4 publication figures (PDF & PNG) from DataFrame."""
    df = _make_dummy_timing_df()
    out_dir = tmp_path / "figs"
    outputs = generate_timing_scaling_figures(df, output_dir=str(out_dir))

    expected_files = [
        "timing_scaling_mean_std.pdf",
        "timing_scaling_mean_std.png",
        "timing_scaling_by_family.pdf",
        "timing_scaling_by_family.png",
    ]
    for fname in expected_files:
        fpath = out_dir / fname
        assert fpath.exists(), f"Expected figure {fname} was not created"
        assert fpath.stat().st_size > 1024, f"Expected figure {fname} is empty or too small"

    assert len(FAMILY_ORDER) == 4
    assert len(APPROACH_STYLES) == 11
    assert set(outputs.keys()) == {"mean_std_pdf", "mean_std_png", "family_pdf", "family_png"}


def test_family_plot_no_relative_overhead_sd_bands():
    """Verify that relative overhead plots do not contain error bars or fill_between bands."""
    import matplotlib.pyplot as plt
    from scripts.plot_bbob_timing_scaling import plot_family_scaling

    df = _make_dummy_timing_df()
    fig, (ax_a, ax_b) = plt.subplots(1, 2)
    plot_family_scaling(df, ax_a, ax_b)

    # Panel a should have fill_between bands (collections / PolyCollection)
    assert len(ax_a.collections) > 0, "Panel a should have shaded uncertainty ribbons"

    # Panel b (relative overhead) should have NO fill_between bands
    assert len(ax_b.collections) == 0, "Panel b should NOT have shaded uncertainty ribbons"

    # Panel b should only have plain lines, not errorbars with line caps/bars
    for line in ax_b.lines:
        assert line.get_label() is not None
    plt.close(fig)


def test_family_plot_excludes_ei_and_uses_lcb_only():
    """Verify that family plot strictly uses Baseline (LCB) and excludes EI."""
    import matplotlib.pyplot as plt
    from scripts.plot_bbob_timing_scaling import (
        FAMILY_ORDER,
        FAMILY_STYLES,
        plot_family_scaling,
    )

    assert "Baseline (LCB)" in FAMILY_ORDER or "Baseline LCB" in FAMILY_ORDER
    assert "Baselines (LCB / EI)" not in FAMILY_ORDER

    df = _make_dummy_timing_df()
    fig, (ax_a, ax_b) = plt.subplots(1, 2)
    plot_family_scaling(df, ax_a, ax_b)

    labels_a = [l.get_label() for l in ax_a.lines if l.get_label() and not l.get_label().startswith("_")]
    labels_b = [l.get_label() for l in ax_b.lines if l.get_label() and not l.get_label().startswith("_")]

    # Check that EI is not in the labels
    for lbl in labels_a + labels_b:
        assert "EI" not in lbl, f"EI found in family plot labels: {lbl}"

    # Check that Baseline LCB is present
    baseline_lbl = [lbl for lbl in labels_b if "Baseline" in lbl][0]
    assert "LCB" in baseline_lbl

    # Check that relative overhead of Baseline LCB in Panel B is identically 1.0
    for line in ax_b.lines:
        if line.get_label() == baseline_lbl:
            ydata = line.get_ydata()
            assert all(abs(y - 1.0) < 1e-5 for y in ydata), f"Baseline LCB relative overhead should be 1.0, got {ydata}"

    plt.close(fig)




def test_plot_bbob_timing_scaling_cli_with_real_csv(tmp_path):
    """Verify CLI execution on real timing_summary.csv produces non-empty PDF and PNG figures."""
    real_csv = Path(PROJECT_ROOT) / "results/sweep_bbob_timing/analysis/timing_summary.csv"
    assert real_csv.exists(), f"Missing real CSV at {real_csv}"

    out_dir = tmp_path / "real_figs"
    res = subprocess.run(
        [
            sys.executable,
            str(Path(PROJECT_ROOT) / "scripts/plot_bbob_timing_scaling.py"),
            "--csv-path",
            str(real_csv),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"CLI failed: {res.stderr}"

    for fname in [
        "timing_scaling_mean_std.pdf",
        "timing_scaling_mean_std.png",
        "timing_scaling_by_family.pdf",
        "timing_scaling_by_family.png",
    ]:
        p = out_dir / fname
        assert p.exists()
        assert p.stat().st_size > 1024
