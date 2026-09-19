"""Tests for HyperSHAP analysis of Meta-SMAC Proximity-LCB results.

Strict TDD suite verifying:
1. Leaderboard loading & data validation (100 rows, baseline cfg_001, optimal cfg_087)
2. ConfigSpace construction and bounds verification
3. Surrogate model training (ExtraTreesRegressor / RandomForestRegressor)
4. HyperSHAP task construction, 1st-order attributions, and 2nd-order FSII interactions
5. Publication artifact generation (SI-Graph plot and Markdown summary)
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ConfigSpace import Configuration, ConfigurationSpace

from scripts.explain_meta_hpo_hypershap import (
    load_leaderboard,
    build_config_space,
    fit_surrogate_model,
    run_hypershap_analysis,
    save_visualizations,
    save_summary_markdown,
)

LEADERBOARD_PATH = Path("results/meta_smac_proximity_hpo/meta_leaderboard.csv")


def test_load_leaderboard():
    """Validates loading of meta_leaderboard.csv and extraction of baseline & optimal configs."""
    assert LEADERBOARD_PATH.exists(), f"Leaderboard not found at {LEADERBOARD_PATH}"
    df = load_leaderboard(LEADERBOARD_PATH)

    # Validate row count and required columns
    assert len(df) == 100, f"Expected 100 iterations, got {len(df)}"
    required_cols = {"iteration", "config_id", "k", "decay_lambda", "eps", "loss_normalized_regret"}
    assert required_cols.issubset(set(df.columns)), f"Missing columns in {df.columns}"

    # Extract baseline cfg_001
    cfg_001 = df[df["config_id"] == "cfg_001"].iloc[0]
    assert cfg_001["k"] == 25
    assert pytest.approx(cfg_001["decay_lambda"], rel=1e-3) == 1.345
    assert pytest.approx(cfg_001["eps"], rel=1e-3) == 0.1678
    assert pytest.approx(cfg_001["loss_normalized_regret"], rel=1e-3) == 0.07946

    # Extract optimal cfg_087
    cfg_087 = df[df["config_id"] == "cfg_087"].iloc[0]
    assert cfg_087["k"] == 28
    assert pytest.approx(cfg_087["decay_lambda"], rel=1e-3) == 0.204867
    assert pytest.approx(cfg_087["eps"], rel=1e-3) == 0.080792
    assert pytest.approx(cfg_087["loss_normalized_regret"], rel=1e-3) == 0.013378

    # Verify cfg_087 is indeed the global minimum
    min_row = df.loc[df["loss_normalized_regret"].idxmin()]
    assert min_row["config_id"] == "cfg_087"


def test_build_config_space():
    """Verifies that the configuration space matches the meta-HPO bounds and default values."""
    cs = build_config_space(seed=42)
    assert isinstance(cs, ConfigurationSpace)
    hyperparameters = dict(cs)

    assert set(hyperparameters.keys()) == {"k", "decay_lambda", "eps"}
    assert hyperparameters["k"].lower == 5
    assert hyperparameters["k"].upper == 30
    assert pytest.approx(hyperparameters["decay_lambda"].lower) == 0.2
    assert pytest.approx(hyperparameters["decay_lambda"].upper) == 2.0
    assert pytest.approx(hyperparameters["eps"].lower) == 0.02
    assert pytest.approx(hyperparameters["eps"].upper) == 0.20

    default_cfg = cs.get_default_configuration()
    assert default_cfg["k"] == 25
    assert pytest.approx(default_cfg["decay_lambda"]) == 1.345
    assert pytest.approx(default_cfg["eps"]) == 0.1678


def test_fit_surrogate_model():
    """Verifies surrogate model fitting and predictive performance on meta-HPO data."""
    df = load_leaderboard(LEADERBOARD_PATH)
    cs = build_config_space(seed=42)
    surrogate = fit_surrogate_model(df, cs=cs, random_state=42)

    # Evaluate surrogate on baseline and optimal configurations
    cfg_baseline = Configuration(cs, values={"k": 25, "decay_lambda": 1.345, "eps": 0.1678})
    cfg_optimal = Configuration(cs, values={"k": 28, "decay_lambda": 0.204867, "eps": 0.080792})

    pred_baseline = surrogate.evaluate_config(cfg_baseline)
    pred_optimal = surrogate.evaluate_config(cfg_optimal)

    assert isinstance(pred_baseline, float)
    assert isinstance(pred_optimal, float)
    assert pred_baseline > 0.0
    assert pred_optimal > 0.0
    assert pred_optimal < pred_baseline


def test_run_hypershap_analysis():
    """Verifies HyperSHAP computation of main effects and 2nd-order FSII interactions."""
    df = load_leaderboard(LEADERBOARD_PATH)
    cs = build_config_space(seed=42)
    surrogate = fit_surrogate_model(df, cs=cs, random_state=42)

    results = run_hypershap_analysis(
        df=df,
        cs=cs,
        surrogate=surrogate,
        baseline_cfg_id="cfg_001",
        best_cfg_id="cfg_087",
    )

    assert "ablation_interactions" in results
    assert "hypershap" in results
    assert "baseline_cfg" in results
    assert "best_cfg" in results

    iv = results["ablation_interactions"]
    named_iv = results["hypershap"].get_interaction_values_with_names(iv)

    assert ("k",) in named_iv
    assert ("decay_lambda",) in named_iv
    assert ("eps",) in named_iv

    pred_baseline = surrogate.evaluate_config(results["baseline_cfg"])
    pred_optimal = surrogate.evaluate_config(results["best_cfg"])
    actual_diff = pred_optimal - pred_baseline

    sum_effects = sum(v for k_tuple, v in named_iv.items() if len(k_tuple) > 0)
    assert pytest.approx(sum_effects, rel=1e-2) == actual_diff


def test_output_artifacts_generation(tmp_path):
    """Verifies SI-graph visualization and Markdown summary generation."""
    df = load_leaderboard(LEADERBOARD_PATH)
    cs = build_config_space(seed=42)
    surrogate = fit_surrogate_model(df, cs=cs, random_state=42)

    results = run_hypershap_analysis(
        df=df,
        cs=cs,
        surrogate=surrogate,
        baseline_cfg_id="cfg_001",
        best_cfg_id="cfg_087",
    )

    out_dir = tmp_path / "hypershap_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_path = save_visualizations(
        shap=results["hypershap"],
        interaction_values=results["ablation_interactions"],
        out_dir=out_dir,
    )
    assert fig_path.exists()
    assert fig_path.stat().st_size > 1000

    summary_path = out_dir / "hypershap_summary.md"
    save_summary_markdown(
        results=results,
        out_path=summary_path,
    )
    assert summary_path.exists()
    content = summary_path.read_text()
    assert "cfg_001" in content
    assert "cfg_087" in content
    assert "decay_lambda" in content
