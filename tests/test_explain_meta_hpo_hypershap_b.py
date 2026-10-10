"""Tests for HyperSHAP analysis of Meta-SMAC Proximity B HPO results.

Strict TDD suite verifying:
1. Leaderboard loading & data validation (decay_lambda, eps, loss_normalized_regret)
2. ConfigSpace construction and bounds verification (decay_lambda: [0.2, 2.0], eps: [0.02, 0.20])
3. Surrogate model training (ExtraTreesRegressor)
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

from scripts.explain_meta_hpo_hypershap_b import (
    load_leaderboard,
    build_config_space,
    fit_surrogate_model,
    run_hypershap_analysis,
    save_visualizations,
    save_summary_markdown,
)


@pytest.fixture
def synthetic_leaderboard_csv(tmp_path: Path) -> Path:
    """Fixture providing a deterministic synthetic leaderboard for Proximity B."""
    csv_path = tmp_path / "meta_leaderboard.csv"
    np.random.seed(42)
    n_rows = 50
    decay_lambdas = np.random.uniform(0.2, 2.0, size=n_rows)
    epsilons = np.random.uniform(0.02, 0.20, size=n_rows)

    # Synthetic loss function with known minimum and interaction:
    # Baseline cfg_001 has decay_lambda=1.345, eps=0.16
    decay_lambdas[0] = 1.345
    epsilons[0] = 0.16

    # Optimal cfg_042 has decay_lambda=0.5, eps=0.05
    decay_lambdas[41] = 0.5
    epsilons[41] = 0.05

    losses = (decay_lambdas - 0.5) ** 2 + 2.0 * (epsilons - 0.05) ** 2 + 0.5 * (decay_lambdas * epsilons) + 0.05
    losses[41] = 0.001  # Guarantee cfg_042 is the global minimum

    rows = []
    for i in range(n_rows):
        cid = f"cfg_{i+1:03d}"
        rows.append({
            "iteration": i + 1,
            "config_id": cid,
            "decay_lambda": decay_lambdas[i],
            "eps": epsilons[i],
            "loss_normalized_regret": float(losses[i]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_leaderboard(synthetic_leaderboard_csv: Path):
    """Validates loading and column validation of Proximity B meta-leaderboard."""
    df = load_leaderboard(synthetic_leaderboard_csv)

    assert len(df) == 50
    required_cols = {"iteration", "config_id", "decay_lambda", "eps", "loss_normalized_regret"}
    assert required_cols.issubset(set(df.columns)), f"Missing columns in {df.columns}"

    # Verify baseline and optimal extraction
    cfg_001 = df[df["config_id"] == "cfg_001"].iloc[0]
    assert pytest.approx(cfg_001["decay_lambda"]) == 1.345
    assert pytest.approx(cfg_001["eps"]) == 0.16

    min_row = df.loc[df["loss_normalized_regret"].idxmin()]
    assert min_row["config_id"] == "cfg_042"


def test_load_leaderboard_missing_columns(tmp_path: Path):
    """Validates error handling when required columns are absent."""
    csv_path = tmp_path / "invalid_leaderboard.csv"
    pd.DataFrame({"iteration": [1], "decay_lambda": [1.0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_leaderboard(csv_path)


def test_build_config_space():
    """Verifies ConfigurationSpace matches Proximity B parameters and bounds."""
    cs = build_config_space(seed=42)
    assert isinstance(cs, ConfigurationSpace)
    hyperparameters = dict(cs)

    assert set(hyperparameters.keys()) == {"decay_lambda", "eps"}
    assert "k" not in hyperparameters

    assert pytest.approx(hyperparameters["decay_lambda"].lower) == 0.2
    assert pytest.approx(hyperparameters["decay_lambda"].upper) == 2.0
    assert pytest.approx(hyperparameters["eps"].lower) == 0.02
    assert pytest.approx(hyperparameters["eps"].upper) == 0.20

    default_cfg = cs.get_default_configuration()
    assert pytest.approx(default_cfg["decay_lambda"]) == 1.345
    assert pytest.approx(default_cfg["eps"]) == 0.16


def test_fit_surrogate_model(synthetic_leaderboard_csv: Path):
    """Verifies surrogate model fitting and config evaluation."""
    df = load_leaderboard(synthetic_leaderboard_csv)
    cs = build_config_space(seed=42)
    surrogate = fit_surrogate_model(df, cs=cs, random_state=42)

    cfg_baseline = Configuration(cs, values={"decay_lambda": 1.345, "eps": 0.16})
    cfg_optimal = Configuration(cs, values={"decay_lambda": 0.5, "eps": 0.05})

    pred_baseline = surrogate.evaluate_config(cfg_baseline)
    pred_optimal = surrogate.evaluate_config(cfg_optimal)

    assert isinstance(pred_baseline, float)
    assert isinstance(pred_optimal, float)
    assert pred_baseline > 0.0
    assert pred_optimal > 0.0
    assert pred_optimal < pred_baseline


def test_run_hypershap_analysis(synthetic_leaderboard_csv: Path):
    """Verifies HyperSHAP computation of main effects and 2nd-order FSII interactions."""
    df = load_leaderboard(synthetic_leaderboard_csv)
    cs = build_config_space(seed=42)
    surrogate = fit_surrogate_model(df, cs=cs, random_state=42)

    best_id = df.loc[df["loss_normalized_regret"].idxmin(), "config_id"]
    results = run_hypershap_analysis(
        df=df,
        cs=cs,
        surrogate=surrogate,
        baseline_cfg_id="cfg_001",
        best_cfg_id=best_id,
    )

    assert "ablation_interactions" in results
    assert "hypershap" in results
    assert "baseline_cfg" in results
    assert "best_cfg" in results
    assert results["best_cfg_id"] == "cfg_042"

    iv = results["ablation_interactions"]
    named_iv = results["hypershap"].get_interaction_values_with_names(iv)

    assert ("decay_lambda",) in named_iv
    assert ("eps",) in named_iv
    assert ("decay_lambda", "eps") in named_iv or ("eps", "decay_lambda") in named_iv

    pred_baseline = surrogate.evaluate_config(results["baseline_cfg"])
    pred_optimal = surrogate.evaluate_config(results["best_cfg"])
    actual_diff = pred_optimal - pred_baseline

    sum_effects = sum(v for k_tuple, v in named_iv.items() if len(k_tuple) > 0)
    assert pytest.approx(sum_effects, rel=1e-2) == actual_diff


def test_output_artifacts_generation(synthetic_leaderboard_csv: Path, tmp_path: Path):
    """Verifies SI-graph visualization and Markdown report generation for Proximity B."""
    df = load_leaderboard(synthetic_leaderboard_csv)
    cs = build_config_space(seed=42)
    surrogate = fit_surrogate_model(df, cs=cs, random_state=42)

    best_id = df.loc[df["loss_normalized_regret"].idxmin(), "config_id"]
    results = run_hypershap_analysis(
        df=df,
        cs=cs,
        surrogate=surrogate,
        baseline_cfg_id="cfg_001",
        best_cfg_id=best_id,
    )

    out_dir = tmp_path / "hypershap_b_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_path = save_visualizations(
        shap=results["hypershap"],
        interaction_values=results["ablation_interactions"],
        out_dir=out_dir,
    )
    assert fig_path.exists()
    assert fig_path.name == "hypershap_si_graph.png"
    assert fig_path.stat().st_size > 500

    summary_path = out_dir / "hypershap_summary.md"
    save_summary_markdown(
        results=results,
        out_path=summary_path,
    )
    assert summary_path.exists()
    content = summary_path.read_text()
    assert "cfg_001" in content
    assert "cfg_042" in content
    assert "decay_lambda" in content
    assert "eps" in content
    # Verify Proximity B specific insights: topological decay vs distance tolerance balance without k-cutoff
    assert "topological decay" in content.lower() or "decay" in content.lower()
    assert "tolerance" in content.lower()
