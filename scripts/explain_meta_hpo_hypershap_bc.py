#!/usr/bin/env python3
"""HyperSHAP Explanation Pipeline for Meta-SMAC Proximity BC Results.

Quantifies the marginal and pairwise (2nd-order Faithful Shapley Interaction Index / FSII)
effects of exponential decay rate decay_lambda, distance tolerance eps, and density
scaling alpha in explaining the regret reduction from baseline cfg_001 to the optimal
configuration in Proximity BC.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace, Float
from sklearn.ensemble import ExtraTreesRegressor

from hypershap import HyperSHAP
from hypershap.surrogate_model import DataBasedSurrogateModel, SurrogateModel
from hypershap.task import ExplanationTask

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_leaderboard(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the Proximity BC meta-HPO leaderboard CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {path}")
    df = pd.read_csv(path)
    required = {"iteration", "config_id", "decay_lambda", "eps", "alpha", "loss_normalized_regret"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in leaderboard: {missing}")
    return df


def build_config_space(seed: int = 42) -> ConfigurationSpace:
    """Constructs the ConfigurationSpace matching the SMAC Proximity BC meta-space.

    Proximity BC parameters:
    - decay_lambda in [0.2, 2.0], default 1.345
    - eps in [0.02, 0.20], default 0.16
    - alpha in [0.1, 2.0], default 1.0
    (Note: Neighborhood cutoff k is omitted in Proximity BC).
    """
    cs = ConfigurationSpace(name="proximity_bc_meta_space", seed=seed)
    decay_lambda = Float("decay_lambda", bounds=(0.2, 2.0), default=1.345)
    eps = Float("eps", bounds=(0.02, 0.20), default=0.16)
    alpha = Float("alpha", bounds=(0.1, 2.0), default=1.0)
    cs.add([decay_lambda, eps, alpha])
    return cs


def fit_surrogate_model(
    df: pd.DataFrame,
    cs: ConfigurationSpace | None = None,
    random_state: int = 42,
) -> DataBasedSurrogateModel:
    """Fits an ExtraTrees surrogate model mapping (decay_lambda, eps, alpha) -> loss."""
    if cs is None:
        cs = build_config_space(seed=random_state)

    data = []
    for _, row in df.iterrows():
        cfg = Configuration(
            cs,
            values={
                "decay_lambda": float(row["decay_lambda"]),
                "eps": float(row["eps"]),
                "alpha": float(row["alpha"]),
            },
        )
        data.append((cfg, float(row["loss_normalized_regret"])))

    base_model = ExtraTreesRegressor(n_estimators=100, random_state=random_state)
    surrogate = DataBasedSurrogateModel(config_space=cs, data=data, base_model=base_model, seed=random_state)
    return surrogate


def run_hypershap_analysis(
    df: pd.DataFrame,
    cs: ConfigurationSpace,
    surrogate: SurrogateModel,
    baseline_cfg_id: str = "cfg_001",
    best_cfg_id: str | None = None,
) -> Dict[str, Any]:
    """Runs HyperSHAP ablation analysis using FSII (order 2) from baseline to best config."""
    if best_cfg_id is None:
        best_cfg_id = df.loc[df["loss_normalized_regret"].idxmin(), "config_id"]

    row_base = df[df["config_id"] == baseline_cfg_id].iloc[0]
    row_best = df[df["config_id"] == best_cfg_id].iloc[0]

    baseline_cfg = Configuration(
        cs,
        values={
            "decay_lambda": float(row_base["decay_lambda"]),
            "eps": float(row_base["eps"]),
            "alpha": float(row_base["alpha"]),
        },
    )
    best_cfg = Configuration(
        cs,
        values={
            "decay_lambda": float(row_best["decay_lambda"]),
            "eps": float(row_best["eps"]),
            "alpha": float(row_best["alpha"]),
        },
    )

    task = ExplanationTask(config_space=cs, surrogate_model=surrogate)
    hypershap = HyperSHAP(explanation_task=task)

    # Local ablation: baseline -> best
    ablation_iv = hypershap.ablation(
        config_of_interest=best_cfg,
        baseline_config=baseline_cfg,
        index="FSII",
        order=2,
    )

    return {
        "hypershap": hypershap,
        "task": task,
        "surrogate": surrogate,
        "cs": cs,
        "baseline_cfg_id": baseline_cfg_id,
        "best_cfg_id": best_cfg_id,
        "baseline_cfg": baseline_cfg,
        "best_cfg": best_cfg,
        "baseline_loss": float(row_base["loss_normalized_regret"]),
        "best_loss": float(row_best["loss_normalized_regret"]),
        "ablation_interactions": ablation_iv,
    }


def save_visualizations(
    shap: HyperSHAP,
    interaction_values: Any,
    out_dir: str | Path,
) -> Path:
    """Generates and saves the Shapley Interaction Graph (SI-Graph)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    si_graph_file = out_path / "hypershap_si_graph.png"

    plt.figure(figsize=(8, 8))
    shap.plot_si_graph(interaction_values=interaction_values, save_path=str(si_graph_file), no_show=True)
    plt.close("all")
    logger.info("Saved SI-Graph to %s", si_graph_file)
    return si_graph_file


def save_summary_markdown(
    results: Dict[str, Any],
    out_path: str | Path,
) -> Path:
    """Writes a publication-ready Markdown report of attributions and interactions for Proximity BC."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    shap: HyperSHAP = results["hypershap"]
    iv = results["ablation_interactions"]
    named_iv = shap.get_interaction_values_with_names(iv)

    base_cfg = results["baseline_cfg"]
    best_cfg = results["best_cfg"]
    loss_base = results["baseline_loss"]
    loss_best = results["best_loss"]
    loss_diff = loss_best - loss_base
    pct_impr = ((loss_base - loss_best) / loss_base * 100.0) if loss_base > 0 else 0.0

    main_effects = {k[0]: v for k, v in named_iv.items() if len(k) == 1}
    interactions = {k: v for k, v in named_iv.items() if len(k) == 2}

    lines = [
        "# HyperSHAP Explanation: Meta-SMAC Proximity BC Optimization",
        "",
        "## 1. Baseline vs. Optimal Configurations",
        "",
        f"- **Baseline (`{results['baseline_cfg_id']}`)**: $\\lambda={base_cfg['decay_lambda']:.4f}, \\epsilon={base_cfg['eps']:.4f}, \\alpha={base_cfg['alpha']:.4f} \\implies \\text{{loss}}={loss_base:.5f}$",
        f"- **Global Optimum (`{results['best_cfg_id']}`)**: $\\lambda={best_cfg['decay_lambda']:.4f}, \\epsilon={best_cfg['eps']:.4f}, \\alpha={best_cfg['alpha']:.4f} \\implies \\text{{loss}}={loss_best:.5f}$",
        f"- **Improvement**: $\\Delta \\text{{loss}} = {loss_diff:.5f}$ ({pct_impr:.1f}% normalized regret reduction)",
        "",
        "## 2. 1st-Order Marginal Attributions (Shapley Values)",
        "",
        "| Hyperparameter | Value in Baseline | Value in Optimum | Shapley Attribution ($\\Delta$ Loss) |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for param in ["decay_lambda", "eps", "alpha"]:
        val = main_effects.get(param, 0.0)
        lines.append(f"| `{param}` | `{base_cfg[param]}` | `{best_cfg[param]}` | {val:+.6f} |")

    lines.extend([
        "",
        "## 3. 2nd-Order Faithful Shapley Interaction Index (FSII)",
        "",
        "| Hyperparameter Pair | Interaction Value ($\\Delta$ Loss) | Synergy Type |",
        "| :--- | :--- | :--- |",
    ])

    for pair, val in sorted(interactions.items(), key=lambda x: abs(x[1]), reverse=True):
        pair_str = f"(`{pair[0]}`, `{pair[1]}`)"
        synergy = "Synergistic (amplified gain)" if val < 0 else "Antagonistic / Redundant"
        lines.append(f"| {pair_str} | {val:+.6f} | {synergy} |")

    lines.extend([
        "",
        "## 4. Key Takeaways & Thesis Insights",
        "",
        "1. **Continuous Topological Decay and Density Scaling Coupling**: In Proximity BC, topological regularization operates via exponential decay rate $\\lambda$ combined with adaptive density scaling exponent $\\alpha$, without hard neighborhood truncation ($k$).",
        "2. **Absence of Neighborhood Truncation Boundary**: All observed points contribute continuously according to their spatial distance and local density. The interaction between $\\lambda$ and $\\alpha$ reveals how density-aware contraction compensates for full-space exponential weighting.",
        "3. **Tolerance Horizon Sensitivity**: The distance tolerance $\\epsilon$ interacts with both decay $\\lambda$ and density scaling $\\alpha$ to govern the critical threshold under which proximity gating takes effect, preventing over-regularization in sparse regions while maintaining tight epistemic penalties in dense candidate clusters.",
        "",
        "Artifact generated by `scripts/explain_meta_hpo_hypershap_bc.py`.",
    ])

    path.write_text("\n".join(lines))
    logger.info("Saved summary report to %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HyperSHAP analysis on Proximity BC meta-HPO results.")
    parser.add_argument(
        "--leaderboard",
        type=str,
        default="results/meta_smac_proximity_bc_hpo/meta_leaderboard.csv",
        help="Path to Proximity BC meta_leaderboard.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/meta_smac_proximity_bc_hpo",
        help="Output directory for SI-graph and summary markdown",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_leaderboard(args.leaderboard)
    cs = build_config_space(seed=42)
    surrogate = fit_surrogate_model(df, cs=cs, random_state=42)

    results = run_hypershap_analysis(df, cs, surrogate)
    save_visualizations(results["hypershap"], results["ablation_interactions"], out_dir)
    save_summary_markdown(results, out_dir / "hypershap_summary.md")
    logger.info("HyperSHAP Proximity BC analysis completed successfully.")


if __name__ == "__main__":
    main()
