"""Unit and integration tests for extrapolation results aggregation and scorecard pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.aggregate_extrapolation_results import (
    build_dimension_strata_matrix,
    build_objective_scorecard_dataframe,
    build_parser,
    build_sample_size_scorecard_dataframe,
    build_scorecard_dataframe,
    build_surrogate_scorecard_dataframe,
    build_uq_ablation_scorecard_dataframe,
    compute_cliffs_delta,
    compute_paired_comparison,
    compute_paired_wilcoxon,
    generate_markdown_report,
    generate_notion_scorecard,
    load_summary_records,
    main,
    run_aggregation,
)


def _make_mock_summary(
    dimension: int,
    n_train: int,
    function_name: str,
    strategy: str,
    seed: int,
    slcb_spearman_dist: float,
    plcb_spearman_dist: float,
    slcb_spearman_err: float,
    plcb_spearman_err: float,
    slcb_picp: float,
    plcb_picp: float,
    slcb_winkler: float,
    plcb_winkler: float,
    slcb_auroc: float,
    plcb_auroc: float,
    surrogate: str = "smac_default",
) -> Dict[str, Any]:
    """Create a mock summary dictionary matching run_single_experiment output schema."""
    summary: Dict[str, Any] = {
        "dimension": dimension,
        "n_train": n_train,
        "function_name": function_name,
        "sampling_strategy": strategy,
        "seed": seed,
        "surrogate": surrogate,
        "surrogate_type": surrogate,
        "n_test": 1000,
        "elapsed_seconds": 1.5,
        # Global metrics
        "spearman_dist_slcb": slcb_spearman_dist,
        "spearman_dist_plcb": plcb_spearman_dist,
        "spearman_err_slcb": slcb_spearman_err,
        "spearman_err_plcb": plcb_spearman_err,
        "picp_slcb": slcb_picp,
        "picp_plcb": plcb_picp,
        "mpiw_slcb": 1.2,
        "mpiw_plcb": 1.8,
        "winkler_slcb": slcb_winkler,
        "winkler_plcb": plcb_winkler,
        "auroc_slcb": slcb_auroc,
        "auroc_plcb": plcb_auroc,
        "outlier_auroc_slcb": slcb_auroc,
        "outlier_auroc_plcb": plcb_auroc,
        "auprc_slcb": 0.4,
        "auprc_plcb": 0.7,
        # Ablation estimators
        "spearman_dist_u_rf_fire_lower": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 - 0.05,
        "spearman_err_u_rf_fire_lower": (slcb_spearman_err + plcb_spearman_err) / 2.0,
        "picp_u_rf_fire_lower": (slcb_picp + plcb_picp) / 2.0,
        "mpiw_u_rf_fire_lower": 1.4,
        "winkler_u_rf_fire_lower": (slcb_winkler + plcb_winkler) / 2.0,
        "outlier_auroc_u_rf_fire_lower": (slcb_auroc + plcb_auroc) / 2.0,
        "spearman_dist_u_prox_a_lower": (slcb_spearman_dist + plcb_spearman_dist) / 2.0,
        "spearman_err_u_prox_a_lower": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.02,
        "picp_u_prox_a_lower": (slcb_picp + plcb_picp) / 2.0 + 0.05,
        "mpiw_u_prox_a_lower": 1.5,
        "winkler_u_prox_a_lower": (slcb_winkler + plcb_winkler) / 2.0 - 2.0,
        "outlier_auroc_u_prox_a_lower": (slcb_auroc + plcb_auroc) / 2.0 + 0.05,
        "spearman_dist_u_prox_b_lower": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 + 0.05,
        "spearman_err_u_prox_b_lower": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.04,
        "picp_u_prox_b_lower": (slcb_picp + plcb_picp) / 2.0 + 0.08,
        "mpiw_u_prox_b_lower": 1.6,
        "winkler_u_prox_b_lower": (slcb_winkler + plcb_winkler) / 2.0 - 4.0,
        "outlier_auroc_u_prox_b_lower": (slcb_auroc + plcb_auroc) / 2.0 + 0.08,
        "spearman_dist_u_prox_bc_lower": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 + 0.08,
        "spearman_err_u_prox_bc_lower": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.05,
        "picp_u_prox_bc_lower": (slcb_picp + plcb_picp) / 2.0 + 0.10,
        "mpiw_u_prox_bc_lower": 1.7,
        "winkler_u_prox_bc_lower": (slcb_winkler + plcb_winkler) / 2.0 - 5.0,
        "outlier_auroc_u_prox_bc_lower": (slcb_auroc + plcb_auroc) / 2.0 + 0.10,
        "spearman_dist_u_shaker_total_lower": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 - 0.02,
        "spearman_err_u_shaker_total_lower": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.01,
        "picp_u_shaker_total_lower": (slcb_picp + plcb_picp) / 2.0 + 0.02,
        "mpiw_u_shaker_total_lower": 1.45,
        "winkler_u_shaker_total_lower": (slcb_winkler + plcb_winkler) / 2.0 - 1.0,
        "outlier_auroc_u_shaker_total_lower": (slcb_auroc + plcb_auroc) / 2.0 + 0.02,
        "global": {
            "slcb": {
                "spearman_dist": slcb_spearman_dist,
                "spearman_err": slcb_spearman_err,
                "picp": slcb_picp,
                "mpiw": 1.2,
                "winkler": slcb_winkler,
                "auroc": slcb_auroc,
                "auprc": 0.4,
            },
            "plcb": {
                "spearman_dist": plcb_spearman_dist,
                "spearman_err": plcb_spearman_err,
                "picp": plcb_picp,
                "mpiw": 1.8,
                "winkler": plcb_winkler,
                "auroc": plcb_auroc,
                "outlier_auroc": plcb_auroc,
                "auprc": 0.7,
            },
            "rf_fire": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 - 0.05,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0,
                "picp": (slcb_picp + plcb_picp) / 2.0,
                "mpiw": 1.4,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0,
            },
            "u_rf_fire_lower": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 - 0.05,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0,
                "picp": (slcb_picp + plcb_picp) / 2.0,
                "mpiw": 1.4,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0,
            },
            "prox_a": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.02,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.05,
                "mpiw": 1.5,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 2.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.05,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.05,
            },
            "u_prox_a_lower": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.02,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.05,
                "mpiw": 1.5,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 2.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.05,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.05,
            },
            "prox_b": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 + 0.05,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.04,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.08,
                "mpiw": 1.6,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 4.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.08,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.08,
            },
            "u_prox_b_lower": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 + 0.05,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.04,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.08,
                "mpiw": 1.6,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 4.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.08,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.08,
            },
            "prox_bc": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 + 0.08,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.05,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.10,
                "mpiw": 1.7,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 5.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.10,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.10,
            },
            "u_prox_bc_lower": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 + 0.08,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.05,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.10,
                "mpiw": 1.7,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 5.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.10,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.10,
            },
            "shaker_total": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 - 0.02,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.01,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.02,
                "mpiw": 1.45,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 1.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.02,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.02,
            },
            "u_shaker_total_lower": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 - 0.02,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.01,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.02,
                "mpiw": 1.45,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 1.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.02,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.02,
            },
        },
    }

    # Strata breakdown
    strata_data = {}
    for s in range(4):
        strata_data[str(s)] = {
            "slcb": {
                "spearman_dist": max(-0.2, slcb_spearman_dist - 0.05 * s),
                "spearman_err": slcb_spearman_err,
                "picp": max(0.2, slcb_picp - 0.1 * s),
                "mpiw": 1.2,
                "winkler": slcb_winkler * (1.0 + 0.2 * s),
                "auroc": slcb_auroc,
                "outlier_auroc": slcb_auroc,
            },
            "plcb": {
                "spearman_dist": max(0.4, plcb_spearman_dist - 0.02 * s),
                "spearman_err": plcb_spearman_err,
                "picp": plcb_picp,
                "mpiw": 1.8 * (1.0 + 0.1 * s),
                "winkler": plcb_winkler * (1.0 + 0.05 * s),
                "auroc": plcb_auroc,
                "outlier_auroc": plcb_auroc,
            },
            "rf_fire": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 - 0.05,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0,
                "picp": (slcb_picp + plcb_picp) / 2.0,
                "mpiw": 1.4,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0,
            },
            "u_rf_fire_lower": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 - 0.05,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0,
                "picp": (slcb_picp + plcb_picp) / 2.0,
                "mpiw": 1.4,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0,
            },
            "prox_a": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.02,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.05,
                "mpiw": 1.5,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 2.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.05,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.05,
            },
            "u_prox_a_lower": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.02,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.05,
                "mpiw": 1.5,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 2.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.05,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.05,
            },
            "prox_b": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 + 0.05,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.04,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.08,
                "mpiw": 1.6,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 4.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.08,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.08,
            },
            "u_prox_b_lower": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 + 0.05,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.04,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.08,
                "mpiw": 1.6,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 4.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.08,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.08,
            },
            "prox_bc": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 + 0.08,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.05,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.10,
                "mpiw": 1.7,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 5.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.10,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.10,
            },
            "u_prox_bc_lower": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 + 0.08,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.05,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.10,
                "mpiw": 1.7,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 5.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.10,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.10,
            },
            "shaker_total": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 - 0.02,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.01,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.02,
                "mpiw": 1.45,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 1.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.02,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.02,
            },
            "u_shaker_total_lower": {
                "spearman_dist": (slcb_spearman_dist + plcb_spearman_dist) / 2.0 - 0.02,
                "spearman_err": (slcb_spearman_err + plcb_spearman_err) / 2.0 + 0.01,
                "picp": (slcb_picp + plcb_picp) / 2.0 + 0.02,
                "mpiw": 1.45,
                "winkler": (slcb_winkler + plcb_winkler) / 2.0 - 1.0,
                "auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.02,
                "outlier_auroc": (slcb_auroc + plcb_auroc) / 2.0 + 0.02,
            },
        }
    summary["strata"] = strata_data
    summary["strata_metrics"] = strata_data
    return summary


@pytest.fixture
def mock_summaries_dir(tmp_path: Path) -> Path:
    """Fixture creating a directory of realistic mock extrapolation summaries."""
    summaries_dir = tmp_path / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    dimensions = [2, 16, 32]
    strategies = ["natural", "stratified"]
    seeds = [0, 1]
    functions = ["sphere", "rastrigin"]

    for d in dimensions:
        for strat in strategies:
            for s in seeds:
                for fn in functions:
                    # In higher dimensions, SLCB collapses (near 0 or negative correlation)
                    # while PLCB stays strong (> 0.75)
                    if d == 2:
                        slcb_dist = 0.35 + 0.02 * s
                        plcb_dist = 0.88 + 0.01 * s
                    elif d == 16:
                        slcb_dist = 0.05 - 0.02 * s
                        plcb_dist = 0.82 + 0.01 * s
                    else:  # 32
                        slcb_dist = -0.08 + 0.01 * s
                        plcb_dist = 0.79 + 0.02 * s

                    summary_data = _make_mock_summary(
                        dimension=d,
                        n_train=112 if d == 2 else 224,
                        function_name=fn,
                        strategy=strat,
                        seed=s,
                        slcb_spearman_dist=slcb_dist,
                        plcb_spearman_dist=plcb_dist,
                        slcb_spearman_err=0.25,
                        plcb_spearman_err=0.65,
                        slcb_picp=0.62,
                        plcb_picp=0.94,
                        slcb_winkler=45.0,
                        plcb_winkler=15.0,
                        slcb_auroc=0.55,
                        plcb_auroc=0.88,
                    )

                    filename = f"summary_{fn}_d{d}_n{summary_data['n_train']}_{strat}_s{s}.json"
                    with open(summaries_dir / filename, "w", encoding="utf-8") as f:
                        json.dump(summary_data, f, indent=2)

    return summaries_dir


class TestStatisticalHelpers:
    """Unit tests for statistical helpers: Cliff's Delta, Wilcoxon test, paired comparisons."""

    def test_cliffs_delta_all_greater(self):
        x = [10.0, 20.0, 30.0]
        y = [1.0, 2.0, 3.0]
        delta = compute_cliffs_delta(x, y)
        assert pytest.approx(delta) == 1.0

    def test_cliffs_delta_all_smaller(self):
        x = [1.0, 2.0, 3.0]
        y = [10.0, 20.0, 30.0]
        delta = compute_cliffs_delta(x, y)
        assert pytest.approx(delta) == -1.0

    def test_cliffs_delta_identical(self):
        x = [5.0, 5.0, 5.0]
        y = [5.0, 5.0, 5.0]
        delta = compute_cliffs_delta(x, y)
        assert pytest.approx(delta) == 0.0

    def test_cliffs_delta_empty(self):
        delta = compute_cliffs_delta([], [])
        assert np.isnan(delta) or delta == 0.0

    def test_paired_wilcoxon_valid(self):
        x = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        stat, pval = compute_paired_wilcoxon(x, y)
        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0
        assert pval < 0.05  # Strongly significant

    def test_paired_wilcoxon_identical_zero_diff(self):
        x = [5.0, 5.0, 5.0, 5.0]
        y = [5.0, 5.0, 5.0, 5.0]
        stat, pval = compute_paired_wilcoxon(x, y)
        assert pytest.approx(stat) == 0.0
        assert pytest.approx(pval) == 1.0

    def test_paired_wilcoxon_single_element(self):
        x = [10.0]
        y = [5.0]
        stat, pval = compute_paired_wilcoxon(x, y)
        # Should not raise exception; return fallback or valid statistic
        assert not np.isnan(stat)
        assert not np.isnan(pval)

    def test_paired_comparison_higher_is_better(self):
        df = pd.DataFrame({
            "plcb": [0.8, 0.9, 0.7, 0.85],
            "slcb": [0.2, 0.3, 0.25, 0.3],
        })
        comp = compute_paired_comparison(df, plcb_col="plcb", slcb_col="slcb", higher_is_better=True)
        assert comp["wins"] == 4
        assert comp["losses"] == 0
        assert comp["ties"] == 0
        assert comp["plcb_mean"] > comp["slcb_mean"]
        assert comp["diff_mean"] > 0.0
        assert comp["cliffs_delta"] > 0.9

    def test_paired_comparison_lower_is_better(self):
        # E.g. Winkler score or PICP error
        df = pd.DataFrame({
            "plcb": [10.0, 12.0, 11.0],
            "slcb": [30.0, 28.0, 35.0],
        })
        comp = compute_paired_comparison(df, plcb_col="plcb", slcb_col="slcb", higher_is_better=False)
        assert comp["wins"] == 3
        assert comp["losses"] == 0
        assert comp["ties"] == 0
        assert comp["diff_mean"] < 0.0  # plcb - slcb is negative, which is desirable


class TestLoadSummaryRecords:
    """Tests for loading and parsing summary JSON files into structured DataFrames."""

    def test_load_nonexistent_directory(self, tmp_path: Path):
        nonexistent = tmp_path / "does_not_exist"
        df = load_summary_records(nonexistent)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_load_empty_directory(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        df = load_summary_records(empty_dir)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_load_valid_summaries(self, mock_summaries_dir: Path):
        df = load_summary_records(mock_summaries_dir)
        assert isinstance(df, pd.DataFrame)
        # 3 dimensions * 2 strategies * 2 seeds * 2 functions = 24 records
        assert len(df) == 24

        # Check essential columns
        expected_cols = [
            "dimension",
            "n_train",
            "function_name",
            "sampling_strategy",
            "seed",
            "surrogate",
            "spearman_dist_plcb",
            "spearman_dist_slcb",
            "spearman_err_plcb",
            "spearman_err_slcb",
            "picp_plcb",
            "picp_slcb",
            "picp_error_plcb",
            "picp_error_slcb",
            "winkler_plcb",
            "winkler_slcb",
            "outlier_auroc_plcb",
            "outlier_auroc_slcb",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing expected column: {col}"

        # Check derived picp_error = abs(picp - 0.95)
        for _, row in df.iterrows():
            assert pytest.approx(row["picp_error_plcb"]) == abs(row["picp_plcb"] - 0.95)
            assert pytest.approx(row["picp_error_slcb"]) == abs(row["picp_slcb"] - 0.95)

    def test_strata_metrics_extraction(self, mock_summaries_dir: Path):
        df = load_summary_records(mock_summaries_dir)
        # Strata 0, 1, 2, 3 should have columns in the DataFrame
        for s in range(4):
            assert f"stratum_{s}_spearman_dist_plcb" in df.columns
            assert f"stratum_{s}_spearman_dist_slcb" in df.columns
            assert f"stratum_{s}_winkler_plcb" in df.columns

    def test_surrogate_column_extraction(self, tmp_path: Path):
        summary_dir = tmp_path / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        # 1. Summary with explicit surrogate key
        s1 = _make_mock_summary(2, 112, "sphere", "natural", 0, 0.5, 0.8, 0.3, 0.7, 0.8, 0.95, 20.0, 10.0, 0.6, 0.9)
        s1["surrogate"] = "mature"
        s1["surrogate_type"] = "mature"
        with open(summary_dir / "summary_sphere_d2_n112_natural_mature_s0.json", "w") as f:
            json.dump(s1, f)

        # 2. Summary with legacy schema (no surrogate key, fallback to smac_default)
        s2 = _make_mock_summary(2, 112, "sphere", "natural", 1, 0.5, 0.8, 0.3, 0.7, 0.8, 0.95, 20.0, 10.0, 0.6, 0.9)
        s2.pop("surrogate", None)
        s2.pop("surrogate_type", None)
        with open(summary_dir / "summary_sphere_d2_n112_natural_s1.json", "w") as f:
            json.dump(s2, f)

        df = load_summary_records(summary_dir)
        assert len(df) == 2
        assert "surrogate" in df.columns
        row_mature = df[df["seed"] == 0].iloc[0]
        row_legacy = df[df["seed"] == 1].iloc[0]
        assert row_mature["surrogate"] == "mature"
        assert row_legacy["surrogate"] == "smac_default"

    def test_load_summary_records_multi_estimator(self, mock_summaries_dir: Path):
        """Verifies multi-estimator and surrogate ingestion in load_summary_records."""
        df = load_summary_records(mock_summaries_dir)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

        # 1. Verify surrogate, n_train, and k_over_n_ratio
        assert "surrogate" in df.columns
        assert "n_train" in df.columns
        assert "k_over_n_ratio" in df.columns

        for _, row in df.iterrows():
            assert row["surrogate"] == "smac_default"
            assert row["n_train"] in [112, 224]
            assert pytest.approx(row["k_over_n_ratio"]) == 28.0 / float(row["n_train"])

        # 2. Verify global metrics exist for methods: slcb, rf_fire, prox_a, prox_b, prox_bc, plcb, shaker_total
        methods = ["slcb", "rf_fire", "prox_a", "prox_b", "prox_bc", "plcb", "shaker_total"]
        global_metrics = [
            "spearman_dist",
            "spearman_err",
            "picp",
            "picp_error",
            "winkler",
            "outlier_auroc",
        ]
        for m in methods:
            for metric in global_metrics:
                col_name = f"{metric}_{m}"
                assert col_name in df.columns, f"Expected global metric column '{col_name}' in DataFrame"
                assert not df[col_name].isna().all(), f"All values are NaN for '{col_name}'"

            # Check derived picp_error = abs(picp - 0.95)
            for _, row in df.iterrows():
                assert pytest.approx(row[f"picp_error_{m}"]) == abs(row[f"picp_{m}"] - 0.95)

        # 3. Verify per-stratum metrics exist for strata 0..3 for these estimators
        strata_metrics = [
            "spearman_dist",
            "spearman_err",
            "picp",
            "winkler",
            "outlier_auroc",
        ]
        for s in range(4):
            for m in methods:
                for metric in strata_metrics:
                    col_name = f"stratum_{s}_{metric}_{m}"
                    assert col_name in df.columns, f"Expected stratum metric column '{col_name}' in DataFrame"
                    assert not df[col_name].isna().all(), f"All values are NaN for '{col_name}'"



class TestScorecardGeneration:
    """Tests for generating the comprehensive calibration scorecard DataFrame."""

    def test_build_scorecard_dataframe(self, mock_summaries_dir: Path):
        df = load_summary_records(mock_summaries_dir)
        scorecard = build_scorecard_dataframe(df)

        assert isinstance(scorecard, pd.DataFrame)
        assert not scorecard.empty

        # Must group by dimension and sampling_strategy, plus an 'All' overall row
        assert "dimension" in scorecard.columns
        assert "sampling_strategy" in scorecard.columns

        # Verify target metric summary columns
        metrics = ["spearman_dist", "spearman_err", "picp_error", "winkler", "outlier_auroc"]
        for m in metrics:
            assert f"{m}_plcb_mean" in scorecard.columns
            assert f"{m}_plcb_sem" in scorecard.columns
            assert f"{m}_slcb_mean" in scorecard.columns
            assert f"{m}_slcb_sem" in scorecard.columns
            assert f"{m}_diff_mean" in scorecard.columns
            assert f"{m}_pvalue" in scorecard.columns
            assert f"{m}_cliffs_delta" in scorecard.columns
            assert f"{m}_wins" in scorecard.columns
            assert f"{m}_ties" in scorecard.columns
            assert f"{m}_losses" in scorecard.columns

        # Check values in high dimension (e.g. D=32)
        d32_rows = scorecard[scorecard["dimension"] == 32]
        assert not d32_rows.empty
        for _, row in d32_rows.iterrows():
            # PLCB spearman_dist should significantly exceed SLCB
            assert row["spearman_dist_plcb_mean"] > 0.7
            assert row["spearman_dist_slcb_mean"] < 0.1
            # Note: For N=4 (sub-strategy groups), minimum possible two-sided Wilcoxon p-value is 2/16 = 0.125.
            # For the combined group (sampling_strategy='All', N=8), p-value is < 0.05.
            if row["sampling_strategy"] == "All":
                assert row["spearman_dist_pvalue"] < 0.05
            else:
                assert row["spearman_dist_pvalue"] <= 0.125
            assert row["spearman_dist_wins"] > row["spearman_dist_losses"]

    def test_build_objective_scorecard_dataframe(self, mock_summaries_dir: Path):
        df = load_summary_records(mock_summaries_dir)
        obj_scorecard = build_objective_scorecard_dataframe(df)

        assert isinstance(obj_scorecard, pd.DataFrame)
        assert not obj_scorecard.empty
        assert "function_name" in obj_scorecard.columns
        assert "n_experiments" in obj_scorecard.columns

        # Verify target metrics columns
        for m in ["spearman_dist", "spearman_err", "picp_error", "winkler", "outlier_auroc"]:
            assert f"{m}_plcb_mean" in obj_scorecard.columns
            assert f"{m}_slcb_mean" in obj_scorecard.columns
            assert f"{m}_diff_mean" in obj_scorecard.columns
            assert f"{m}_pvalue" in obj_scorecard.columns
            assert f"{m}_cliffs_delta" in obj_scorecard.columns
            assert f"{m}_wins" in obj_scorecard.columns
            assert f"{m}_ties" in obj_scorecard.columns
            assert f"{m}_losses" in obj_scorecard.columns

        # Check function names present: sphere, rastrigin, All
        funcs = obj_scorecard["function_name"].tolist()
        assert "sphere" in funcs
        assert "rastrigin" in funcs
        assert "All" in funcs

        # Total experiments in 'All' row should equal total records
        all_row = obj_scorecard[obj_scorecard["function_name"] == "All"]
        assert all_row.iloc[0]["n_experiments"] == len(df)

    def test_build_dimension_strata_matrix(self, mock_summaries_dir: Path):
        df = load_summary_records(mock_summaries_dir)
        matrix = build_dimension_strata_matrix(df)

        assert isinstance(matrix, pd.DataFrame)
        assert not matrix.empty

        expected_cols = [
            "dimension",
            "stratum",
            "n_experiments",
            "winkler_plcb_mean",
            "winkler_plcb_sem",
            "winkler_slcb_mean",
            "winkler_slcb_sem",
            "winkler_ratio",
            "picp_plcb_mean",
            "picp_plcb_sem",
            "picp_slcb_mean",
            "picp_slcb_sem",
            "auroc_plcb_mean",
            "auroc_plcb_sem",
            "auroc_slcb_mean",
            "auroc_slcb_sem",
        ]
        for col in expected_cols:
            assert col in matrix.columns, f"Missing column: {col}"

        # Dimensions should include 2, 16, 32, and 'All'
        dims = set(matrix["dimension"].dropna().astype(str).unique())
        assert {"2", "16", "32", "All"}.issubset(dims)

        # Strata should include 0, 1, 2, 3, and 'All'
        strata = set(matrix["stratum"].dropna().astype(str).unique())
        assert {"0", "1", "2", "3", "All"}.issubset(strata)

        # Check winkler_ratio is approximately log(winkler_plcb_mean / winkler_slcb_mean)
        for _, r in matrix.iterrows():
            if not np.isnan(r["winkler_ratio"]):
                expected_ratio = np.log(r["winkler_plcb_mean"] / r["winkler_slcb_mean"])
                assert pytest.approx(r["winkler_ratio"]) == expected_ratio


class TestReportGeneration:
    """Tests for Markdown report and Notion text generation."""

    def test_generate_markdown_report(self, mock_summaries_dir: Path):
        df = load_summary_records(mock_summaries_dir)
        scorecard = build_scorecard_dataframe(df)
        report = generate_markdown_report(df, scorecard)

        assert isinstance(report, str)
        assert len(report) > 500

        # Verify key hypotheses sections are present
        assert "Hypothesis 1" in report
        assert "SLCB" in report
        assert "Monotonicity Collapse" in report or "monotonicity" in report.lower()

        assert "Hypothesis 2" in report
        assert "PLCB" in report
        assert "Distance" in report

        assert "Hypothesis 3" in report
        assert "Coverage" in report or "Winkler" in report

        # Verify markdown table syntax
        assert "| Dimension |" in report or "| Stratum |" in report or "|" in report

        # Verify Objective Breakdown and Dimension x Strata matrix sections
        assert "Objective Function Breakdown" in report or "Objective Breakdown" in report
        assert "Dimension x Strata Matrix" in report or "Strata Matrix" in report

    def test_generate_markdown_report_bbob_paradox(self, mock_summaries_dir: Path):
        df = load_summary_records(mock_summaries_dir)
        scorecard = build_scorecard_dataframe(df)
        report = generate_markdown_report(df, scorecard)

        # 1. BBOB Paradox and related mechanisms
        assert "BBOB Optimization Paradox" in report
        assert "Hallucinated Exploration Trap" in report
        assert "Implicit Trust Region" in report

        # 2. Sections or links for Surrogate Architecture, Sample Size Scaling, UQ Component Ablation
        assert "Surrogate Architecture" in report or "extrapolation_surrogate_scorecard.csv" in report
        assert "Sample Size" in report or "extrapolation_sample_size_scorecard.csv" in report
        assert "UQ Component Ablation" in report or "extrapolation_uq_ablation_scorecard.csv" in report

        # 3. Dynamic verdict evaluation:
        # When PLCB has positive correlation and high win rate (like in mock_summaries_dir),
        # Hypothesis 2 should be confirmed:
        assert "CONFIRMED" in report

        # When PLCB has negative correlation or loses majority of runs (win rate < 50%),
        # Hypothesis 2 verdict must NOT be hardcoded "CONFIRMED".
        # It must be marked as "REFUTED (Open-Loop Extrapolation)" and mention "boundary leaf saturation".
        df_negative = df.copy()
        df_negative["spearman_dist_plcb"] = -0.40
        df_negative["spearman_dist_slcb"] = 0.50
        scorecard_neg = build_scorecard_dataframe(df_negative)
        report_neg = generate_markdown_report(df_negative, scorecard_neg)

        assert "REFUTED (Open-Loop Extrapolation)" in report_neg
        assert "boundary leaf saturation" in report_neg.lower()
        # In the Hypothesis 2 section of report_neg, the verdict should be REFUTED (Open-Loop Extrapolation)
        assert "CONFIRMED" not in report_neg.split("## Hypothesis 2")[1].split("---")[0]

    def test_generate_notion_scorecard(self, mock_summaries_dir: Path):
        df = load_summary_records(mock_summaries_dir)
        scorecard = build_scorecard_dataframe(df)
        notion_text = generate_notion_scorecard(df, scorecard)

        assert isinstance(notion_text, str)
        assert len(notion_text) > 300
        # Check Notion formatting markers
        assert "# " in notion_text
        assert "## " in notion_text
        assert "|" in notion_text


class TestCLIExecution:
    """Tests for CLI arguments, execution, and output file persistence."""

    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.summaries_dir == "results/extrapolation_uq/summaries"
        assert args.output_csv == "results/extrapolation_uq/analysis/extrapolation_calibration_scorecard.csv"
        assert args.output_report == "results/extrapolation_uq/analysis/HYPOTHESIS_EVALUATION_REPORT.md"
        assert args.output_notion == "bachelorthesis/extrapolation_uq_scorecard_notion.txt"
        assert args.output_surrogate_csv is None
        assert args.output_sample_size_csv is None
        assert args.output_ablation_csv is None

    def test_cli_empty_directory_graceful(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_summaries"
        empty_dir.mkdir()
        out_csv = tmp_path / "scorecard.csv"
        out_report = tmp_path / "report.md"
        out_notion = tmp_path / "notion.txt"

        exit_code = main([
            "--summaries-dir", str(empty_dir),
            "--output-csv", str(out_csv),
            "--output-report", str(out_report),
            "--output-notion", str(out_notion),
        ])
        assert exit_code == 0
        # Output files should be created (even if containing empty placeholders / headers)
        assert out_csv.is_file()
        assert out_report.is_file()
        assert out_notion.is_file()
        assert (out_csv.parent / "extrapolation_objective_scorecard.csv").is_file()
        assert (out_csv.parent / "extrapolation_dimension_strata_matrix.csv").is_file()
        assert (out_csv.parent / "extrapolation_surrogate_scorecard.csv").is_file()
        assert (out_csv.parent / "extrapolation_sample_size_scorecard.csv").is_file()
        assert (out_csv.parent / "extrapolation_uq_ablation_scorecard.csv").is_file()

    def test_cli_full_execution(self, mock_summaries_dir: Path, tmp_path: Path):
        out_csv = tmp_path / "analysis" / "extrapolation_calibration_scorecard.csv"
        out_report = tmp_path / "analysis" / "HYPOTHESIS_EVALUATION_REPORT.md"
        out_notion = tmp_path / "bachelorthesis" / "extrapolation_uq_scorecard_notion.txt"

        exit_code = main([
            "--summaries-dir", str(mock_summaries_dir),
            "--output-csv", str(out_csv),
            "--output-report", str(out_report),
            "--output-notion", str(out_notion),
        ])
        assert exit_code == 0

        assert out_csv.is_file()
        assert out_csv.stat().st_size > 0
        assert out_report.is_file()
        assert out_report.stat().st_size > 0
        assert out_notion.is_file()
        assert out_notion.stat().st_size > 0

        # Verify additional CSV scorecards were created
        obj_csv = out_csv.parent / "extrapolation_objective_scorecard.csv"
        matrix_csv = out_csv.parent / "extrapolation_dimension_strata_matrix.csv"
        surr_csv = out_csv.parent / "extrapolation_surrogate_scorecard.csv"
        sample_csv = out_csv.parent / "extrapolation_sample_size_scorecard.csv"
        ablation_csv = out_csv.parent / "extrapolation_uq_ablation_scorecard.csv"

        assert obj_csv.is_file()
        assert obj_csv.stat().st_size > 0
        assert matrix_csv.is_file()
        assert matrix_csv.stat().st_size > 0
        assert surr_csv.is_file()
        assert surr_csv.stat().st_size > 0
        assert sample_csv.is_file()
        assert sample_csv.stat().st_size > 0
        assert ablation_csv.is_file()
        assert ablation_csv.stat().st_size > 0

        # Verify CSV can be parsed by pandas
        scorecard_df = pd.read_csv(out_csv)
        assert not scorecard_df.empty
        assert "spearman_dist_plcb_mean" in scorecard_df.columns

        obj_df = pd.read_csv(obj_csv)
        assert not obj_df.empty
        assert "function_name" in obj_df.columns

        matrix_df = pd.read_csv(matrix_csv)
        assert not matrix_df.empty
        assert "winkler_ratio" in matrix_df.columns

        surr_df = pd.read_csv(surr_csv)
        assert not surr_df.empty
        assert "surrogate" in surr_df.columns

        sample_df = pd.read_csv(sample_csv)
        assert not sample_df.empty
        assert "n_train" in sample_df.columns
        assert "k_over_n_ratio" in sample_df.columns

        ablation_df = pd.read_csv(ablation_csv)
        assert not ablation_df.empty
        assert "estimator" in ablation_df.columns
        assert "spearman_dist_diff_vs_slcb" in ablation_df.columns


class TestSurrogateScorecard:
    """Tests for surrogate hyperparameter scorecard generation."""

    METRICS = ["spearman_dist", "spearman_err", "picp_error", "winkler", "outlier_auroc"]
    EXPECTED_SCHEMA = [
        "surrogate",
        "dimension",
        "n_experiments",
        # spearman_dist
        "spearman_dist_plcb_mean", "spearman_dist_plcb_sem",
        "spearman_dist_slcb_mean", "spearman_dist_slcb_sem",
        "spearman_dist_diff_mean", "spearman_dist_pvalue",
        "spearman_dist_cliffs_delta", "spearman_dist_wins",
        "spearman_dist_ties", "spearman_dist_losses",
        # spearman_err
        "spearman_err_plcb_mean", "spearman_err_plcb_sem",
        "spearman_err_slcb_mean", "spearman_err_slcb_sem",
        "spearman_err_diff_mean", "spearman_err_pvalue",
        "spearman_err_cliffs_delta", "spearman_err_wins",
        "spearman_err_ties", "spearman_err_losses",
        # picp_error
        "picp_error_plcb_mean", "picp_error_plcb_sem",
        "picp_error_slcb_mean", "picp_error_slcb_sem",
        "picp_error_diff_mean", "picp_error_pvalue",
        "picp_error_cliffs_delta", "picp_error_wins",
        "picp_error_ties", "picp_error_losses",
        # winkler
        "winkler_plcb_mean", "winkler_plcb_sem",
        "winkler_slcb_mean", "winkler_slcb_sem",
        "winkler_diff_mean", "winkler_pvalue",
        "winkler_cliffs_delta", "winkler_wins",
        "winkler_ties", "winkler_losses",
        # outlier_auroc
        "outlier_auroc_plcb_mean", "outlier_auroc_plcb_sem",
        "outlier_auroc_slcb_mean", "outlier_auroc_slcb_sem",
        "outlier_auroc_diff_mean", "outlier_auroc_pvalue",
        "outlier_auroc_cliffs_delta", "outlier_auroc_wins",
        "outlier_auroc_ties", "outlier_auroc_losses",
    ]

    def test_build_surrogate_scorecard_empty(self):
        """Test that empty DataFrame returns empty scorecard with correct column schema."""
        empty_df = pd.DataFrame()
        sc = build_surrogate_scorecard_dataframe(empty_df)
        assert isinstance(sc, pd.DataFrame)
        assert sc.empty
        assert list(sc.columns) == self.EXPECTED_SCHEMA

    def test_build_surrogate_scorecard_schema_and_grouping(self, mock_summaries_dir: Path):
        """Test scorecard schema and grouping by (surrogate, dimension) with aggregate rows."""
        df = load_summary_records(mock_summaries_dir)
        sc = build_surrogate_scorecard_dataframe(df)

        assert isinstance(sc, pd.DataFrame)
        assert not sc.empty
        assert list(sc.columns) == self.EXPECTED_SCHEMA

        # Check surrogate present: smac_default
        surrogates = sc["surrogate"].unique().tolist()
        assert "smac_default" in surrogates
        assert "All" in surrogates

        # Check dimension slices for smac_default
        smac_rows = sc[sc["surrogate"] == "smac_default"]
        smac_dims = smac_rows["dimension"].tolist()
        assert 2 in smac_dims
        assert 16 in smac_dims
        assert 32 in smac_dims
        assert "All" in smac_dims

        # Check total grand row
        total_rows = sc[(sc["surrogate"] == "All") & (sc["dimension"] == "All")]
        assert len(total_rows) == 1
        assert total_rows.iloc[0]["n_experiments"] == len(df)

        # Check metric values in row
        d32_row = sc[(sc["surrogate"] == "smac_default") & (sc["dimension"] == 32)].iloc[0]
        assert d32_row["spearman_dist_plcb_mean"] > 0.7
        assert d32_row["spearman_dist_slcb_mean"] < 0.1
        assert d32_row["spearman_dist_diff_mean"] > 0.0

    def test_build_surrogate_scorecard_multi_surrogate(self, tmp_path: Path):
        """Test with multi-surrogate mock data (smac_default, mature, shallow)."""
        summary_dir = tmp_path / "multi_surr_summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)

        surrogates = ["smac_default", "mature", "shallow"]
        dims = [2, 16]
        seeds = [0, 1]

        total_created = 0
        for surr in surrogates:
            for d in dims:
                for s in seeds:
                    mock_data = _make_mock_summary(
                        dimension=d,
                        n_train=112 if d == 2 else 224,
                        function_name="sphere",
                        strategy="natural",
                        seed=s,
                        slcb_spearman_dist=0.3,
                        plcb_spearman_dist=0.8,
                        slcb_spearman_err=0.2,
                        plcb_spearman_err=0.6,
                        slcb_picp=0.6,
                        plcb_picp=0.95,
                        slcb_winkler=40.0,
                        plcb_winkler=15.0,
                        slcb_auroc=0.5,
                        plcb_auroc=0.85,
                        surrogate=surr,
                    )
                    fn = f"summary_sphere_d{d}_{surr}_s{s}.json"
                    with open(summary_dir / fn, "w", encoding="utf-8") as f:
                        json.dump(mock_data, f)
                    total_created += 1

        df = load_summary_records(summary_dir)
        assert len(df) == total_created

        sc = build_surrogate_scorecard_dataframe(df)
        assert list(sc.columns) == self.EXPECTED_SCHEMA

        # Check all surrogates have per-dimension rows + an 'All' aggregate row
        for surr in surrogates:
            surr_rows = sc[sc["surrogate"] == surr]
            assert not surr_rows.empty
            surr_dims = surr_rows["dimension"].tolist()
            for d in dims:
                assert d in surr_dims
            assert "All" in surr_dims

            # Verify n_experiments for (surr, 'All') is 4 (2 dims * 2 seeds)
            surr_all = surr_rows[surr_rows["dimension"] == "All"].iloc[0]
            assert surr_all["n_experiments"] == len(dims) * len(seeds)

        # Check master grand total row
        grand_total = sc[(sc["surrogate"] == "All") & (sc["dimension"] == "All")]
        assert len(grand_total) == 1
        assert grand_total.iloc[0]["n_experiments"] == total_created


class TestSampleSizeScorecard:
    """Tests for sample size (n_train) scorecard generation."""

    METRICS = ["spearman_dist", "spearman_err", "picp_error", "winkler", "outlier_auroc"]
    EXPECTED_SCHEMA = [
        "n_train",
        "k_over_n_ratio",
        "dimension",
        "n_experiments",
        # spearman_dist
        "spearman_dist_plcb_mean", "spearman_dist_plcb_sem",
        "spearman_dist_slcb_mean", "spearman_dist_slcb_sem",
        "spearman_dist_diff_mean", "spearman_dist_pvalue",
        "spearman_dist_cliffs_delta", "spearman_dist_wins",
        "spearman_dist_ties", "spearman_dist_losses",
        # spearman_err
        "spearman_err_plcb_mean", "spearman_err_plcb_sem",
        "spearman_err_slcb_mean", "spearman_err_slcb_sem",
        "spearman_err_diff_mean", "spearman_err_pvalue",
        "spearman_err_cliffs_delta", "spearman_err_wins",
        "spearman_err_ties", "spearman_err_losses",
        # picp_error
        "picp_error_plcb_mean", "picp_error_plcb_sem",
        "picp_error_slcb_mean", "picp_error_slcb_sem",
        "picp_error_diff_mean", "picp_error_pvalue",
        "picp_error_cliffs_delta", "picp_error_wins",
        "picp_error_ties", "picp_error_losses",
        # winkler
        "winkler_plcb_mean", "winkler_plcb_sem",
        "winkler_slcb_mean", "winkler_slcb_sem",
        "winkler_diff_mean", "winkler_pvalue",
        "winkler_cliffs_delta", "winkler_wins",
        "winkler_ties", "winkler_losses",
        # outlier_auroc
        "outlier_auroc_plcb_mean", "outlier_auroc_plcb_sem",
        "outlier_auroc_slcb_mean", "outlier_auroc_slcb_sem",
        "outlier_auroc_diff_mean", "outlier_auroc_pvalue",
        "outlier_auroc_cliffs_delta", "outlier_auroc_wins",
        "outlier_auroc_ties", "outlier_auroc_losses",
    ]

    def test_build_sample_size_scorecard_empty(self):
        """Test that empty DataFrame returns empty scorecard with correct column schema."""
        empty_df = pd.DataFrame()
        sc = build_sample_size_scorecard_dataframe(empty_df)
        assert isinstance(sc, pd.DataFrame)
        assert sc.empty
        assert list(sc.columns) == self.EXPECTED_SCHEMA

    def test_build_sample_size_scorecard_schema_and_grouping(self, mock_summaries_dir: Path):
        """Test scorecard schema, grouping by (n_train, dimension), and k_over_n_ratio values."""
        df = load_summary_records(mock_summaries_dir)
        sc = build_sample_size_scorecard_dataframe(df)

        assert isinstance(sc, pd.DataFrame)
        assert not sc.empty
        assert list(sc.columns) == self.EXPECTED_SCHEMA

        # Check n_train present: 112, 224, All
        n_train_vals = sc["n_train"].unique().tolist()
        assert 112 in n_train_vals
        assert 224 in n_train_vals
        assert "All" in n_train_vals

        # Check n_train=112 rows:
        # In mock data, n_train=112 is used for d=2
        rows_112 = sc[sc["n_train"] == 112]
        dims_112 = rows_112["dimension"].tolist()
        assert 2 in dims_112
        assert "All" in dims_112

        # Check k_over_n_ratio for 112: 28.0 / 112 = 0.25
        for _, r in rows_112.iterrows():
            assert pytest.approx(r["k_over_n_ratio"]) == 0.25

        # Check n_train=224 rows:
        # In mock data, n_train=224 is used for d=16, 32
        rows_224 = sc[sc["n_train"] == 224]
        dims_224 = rows_224["dimension"].tolist()
        assert 16 in dims_224
        assert 32 in dims_224
        assert "All" in dims_224

        # Check k_over_n_ratio for 224: 28.0 / 224 = 0.125
        for _, r in rows_224.iterrows():
            assert pytest.approx(r["k_over_n_ratio"]) == 0.125

        # Check master grand total row: n_train="All", dimension="All"
        grand_total = sc[(sc["n_train"] == "All") & (sc["dimension"] == "All")]
        assert len(grand_total) == 1
        assert grand_total.iloc[0]["n_experiments"] == len(df)
        ratio_all = grand_total.iloc[0]["k_over_n_ratio"]
        assert ratio_all is None or pd.isna(ratio_all)

    def test_build_sample_size_scorecard_custom_sample_sizes(self, tmp_path: Path):
        """Test with varying sample sizes (56, 112, 224) and verify ratios."""
        summary_dir = tmp_path / "custom_n_train_summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)

        sample_sizes = [56, 112, 224]
        dims = [2, 4]
        total_created = 0

        for n in sample_sizes:
            for d in dims:
                mock_data = _make_mock_summary(
                    dimension=d,
                    n_train=n,
                    function_name="sphere",
                    strategy="natural",
                    seed=0,
                    slcb_spearman_dist=0.3,
                    plcb_spearman_dist=0.8,
                    slcb_spearman_err=0.2,
                    plcb_spearman_err=0.6,
                    slcb_picp=0.6,
                    plcb_picp=0.95,
                    slcb_winkler=40.0,
                    plcb_winkler=15.0,
                    slcb_auroc=0.5,
                    plcb_auroc=0.85,
                )
                fn = f"summary_sphere_d{d}_n{n}_s0.json"
                with open(summary_dir / fn, "w", encoding="utf-8") as f:
                    json.dump(mock_data, f)
                total_created += 1

        df = load_summary_records(summary_dir)
        sc = build_sample_size_scorecard_dataframe(df)

        expected_ratios = {56: 0.5, 112: 0.25, 224: 0.125}
        for n, expected_ratio in expected_ratios.items():
            n_rows = sc[sc["n_train"] == n]
            assert not n_rows.empty
            for _, r in n_rows.iterrows():
                assert pytest.approx(r["k_over_n_ratio"]) == expected_ratio

        # Grand total
        grand_total = sc[(sc["n_train"] == "All") & (sc["dimension"] == "All")].iloc[0]
        assert grand_total["n_experiments"] == total_created
        assert grand_total["k_over_n_ratio"] is None or pd.isna(grand_total["k_over_n_ratio"])


class TestUQAblationScorecard:
    """Tests for UQ component-level ablation scorecard generation."""

    EXPECTED_SCHEMA = [
        "estimator",
        "dimension",
        "n_experiments",
        "spearman_dist_mean",
        "spearman_dist_sem",
        "spearman_err_mean",
        "spearman_err_sem",
        "picp_mean",
        "picp_sem",
        "picp_error_mean",
        "picp_error_sem",
        "mpiw_mean",
        "mpiw_sem",
        "winkler_mean",
        "winkler_sem",
        "outlier_auroc_mean",
        "outlier_auroc_sem",
        "spearman_dist_diff_vs_slcb",
        "spearman_dist_pvalue_vs_slcb",
        "spearman_dist_cliffs_delta_vs_slcb",
        "spearman_dist_win_rate_vs_slcb",
        "winkler_diff_vs_slcb",
        "winkler_pvalue_vs_slcb",
        "winkler_cliffs_delta_vs_slcb",
        "winkler_win_rate_vs_slcb",
        "outlier_auroc_diff_vs_slcb",
        "outlier_auroc_pvalue_vs_slcb",
        "outlier_auroc_cliffs_delta_vs_slcb",
        "outlier_auroc_win_rate_vs_slcb",
    ]

    def test_build_uq_ablation_scorecard_empty(self):
        """Empty DataFrame handling returning empty DataFrame with schema."""
        empty_df = pd.DataFrame()
        sc = build_uq_ablation_scorecard_dataframe(empty_df)
        assert isinstance(sc, pd.DataFrame)
        assert sc.empty
        assert list(sc.columns) == self.EXPECTED_SCHEMA

    def test_build_uq_ablation_scorecard_schema_and_methods(self, mock_summaries_dir: Path):
        """Verify that all methods (slcb, rf_fire, prox_a, prox_b, prox_bc, plcb) are represented for each dimension and dimension='All'."""
        df = load_summary_records(mock_summaries_dir)
        sc = build_uq_ablation_scorecard_dataframe(df)

        assert isinstance(sc, pd.DataFrame)
        assert not sc.empty
        assert list(sc.columns) == self.EXPECTED_SCHEMA

        expected_methods = ["slcb", "rf_fire", "prox_a", "prox_b", "prox_bc", "plcb", "shaker_total"]
        dimensions = [2, 16, 32, "All"]

        for method in expected_methods:
            method_rows = sc[sc["estimator"] == method]
            assert not method_rows.empty, f"Method {method} missing from scorecard"
            dim_values = method_rows["dimension"].tolist()
            for d in dimensions:
                assert d in dim_values, f"Dimension {d} missing for method {method}"

    def test_build_uq_ablation_scorecard_paired_comparisons_vs_slcb(self, mock_summaries_dir: Path):
        """Verify that paired comparisons against slcb work correctly."""
        df = load_summary_records(mock_summaries_dir)
        sc = build_uq_ablation_scorecard_dataframe(df)

        # For slcb, diff should be 0 and win rate should be 0.0
        slcb_rows = sc[sc["estimator"] == "slcb"]
        assert not slcb_rows.empty
        for _, r in slcb_rows.iterrows():
            assert pytest.approx(r["spearman_dist_diff_vs_slcb"]) == 0.0
            assert pytest.approx(r["spearman_dist_win_rate_vs_slcb"]) == 0.0
            assert pytest.approx(r["spearman_dist_cliffs_delta_vs_slcb"]) == 0.0
            assert pytest.approx(r["winkler_diff_vs_slcb"]) == 0.0
            assert pytest.approx(r["winkler_win_rate_vs_slcb"]) == 0.0
            assert pytest.approx(r["winkler_cliffs_delta_vs_slcb"]) == 0.0
            assert pytest.approx(r["outlier_auroc_diff_vs_slcb"]) == 0.0
            assert pytest.approx(r["outlier_auroc_win_rate_vs_slcb"]) == 0.0
            assert pytest.approx(r["outlier_auroc_cliffs_delta_vs_slcb"]) == 0.0

        # For plcb, win rates, effect sizes, and p-values are computed
        plcb_rows = sc[sc["estimator"] == "plcb"]
        assert not plcb_rows.empty
        for _, r in plcb_rows.iterrows():
            assert r["spearman_dist_diff_vs_slcb"] > 0.0
            assert r["spearman_dist_win_rate_vs_slcb"] > 0.0
            assert np.isfinite(r["spearman_dist_pvalue_vs_slcb"])
            assert r["spearman_dist_cliffs_delta_vs_slcb"] > 0.0
            assert r["winkler_diff_vs_slcb"] < 0.0
            assert r["winkler_win_rate_vs_slcb"] > 0.0
            assert np.isfinite(r["winkler_pvalue_vs_slcb"])
            assert r["winkler_cliffs_delta_vs_slcb"] < 0.0
            assert r["outlier_auroc_diff_vs_slcb"] > 0.0
            assert r["outlier_auroc_win_rate_vs_slcb"] > 0.0
            assert np.isfinite(r["outlier_auroc_pvalue_vs_slcb"])
            assert r["outlier_auroc_cliffs_delta_vs_slcb"] > 0.0

