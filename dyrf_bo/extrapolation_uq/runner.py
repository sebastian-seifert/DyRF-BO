"""Execution pipeline and runner for extrapolation uncertainty quantification experiments.

Runs single seed configurations across dimension, sample size, benchmark objective,
and sampling strategy, producing per-point evaluations (saved to Parquet) and summary metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

from .metrics_suite import compute_comprehensive_metrics
from .qp_convex_hull import ConvexHullProjectionSolver
from .samplers import (
    sample_natural_test,
    sample_stratified_test,
    sample_subdomain_training,
)
from .test_objectives import StandardizedObjective, get_objective
from .uq_evaluator import DualUQEvaluator


@dataclass
class ExtrapolationRunConfig:
    """Configuration for an extrapolation uncertainty quantification experiment run.

    Attributes
    ----------
    dimension : int
        Feature space dimensionality D.
    n_train : int
        Number of training observations N in sub-domain [-0.5, 0.5]^D.
    function_name : str
        Benchmark objective name ('sphere', 'rosenbrock', 'rastrigin', 'ackley').
    sampling_strategy : str
        Test sampling protocol ('natural' or 'stratified').
    seed : int
        Random seed for training sampling, test sampling, and model initialization.
    n_test : int, default=10000
        Total test sample size.
    k : int, default=28
        Proximity nearest-neighbor count.
    eps : float, default=0.080791
        Proximity exploration floor coefficient epsilon.
    decay_lambda : float, default=0.20486
        Topological decay parameter lambda.
    n_trees : int, default=10
        Number of trees in ensemble surrogate.
    """

    dimension: int
    n_train: int
    function_name: str
    sampling_strategy: str
    seed: int
    n_test: int = 10000
    k: int = 28
    eps: float = 0.080791
    decay_lambda: float = 0.20486
    n_trees: int = 10


def run_single_experiment(
    config: ExtrapolationRunConfig,
    output_dir: str | Path | None = None,
    save_parquet: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Execute a single extrapolation uncertainty quantification experiment.

    Steps:
    1. Instantiates standardized objective function.
    2. Samples training points X_train ~ U([-0.5, 0.5]^D) via LHS.
    3. Fits objective on X_train to extract normalization stats and y_train_tilde.
    4. Initializes ConvexHullProjectionSolver(X_train).
    5. Generates test set via natural or stratified sampling.
    6. Evaluates ground truth test values y_test_tilde.
    7. Creates and fits DualUQEvaluator with SMAC and Proximity LCB.
    8. Evaluates test set to obtain predictions and uncertainties.
    9. Builds per-point evaluations DataFrame.
    10. Optionally writes DataFrame to Parquet storage.
    11. Computes comprehensive calibration scorecard summary.
    12. Returns (summary_dict, point_df).

    Parameters
    ----------
    config : ExtrapolationRunConfig
        Experiment configuration dataclass.
    output_dir : str | Path | None, default=None
        Target directory or file path for saving Parquet telemetry.
    save_parquet : bool, default=True
        Whether to serialize the resulting DataFrame to Parquet format.

    Returns
    -------
    tuple[dict[str, Any], pd.DataFrame]
        Summary scorecard dictionary and per-point evaluations DataFrame.
    """
    if config.sampling_strategy not in {"natural", "stratified"}:
        raise ValueError(
            f"Unsupported sampling strategy '{config.sampling_strategy}'. "
            "Must be 'natural' or 'stratified'."
        )

    # 1. Objective function
    if isinstance(config.function_name, StandardizedObjective):
        objective = config.function_name
    elif isinstance(config.function_name, str):
        objective = get_objective(config.function_name)
    elif callable(config.function_name):
        objective = StandardizedObjective(config.function_name)
    else:
        raise TypeError(f"Invalid function_name type: {type(config.function_name)}")

    # 2. Seed decoupling via SeedSequence to prevent cross-stream correlation
    ss = np.random.SeedSequence(config.seed)
    seed_train, seed_test, seed_model = [int(s.generate_state(1)[0]) for s in ss.spawn(3)]

    # 3. Training sample generation in sub-domain [-0.5, 0.5]^D
    X_train = sample_subdomain_training(
        n_samples=config.n_train,
        dimension=config.dimension,
        domain_half_width=0.5,
        seed=seed_train,
    )

    # 4. Fit objective and standardize training targets
    objective.fit(X_train)
    y_train_tilde = objective.evaluate(X_train, standardized=True)

    # 5. Convex Hull projection solver
    solver = ConvexHullProjectionSolver(X_train)

    # 6. Test set generation
    if config.sampling_strategy == "natural":
        X_test, proj_res, _ = sample_natural_test(
            n_samples=config.n_test,
            dimension=config.dimension,
            solver=solver,
            seed=seed_test,
        )
        # Assign strata labels based on normalized distance bands
        strata_labels = np.zeros(len(proj_res.d_norm), dtype=np.int64)
        strata_labels[(proj_res.d_norm >= 1e-6) & (proj_res.d_norm <= 0.15)] = 1
        strata_labels[(proj_res.d_norm > 0.15) & (proj_res.d_norm <= 0.40)] = 2
        strata_labels[proj_res.d_norm > 0.40] = 3
    else:  # stratified
        n_per_stratum = config.n_test // 4
        X_test, proj_res, strata_labels = sample_stratified_test(
            n_per_stratum=n_per_stratum,
            dimension=config.dimension,
            X_train=X_train,
            solver=solver,
            seed=seed_test,
        )

    # 7. Evaluate ground truth test values
    y_true = objective.evaluate(X_test, standardized=True)

    # 8. Dual UQ Inference Engine
    evaluator = DualUQEvaluator(
        seed=seed_model,
        n_trees=config.n_trees,
        k=config.k,
        epsilon=config.eps,
        topological_decay_lambda=config.decay_lambda,
    )
    evaluator.fit(X_train, y_train_tilde)

    # 8. Evaluate test set
    uq_res = evaluator.evaluate(X_test)

    # 9. Build per-point DataFrame
    abs_error = np.abs(y_true - uq_res.y_hat)
    point_df = pd.DataFrame(
        {
            "point_id": np.arange(len(X_test), dtype=np.int64),
            "stratum": strata_labels.astype(np.int64),
            "d_norm": np.asarray(proj_res.d_norm, dtype=np.float64),
            "d_rel": np.asarray(proj_res.d_rel, dtype=np.float64),
            "d_inf": np.asarray(proj_res.d_inf, dtype=np.float64),
            "is_interpolating": np.asarray(proj_res.is_interpolating, dtype=bool),
            "y_true": np.asarray(y_true, dtype=np.float64),
            "y_hat": np.asarray(uq_res.y_hat, dtype=np.float64),
            "abs_error": np.asarray(abs_error, dtype=np.float64),
            "u_slcb": np.asarray(uq_res.u_slcb, dtype=np.float64),
            "u_plcb": np.asarray(uq_res.u_plcb, dtype=np.float64),
        }
    )

    # 10. Write to Parquet if requested
    if save_parquet and output_dir is not None:
        out_p = Path(output_dir)
        if out_p.suffix == ".parquet":
            out_p.parent.mkdir(parents=True, exist_ok=True)
            parquet_path = out_p
        else:
            out_p.mkdir(parents=True, exist_ok=True)
            func_name = getattr(objective, "name", str(config.function_name))
            filename = (
                f"extrapolation_{func_name}"
                f"_d{config.dimension}"
                f"_n{config.n_train}"
                f"_{config.sampling_strategy}"
                f"_s{config.seed}.parquet"
            )
            parquet_path = out_p / filename
        point_df.to_parquet(parquet_path, index=False)

    # 11. Comprehensive calibration scorecard
    summary_dict = compute_comprehensive_metrics(
        y_true=y_true,
        y_hat=uq_res.y_hat,
        u_slcb=uq_res.u_slcb,
        u_plcb=uq_res.u_plcb,
        d_norm=proj_res.d_norm,
        strata_labels=strata_labels,
    )
    summary_dict["dimension"] = config.dimension
    summary_dict["n_train"] = config.n_train
    summary_dict["function_name"] = str(config.function_name)
    summary_dict["sampling_strategy"] = config.sampling_strategy
    summary_dict["seed"] = config.seed
    summary_dict["n_test"] = len(point_df)

    # 12. Return
    return summary_dict, point_df
