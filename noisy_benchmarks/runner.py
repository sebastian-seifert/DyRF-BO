"""Unified execution harness connecting SMAC3 and DyRF-BO extractors to noisy benchmarks."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ConfigSpace import Configuration

# SMAC3 imports
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
from smac.scenario import Scenario

# DyRF-BO custom uncertainty surrogate
from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
from ep_extractors import UQExtractorRegistry
from noisy_benchmarks.base import NoisyBenchmarkProblem
from noisy_benchmarks.telemetry import NoisyTelemetryLogger


class NoisyBOHarness:
    """Universal benchmarking runner for noisy & heteroscedastic Bayesian Optimization."""

    @staticmethod
    def run_smac3(
        problem: NoisyBenchmarkProblem,
        approach_name: str = "smac3_baseline",
        n_trials: int = 50,
        seed: int = 0,
        output_dir: Optional[str] = None,
        logging_level: int = 40,  # Logging.ERROR to suppress SMAC3 verbose logs
    ) -> NoisyTelemetryLogger:
        """Executes SMAC3 on a noisy benchmark problem using either standard RF or CustomUncertaintyRandomForest."""
        telemetry = NoisyTelemetryLogger(
            problem=problem,
            optimizer_name=approach_name,
            output_dir=output_dir
        )
        target_func = problem.get_smac_target_function(telemetry_recorder=telemetry.record_evaluation)

        from pathlib import Path
        out_path = Path(output_dir) if output_dir else Path("smac3_output")
        out_path.mkdir(parents=True, exist_ok=True)

        # Configure SMAC3 scenario (deterministic=False for stochastic targets)
        scenario = Scenario(
            configspace=problem.configspace,
            deterministic=False,
            n_trials=n_trials,
            seed=seed,
            output_directory=out_path,
        )

        # Model resolution
        if approach_name in ["smac3_baseline", "smac3", "baseline", "standard_rf"]:
            model = None  # Default SMAC3 RandomForest
        else:
            # Custom uncertainty surrogate with registered extractor
            model = CustomUncertaintyRandomForest(
                uncertainty_func=approach_name,
                configspace=problem.configspace,
                seed=seed,
            )

        facade = HPOFacade(
            scenario=scenario,
            target_function=target_func,
            model=model,
            overwrite=True,
            logging_level=logging_level,
        )

        incumbent = facade.optimize()
        if output_dir:
            telemetry.save()

        return telemetry

    @staticmethod
    def run_additive_epistemic_bo(
        problem: NoisyBenchmarkProblem,
        extractor_name: str = "proximity_bc",
        n_trials: int = 50,
        n_init: int = 10,
        beta_max: float = 1.0,
        warmup_ratio: float = 0.20,
        seed: int = 0,
        output_dir: Optional[str] = None,
    ) -> NoisyTelemetryLogger:
        """Executes Decoupled Additive Epistemic BO: Acq(x) = EI(x) + beta(t) * U_ep(x)."""
        telemetry = NoisyTelemetryLogger(
            problem=problem,
            optimizer_name=f"additive_{extractor_name}",
            output_dir=output_dir
        )

        rng = np.random.default_rng(seed)
        cs = problem.configspace
        dim = problem.metadata.dimension

        # 1. Initial Design (Uniform / Sobol)
        X_hist_list = []
        y_noisy_list = []

        for i in range(n_init):
            cfg = cs.sample_configuration()
            x = problem.config_to_vector(cfg)
            res = problem.evaluate(x, trial_idx=i, rng=rng)
            X_hist_list.append(x)
            y_noisy_list.append(res.y_noisy)
            telemetry.record_evaluation(res, beta_t=1.0)

        # 2. Active Optimization Loop
        from scipy.stats import norm

        for t in range(n_init, n_trials):
            X_train = np.array(X_hist_list)
            y_train = np.array(y_noisy_list)

            # Fit custom uncertainty model
            surrogate = CustomUncertaintyRandomForest(
                uncertainty_func=extractor_name,
                configspace=cs,
                seed=seed + t,
            )
            surrogate.train(X_train, y_train.reshape(-1, 1))

            # Compute Warmup-Cosine Beta Schedule
            progress = t / max(1, n_trials)
            if progress < warmup_ratio:
                beta_t = beta_max
            else:
                rel = (progress - warmup_ratio) / max(1e-12, 1.0 - warmup_ratio)
                beta_t = beta_max * 0.5 * (1.0 + np.cos(np.pi * rel))

            # Candidate Generation (Sobol / Random samples)
            n_cands = 2000
            cand_configs = [cs.sample_configuration() for _ in range(n_cands)]
            X_cand = np.array([problem.config_to_vector(c) for c in cand_configs])

            # Predict Mean and Custom Uncertainty
            mean_pred, var_pred = surrogate.predict(X_cand)
            mean_pred = mean_pred.flatten()
            var_pred = var_pred.flatten()
            sigma_pred = np.sqrt(np.maximum(1e-12, var_pred))

            # Incumbent so far in training
            y_best = np.min(y_train)

            # Compute Expected Improvement (Standard EI)
            improvement = y_best - mean_pred
            z = improvement / sigma_pred
            ei = improvement * norm.cdf(z) + sigma_pred * norm.pdf(z)
            ei = np.maximum(0.0, ei)

            # Compute Custom Epistemic Uncertainty
            # For CustomUncertaintyRandomForest, sigma_pred is already U_ep(x)
            u_ep = sigma_pred

            # Min-Max Normalization of Acquisition terms
            norm_ei = (ei - np.min(ei)) / (np.max(ei) - np.min(ei) + 1e-12)
            norm_uep = (u_ep - np.min(u_ep)) / (np.max(u_ep) - np.min(u_ep) + 1e-12)

            # Additive Acquisition Score
            acq_score = norm_ei + beta_t * norm_uep
            best_cand_idx = int(np.argmax(acq_score))
            x_next = X_cand[best_cand_idx]

            # Evaluate next point
            res = problem.evaluate(x_next, trial_idx=t, rng=rng)
            X_hist_list.append(x_next)
            y_noisy_list.append(res.y_noisy)

            telemetry.record_evaluation(
                res,
                surrogate_mu=float(mean_pred[best_cand_idx]),
                surrogate_var=float(var_pred[best_cand_idx]),
                surrogate_u_ep=float(u_ep[best_cand_idx]),
                beta_t=float(beta_t),
            )

        if output_dir:
            telemetry.save()

        return telemetry
