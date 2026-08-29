"""Telemetry logger for noisy Bayesian Optimization benchmarks."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from noisy_benchmarks.base import EvaluationResult, NoisyBenchmarkProblem


class NoisyTelemetryLogger:
    """Records trial-by-trial evaluation metrics, true signals, and regrets."""

    def __init__(
        self,
        problem: NoisyBenchmarkProblem,
        optimizer_name: str = "smac3_baseline",
        output_dir: Optional[str] = None,
    ):
        self.problem = problem
        self.optimizer_name = optimizer_name
        self.output_dir = output_dir
        self.records: List[Dict[str, Any]] = []
        self.best_true_cost: float = float("inf")
        self.start_time = time.time()

    def record_evaluation(
        self,
        eval_result: EvaluationResult,
        surrogate_mu: Optional[float] = None,
        surrogate_var: Optional[float] = None,
        surrogate_u_ep: Optional[float] = None,
        surrogate_u_al: Optional[float] = None,
        beta_t: Optional[float] = None,
    ):
        """Logs an evaluation step and updates the true incumbent regret."""
        y_true = eval_result.y_true
        if y_true < self.best_true_cost:
            self.best_true_cost = y_true

        true_incumbent_regret = self.best_true_cost - self.problem.metadata.f_optimum
        elapsed = time.time() - self.start_time

        record = {
            "trial_idx": eval_result.trial_idx,
            "seed": eval_result.seed,
            "problem_name": self.problem.metadata.name,
            "optimizer_name": self.optimizer_name,
            "x": eval_result.x.tolist(),
            "y_noisy": eval_result.y_noisy,
            "y_true": eval_result.y_true,
            "sigma_true": eval_result.sigma_true,
            "noise_residual": eval_result.noise_residual,
            "instantaneous_regret": eval_result.instantaneous_regret,
            "sampled_incumbent_true": self.best_true_cost,
            "sampled_incumbent_regret": true_incumbent_regret,
            "surrogate_mu": surrogate_mu,
            "surrogate_var": surrogate_var,
            "surrogate_u_ep": surrogate_u_ep,
            "surrogate_u_al": surrogate_u_al,
            "beta_t": beta_t,
            "wallclock_seconds": elapsed,
        }
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts telemetry records to a pandas DataFrame."""
        return pd.DataFrame(self.records)

    def save(self, base_filename: Optional[str] = None) -> Dict[str, str]:
        """Saves telemetry to JSON, Parquet, and CSV."""
        if not self.output_dir:
            return {}
        os.makedirs(self.output_dir, exist_ok=True)

        if base_filename is None:
            base_filename = f"telemetry_{self.optimizer_name}_{self.problem.metadata.name}_seed{self.problem.seed}"

        json_path = os.path.join(self.output_dir, f"{base_filename}.json")
        csv_path = os.path.join(self.output_dir, f"{base_filename}.csv")
        parquet_path = os.path.join(self.output_dir, f"{base_filename}.parquet")

        # 1. JSON
        with open(json_path, "w") as f:
            json.dump(self.records, f, indent=2)

        # 2. DataFrame -> CSV & Parquet
        df = self.to_dataframe()
        df.to_csv(csv_path, index=False)
        # Convert list column to string for parquet serialization
        df_parquet = df.copy()
        if "x" in df_parquet.columns:
            df_parquet["x"] = df_parquet["x"].apply(str)
        df_parquet.to_parquet(parquet_path, index=False)

        return {
            "json": json_path,
            "csv": csv_path,
            "parquet": parquet_path,
        }
