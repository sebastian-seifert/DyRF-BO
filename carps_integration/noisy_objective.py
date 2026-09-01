"""CARP-S ObjectiveFunction adapter for Noisy & Heteroscedastic benchmarks."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from ConfigSpace import ConfigurationSpace
from carps.loggers.abstract_logger import AbstractLogger
from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue

from noisy_benchmarks.registry import NoisyBenchmarkRegistry


class CARPSNoisyObjectiveFunction(ObjectiveFunction):
    """Bridges noisy_benchmarks problems into the CARP-S benchmarking suite."""

    def __init__(
        self,
        problem_name: str,
        seed: int = 0,
        loggers: Optional[List[AbstractLogger]] = None,
        **kwargs
    ):
        super().__init__(loggers)
        self.problem_name = problem_name
        self.seed = seed
        self.problem = NoisyBenchmarkRegistry.get_problem(problem_name, seed=seed, **kwargs)
        self._configspace = self.problem.configspace

    @property
    def f_min(self) -> Optional[float]:
        """Known global minimum function value."""
        return self.problem.metadata.f_optimum

    @property
    def configspace(self) -> ConfigurationSpace:
        """Returns ConfigSpace required by CARP-S optimizers."""
        return self._configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluates noisy objective function and populates trial values and ground truth telemetry."""
        starttime = time.time()
        x = self.problem.config_to_vector(trial_info.config)
        res = self.problem.evaluate(x, trial_idx=int(self.n_trials), rng=self.problem.rng)
        endtime = time.time()

        return TrialValue(
            cost=float(res.y_noisy),
            time=max(0.0001, endtime - starttime),
            starttime=starttime,
            endtime=endtime,
            additional_info={
                "y_true": float(res.y_true),
                "sigma_true": float(res.sigma_true),
                "instantaneous_regret": float(res.instantaneous_regret),
                "noise_residual": float(res.noise_residual),
            }
        )
