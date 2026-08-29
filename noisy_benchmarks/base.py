"""Base abstractions and data structures for noisy continuous optimization problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float


@dataclass(frozen=True)
class BenchmarkMetadata:
    """Metadata describing a continuous noisy benchmark function."""

    name: str
    dimension: int
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    f_optimum: float
    x_optimum: Optional[np.ndarray] = None
    is_minimization: bool = True
    noise_type: str = "heteroscedastic"  # "gaussian", "uniform", "cauchy", "heteroscedastic"
    description: str = ""


@dataclass
class EvaluationResult:
    """Carries full ground-truth and noisy observation data for an evaluation."""

    x: np.ndarray
    y_noisy: float
    y_true: float
    sigma_true: float
    noise_residual: float               # y_noisy - y_true
    instantaneous_regret: float         # y_true - f_optimum
    seed: int
    trial_idx: int = 0
    extra_info: Dict[str, Any] = field(default_factory=dict)


class NoisyBenchmarkProblem(ABC):
    """Abstract Base Class for all continuous benchmark problems with explicit noise models."""

    def __init__(self, metadata: BenchmarkMetadata, seed: int = 0):
        self.metadata = metadata
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._configspace: Optional[ConfigurationSpace] = None

    @property
    def configspace(self) -> ConfigurationSpace:
        """Returns standard ConfigSpace representation for SMAC3."""
        if self._configspace is None:
            self._configspace = self._build_configspace()
        return self._configspace

    def _build_configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=self.seed)
        hps = []
        for i in range(self.metadata.dimension):
            lb = float(self.metadata.lower_bounds[i])
            ub = float(self.metadata.upper_bounds[i])
            default = 0.5 * (lb + ub)
            hps.append(Float(f"x{i+1}", bounds=(lb, ub), default=default))
        cs.add(hps)
        return cs

    def config_to_vector(self, config: Configuration | Dict[str, float]) -> np.ndarray:
        return np.array([config[f"x{i+1}"] for i in range(self.metadata.dimension)], dtype=float)

    def vector_to_config(self, x: np.ndarray) -> Configuration:
        values = {f"x{i+1}": float(x[i]) for i in range(self.metadata.dimension)}
        return Configuration(self.configspace, values=values)

    @abstractmethod
    def evaluate_true(self, x: np.ndarray) -> float:
        """Exact deterministic ground truth f_true(x)."""
        pass

    @abstractmethod
    def evaluate_noise_std(self, x: np.ndarray) -> float:
        """Exact noise standard deviation sigma_true(x)."""
        pass

    @abstractmethod
    def sample_noise(self, x: np.ndarray, sigma: float, rng: Optional[np.random.Generator] = None) -> float:
        """Generates random noise instance eps ~ NoiseDist(sigma(x))."""
        pass

    def evaluate(
        self,
        x: np.ndarray,
        trial_idx: int = 0,
        rng: Optional[np.random.Generator] = None
    ) -> EvaluationResult:
        """Evaluates ground truth, computes exact sigma, samples noise, and returns EvaluationResult."""
        rng = rng or self.rng
        y_true = float(self.evaluate_true(x))
        sigma_true = float(self.evaluate_noise_std(x))
        eps = float(self.sample_noise(x, sigma_true, rng=rng))
        y_noisy = y_true + eps
        inst_regret = y_true - self.metadata.f_optimum
        return EvaluationResult(
            x=np.asarray(x, dtype=float).copy(),
            y_noisy=y_noisy,
            y_true=y_true,
            sigma_true=sigma_true,
            noise_residual=eps,
            instantaneous_regret=inst_regret,
            seed=self.seed,
            trial_idx=trial_idx
        )

    def get_smac_target_function(
        self,
        telemetry_recorder: Optional[Callable[[EvaluationResult], None]] = None
    ) -> Callable[[Configuration, Optional[int]], float]:
        """Creates a SMAC3-compatible objective function: target_function(config, seed=0) -> float (y_noisy)."""
        trial_counter = 0

        def target_func(config: Configuration, seed: Optional[int] = None) -> float:
            nonlocal trial_counter
            eval_rng = np.random.default_rng(seed) if seed is not None else self.rng
            x = self.config_to_vector(config)
            res = self.evaluate(x, trial_idx=trial_counter, rng=eval_rng)
            trial_counter += 1
            if telemetry_recorder is not None:
                telemetry_recorder(res)
            return res.y_noisy

        return target_func
