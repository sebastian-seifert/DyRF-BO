"""BBOB-Noisy Benchmark Suite implementation with exact analytical functions and 3 noise models."""

from __future__ import annotations

from typing import Optional, Literal
import numpy as np

from noisy_benchmarks.base import BenchmarkMetadata, NoisyBenchmarkProblem
from noisy_benchmarks.noise_models import NoiseModel


BBOB_FUNCTION_NAMES = [
    "sphere",
    "rosenbrock",
    "rastrigin",
    "bent_cigar",
    "attractive_sector",
    "schwefel",
    "ellipsoid",
    "discus",
]

BBOB_NOISE_MODELS = Literal["gaussian", "uniform", "cauchy"]


class BBOBNoisyProblem(NoisyBenchmarkProblem):
    """BBOB continuous optimization problem with standard BBOB noise distributions."""

    def __init__(
        self,
        func_name: str = "sphere",
        dimension: int = 2,
        noise_model: BBOB_NOISE_MODELS = "gaussian",
        sigma_add: float = 0.10,
        sigma_mult: float = 0.05,
        p_cauchy_outlier: float = 0.10,
        gamma_cauchy: float = 1.0,
        seed: int = 0,
    ):
        self.func_name = func_name.lower()
        if self.func_name not in BBOB_FUNCTION_NAMES:
            raise ValueError(f"Unknown BBOB function '{func_name}'. Available: {BBOB_FUNCTION_NAMES}")

        self.noise_model = noise_model
        self.sigma_add = sigma_add
        self.sigma_mult = sigma_mult
        self.p_cauchy_outlier = p_cauchy_outlier
        self.gamma_cauchy = gamma_cauchy

        # Domain bounds: standard BBOB is [-5.0, 5.0]^d
        lower_bounds = np.full(dimension, -5.0, dtype=float)
        upper_bounds = np.full(dimension, 5.0, dtype=float)

        f_opt, x_opt = self._get_known_optimum(dimension)

        metadata = BenchmarkMetadata(
            name=f"bbob_noisy_{self.func_name}_{dimension}d_{self.noise_model}",
            dimension=dimension,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            f_optimum=f_opt,
            x_optimum=x_opt,
            is_minimization=True,
            noise_type=self.noise_model,
            description=f"BBOB {self.func_name} ({dimension}D) with {self.noise_model} noise",
        )
        super().__init__(metadata, seed=seed)

    def _get_known_optimum(self, dimension: int) -> tuple[float, Optional[np.ndarray]]:
        if self.func_name == "sphere":
            return 0.0, np.zeros(dimension)
        elif self.func_name == "rosenbrock":
            return 0.0, np.ones(dimension)
        elif self.func_name == "rastrigin":
            return 0.0, np.zeros(dimension)
        elif self.func_name == "bent_cigar":
            return 0.0, np.zeros(dimension)
        elif self.func_name == "attractive_sector":
            return 0.0, np.zeros(dimension)
        elif self.func_name == "schwefel":
            return 0.0, np.full(dimension, 420.968746)
        elif self.func_name in ["ellipsoid", "discus"]:
            return 0.0, np.zeros(dimension)
        return 0.0, None

    def evaluate_true(self, x: np.ndarray) -> float:
        """Exact deterministic ground truth value."""
        x = np.asarray(x, dtype=float)
        d = len(x)

        if self.func_name == "sphere":
            return float(np.sum(x**2))

        elif self.func_name == "rosenbrock":
            return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2))

        elif self.func_name == "rastrigin":
            return float(10.0 * d + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))

        elif self.func_name == "bent_cigar":
            return float(x[0]**2 + 1e6 * np.sum(x[1:]**2))

        elif self.func_name == "attractive_sector":
            s = np.where(x > 0.0, 100.0, 1.0)
            return float(np.sum((s * x)**2))

        elif self.func_name == "schwefel":
            # Standard Schwefel shifted so min = 0
            return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x) + 1e-12))))

        elif self.func_name == "ellipsoid":
            scales = 10.0 ** (6.0 * np.arange(d) / max(1, d - 1))
            return float(np.sum(scales * x**2))

        elif self.func_name == "discus":
            return float(1e6 * x[0]**2 + np.sum(x[1:]**2))

        raise NotImplementedError(f"Function {self.func_name} not implemented")

    def evaluate_noise_std(self, x: np.ndarray) -> float:
        """Computes true noise standard deviation at point x."""
        y_true = self.evaluate_true(x)
        if self.noise_model == "gaussian":
            # Combined additive + multiplicative standard deviation
            return float(np.sqrt(self.sigma_add**2 + (self.sigma_mult * abs(y_true))**2))
        elif self.noise_model == "uniform":
            # Uniform noise standard deviation
            raw_half = self.sigma_add + (self.sigma_mult * abs(y_true)) / len(x)
            return float(raw_half / np.sqrt(3.0))
        elif self.noise_model == "cauchy":
            # Cauchy has undefined theoretical variance; return scale parameter gamma
            return float(self.gamma_cauchy)
        return float(self.sigma_add)

    def sample_noise(
        self,
        x: np.ndarray,
        sigma: float,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """Samples random noise based on the chosen BBOB noise model."""
        r = rng if rng is not None else self.rng
        y_true = self.evaluate_true(x)

        if self.noise_model == "gaussian":
            eps_add = NoiseModel.sample_gaussian(self.sigma_add, rng=r)
            eps_mult = NoiseModel.sample_gaussian(self.sigma_mult * abs(y_true), rng=r)
            return float(eps_add + eps_mult)

        elif self.noise_model == "uniform":
            half_width = self.sigma_add + (self.sigma_mult * abs(y_true)) / len(x)
            return float(r.uniform(low=-half_width, high=half_width))

        elif self.noise_model == "cauchy":
            return float(NoiseModel.sample_bbob_cauchy_mixture(
                gamma=self.gamma_cauchy,
                p_outlier=self.p_cauchy_outlier,
                sigma_base=self.sigma_add,
                rng=r
            ))

        return 0.0
