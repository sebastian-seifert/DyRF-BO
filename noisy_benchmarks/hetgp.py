"""hetGP Heteroscedastic Benchmark Suite with exact closed-form f_true(x) and sigma_true(x).

Implements canonical heteroscedastic testbeds (Yuan-Wahba 1D, Heteroscedastic Branin 2D,
Heteroscedastic Goldstein-Price 2D, and Scalable 1D-15D Heteroscedastic Sinusoid).
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from noisy_benchmarks.base import BenchmarkMetadata, NoisyBenchmarkProblem
from noisy_benchmarks.noise_models import NoiseModel


class HetGPProblem(NoisyBenchmarkProblem):
    """Heteroscedastic continuous optimization problem with input-dependent noise sigma(x)."""

    def __init__(
        self,
        func_name: str = "branin",
        dimension: int = 2,
        sigma_base: float = 0.50,
        noise_amplitude: float = 1.0,
        seed: int = 0,
    ):
        self.func_name = func_name.lower()
        self.sigma_base = sigma_base
        self.noise_amplitude = noise_amplitude

        if self.func_name == "yuan_wahba":
            dimension = 1
            lower_bounds = np.array([0.0], dtype=float)
            upper_bounds = np.array([1.0], dtype=float)
            f_opt = float(2.0 * np.sin(4.0))
            x_opt = np.array([1.0], dtype=float)
            desc = "Yuan-Wahba 1D with cos(4x) heteroscedastic noise"

        elif self.func_name == "branin":
            dimension = 2
            lower_bounds = np.array([-5.0, 0.0], dtype=float)
            upper_bounds = np.array([10.0, 15.0], dtype=float)
            f_opt = 0.397887
            x_opt = np.array([-np.pi, 12.275])
            desc = "Heteroscedastic Branin 2D with x1-dependent noise"

        elif self.func_name in ["goldstein_price", "goldsteinprice"]:
            self.func_name = "goldstein_price"
            dimension = 2
            lower_bounds = np.array([-2.0, -2.0], dtype=float)
            upper_bounds = np.array([2.0, 2.0], dtype=float)
            f_opt = float(np.log10(3.0))
            x_opt = np.array([0.0, -1.0], dtype=float)
            desc = "Heteroscedastic Goldstein-Price 2D with noise peak at optimum"

        elif self.func_name == "sinusoid":
            lower_bounds = np.full(dimension, -5.0, dtype=float)
            upper_bounds = np.full(dimension, 5.0, dtype=float)
            f_opt = 0.0
            x_opt = np.zeros(dimension)
            desc = f"Scalable {dimension}D Heteroscedastic Sinusoid"

        else:
            raise ValueError(f"Unknown hetGP function '{func_name}'. Available: yuan_wahba, branin, goldstein_price, sinusoid")

        metadata = BenchmarkMetadata(
            name=f"hetgp_{self.func_name}_{dimension}d",
            dimension=dimension,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            f_optimum=f_opt,
            x_optimum=x_opt,
            is_minimization=True,
            noise_type="heteroscedastic",
            description=desc,
        )
        super().__init__(metadata, seed=seed)

    def evaluate_true(self, x: np.ndarray) -> float:
        """Exact deterministic ground truth."""
        x = np.asarray(x, dtype=float)

        if self.func_name == "yuan_wahba":
            # f(x) = 2 * sin(4x)
            return float(2.0 * np.sin(4.0 * x[0]))

        elif self.func_name == "branin":
            # Standard Branin formula
            x1, x2 = float(x[0]), float(x[1])
            a = 1.0
            b = 5.1 / (4.0 * np.pi**2)
            c = 5.0 / np.pi
            r = 6.0
            s = 10.0
            t = 1.0 / (8.0 * np.pi)
            return float(a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1.0 - t) * np.cos(x1) + s)

        elif self.func_name == "goldstein_price":
            # Standard Goldstein-Price formula (in log10 scale)
            x1, x2 = float(x[0]), float(x[1])
            part1 = 1.0 + (x1 + x2 + 1.0)**2 * (19.0 - 14.0 * x1 + 3.0 * x1**2 - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2**2)
            part2 = 30.0 + (2.0 * x1 - 3.0 * x2)**2 * (18.0 - 32.0 * x1 + 12.0 * x1**2 + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2**2)
            raw_val = part1 * part2
            # Use log10 transformation for numerical stability
            return float(np.log10(max(1e-12, raw_val)))

        elif self.func_name == "sinusoid":
            d = len(x)
            return float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0))

        raise NotImplementedError(f"Function {self.func_name} not implemented")

    def evaluate_noise_std(self, x: np.ndarray) -> float:
        """Computes true input-dependent noise standard deviation sigma_true(x)."""
        x = np.asarray(x, dtype=float)

        if self.func_name == "yuan_wahba":
            # sigma(x) = sigma_base * 0.5 * (1 + cos(4x))
            return float(self.sigma_base + self.noise_amplitude * 0.5 * (1.0 + np.cos(4.0 * x[0])))

        elif self.func_name == "branin":
            # Noise decays exponentially from left (x1 = -5) to right (x1 = 10)
            # High noise at optimum 1 (-pi, 12.275), medium at optimum 2 (pi, 2.275), low at optimum 3 (9.42, 2.47)
            norm_x1 = (x[0] + 5.0) / 15.0
            return float(self.sigma_base + self.noise_amplitude * np.exp(-2.0 * norm_x1))

        elif self.func_name == "goldstein_price":
            # Gaussian noise peak located directly at global optimum (0, -1)
            dist_to_opt_sq = (x[0] - 0.0)**2 + (x[1] - (-1.0))**2
            noise_peak = np.exp(-0.5 * dist_to_opt_sq / (0.5**2))
            return float(self.sigma_base + self.noise_amplitude * noise_peak)

        elif self.func_name == "sinusoid":
            # Noise increases with distance from center
            norm_dist = np.mean(np.abs(x) / 5.0)
            return float(self.sigma_base + self.noise_amplitude * norm_dist)

        return float(self.sigma_base)

    def sample_noise(
        self,
        x: np.ndarray,
        sigma: float,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """Sample heteroscedastic Gaussian noise: eps ~ N(0, sigma(x)^2)."""
        return NoiseModel.sample_gaussian(sigma=sigma, rng=rng)
