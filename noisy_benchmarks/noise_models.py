"""Noise distribution samplers and models for continuous optimization benchmarks."""

from __future__ import annotations

from typing import Optional
import numpy as np


class NoiseModel:
    """Noise generator implementing Gaussian, Uniform, Cauchy, and Heteroscedastic noise."""

    @staticmethod
    def sample_gaussian(
        sigma: float,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """Sample zero-mean Gaussian noise: eps ~ N(0, sigma^2)."""
        if sigma <= 0:
            return 0.0
        r = rng if rng is not None else np.random.default_rng()
        return float(r.normal(loc=0.0, scale=sigma))

    @staticmethod
    def sample_uniform(
        sigma: float,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """Sample zero-mean Uniform noise with standard deviation sigma: eps ~ U(-sqrt(3)*sigma, sqrt(3)*sigma)."""
        if sigma <= 0:
            return 0.0
        r = rng if rng is not None else np.random.default_rng()
        half_width = np.sqrt(3.0) * sigma
        return float(r.uniform(low=-half_width, high=half_width))

    @staticmethod
    def sample_cauchy(
        gamma: float,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """Sample standard Cauchy noise scaled by gamma: eps ~ Cauchy(0, gamma)."""
        if gamma <= 0:
            return 0.0
        r = rng if rng is not None else np.random.default_rng()
        # Standard Cauchy is standard_normal / standard_normal
        u1 = r.normal(0.0, 1.0)
        u2 = r.normal(0.0, 1.0)
        if u2 == 0:
            u2 = 1e-12
        return float(gamma * (u1 / u2))

    @staticmethod
    def sample_bbob_cauchy_mixture(
        gamma: float,
        p_outlier: float = 0.10,
        sigma_base: float = 0.01,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """Sample mixture of baseline Gaussian noise + Cauchy outliers."""
        r = rng if rng is not None else np.random.default_rng()
        if r.random() < p_outlier:
            return float(NoiseModel.sample_cauchy(gamma, rng=r))
        else:
            return float(NoiseModel.sample_gaussian(sigma_base, rng=r))
