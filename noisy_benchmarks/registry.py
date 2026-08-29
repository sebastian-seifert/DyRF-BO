"""Unified Registry for BBOB-Noisy and hetGP benchmark problems."""

from __future__ import annotations

from typing import Dict, List, Optional
from noisy_benchmarks.base import NoisyBenchmarkProblem
from noisy_benchmarks.bbob import BBOBNoisyProblem, BBOB_FUNCTION_NAMES
from noisy_benchmarks.hetgp import HetGPProblem


class NoisyBenchmarkRegistry:
    """Central registry to discover and instantiate noisy and heteroscedastic benchmark problems."""

    @staticmethod
    def list_available_problems() -> List[str]:
        """Returns a list of all pre-configured benchmark problem identifiers."""
        problems = [
            # hetGP Suite
            "hetgp_yuan_wahba_1d",
            "hetgp_branin_2d",
            "hetgp_goldstein_price_2d",
            "hetgp_sinusoid_2d",
            "hetgp_sinusoid_4d",
            "hetgp_sinusoid_8d",
        ]
        # BBOB-Noisy Suite
        for fn in ["sphere", "rosenbrock", "rastrigin", "bent_cigar", "attractive_sector", "schwefel"]:
            for d in [2, 4]:
                for noise in ["gaussian", "uniform", "cauchy"]:
                    problems.append(f"bbob_noisy_{fn}_{d}d_{noise}")
        return sorted(problems)

    @staticmethod
    def get_problem(
        name: str,
        seed: int = 0,
        **kwargs
    ) -> NoisyBenchmarkProblem:
        """Instantiates a benchmark problem by name."""
        name_lower = name.lower()

        # 1. hetGP benchmarks
        if name_lower.startswith("hetgp_"):
            parts = name_lower.replace("hetgp_", "").split("_")
            if "yuan_wahba" in name_lower:
                return HetGPProblem(func_name="yuan_wahba", dimension=1, seed=seed, **kwargs)
            elif "branin" in name_lower:
                return HetGPProblem(func_name="branin", dimension=2, seed=seed, **kwargs)
            elif "goldstein_price" in name_lower:
                return HetGPProblem(func_name="goldstein_price", dimension=2, seed=seed, **kwargs)
            elif "sinusoid" in name_lower:
                dim = int(parts[-1].replace("d", "")) if parts[-1].endswith("d") else 2
                return HetGPProblem(func_name="sinusoid", dimension=dim, seed=seed, **kwargs)

        # 2. BBOB-Noisy benchmarks
        elif name_lower.startswith("bbob_noisy_"):
            parts = name_lower.replace("bbob_noisy_", "").split("_")
            # Expected format: bbob_noisy_{fn}_{dim}d_{noise}
            noise_type = parts[-1] if parts[-1] in ["gaussian", "uniform", "cauchy"] else "gaussian"
            dim = 2
            fn_parts = []
            for p in parts[:-1]:
                if p.endswith("d") and p[:-1].isdigit():
                    dim = int(p[:-1])
                else:
                    fn_parts.append(p)
            fn_name = "_".join(fn_parts)
            return BBOBNoisyProblem(
                func_name=fn_name,
                dimension=dim,
                noise_model=noise_type,
                seed=seed,
                **kwargs
            )

        raise ValueError(f"Unknown benchmark problem '{name}'. Use NoisyBenchmarkRegistry.list_available_problems() to inspect.")
