"""Standalone Noisy & Heteroscedastic Bayesian Optimization Benchmark Suite.

Independent from CARP-S, provides BBOB-Noisy and hetGP benchmark suites with
seamless integration for SMAC3, CustomUncertaintyRandomForest, and Decoupled Additive Epistemic BO.
"""

from noisy_benchmarks.base import (
    BenchmarkMetadata,
    EvaluationResult,
    NoisyBenchmarkProblem,
)
from noisy_benchmarks.bbob import BBOBNoisyProblem
from noisy_benchmarks.hetgp import HetGPProblem
from noisy_benchmarks.registry import NoisyBenchmarkRegistry
from noisy_benchmarks.runner import NoisyBOHarness
from noisy_benchmarks.telemetry import NoisyTelemetryLogger

__all__ = [
    "BenchmarkMetadata",
    "EvaluationResult",
    "NoisyBenchmarkProblem",
    "BBOBNoisyProblem",
    "HetGPProblem",
    "NoisyBenchmarkRegistry",
    "NoisyBOHarness",
    "NoisyTelemetryLogger",
]
