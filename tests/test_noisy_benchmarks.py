"""Comprehensive TDD Test Suite for Standalone Noisy & Heteroscedastic Benchmark Suite."""

import os
import sys
import numpy as np
import pytest

# Ensure DyRF-BO root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noisy_benchmarks.base import BenchmarkMetadata, EvaluationResult, NoisyBenchmarkProblem
from noisy_benchmarks.noise_models import NoiseModel
from noisy_benchmarks.bbob import BBOBNoisyProblem, BBOB_FUNCTION_NAMES
from noisy_benchmarks.hetgp import HetGPProblem
from noisy_benchmarks.registry import NoisyBenchmarkRegistry
from noisy_benchmarks.telemetry import NoisyTelemetryLogger
from noisy_benchmarks.runner import NoisyBOHarness


# =====================================================================
# 1. Noise Model Unit Tests
# =====================================================================

def test_gaussian_noise_properties():
    """Verify Gaussian noise statistics."""
    rng = np.random.default_rng(42)
    samples = [NoiseModel.sample_gaussian(sigma=0.5, rng=rng) for _ in range(2000)]
    assert np.isclose(np.mean(samples), 0.0, atol=0.05)
    assert np.isclose(np.std(samples), 0.5, atol=0.05)


def test_uniform_noise_properties():
    """Verify Uniform noise variance."""
    rng = np.random.default_rng(42)
    sigma = 1.0
    samples = [NoiseModel.sample_uniform(sigma=sigma, rng=rng) for _ in range(5000)]
    assert np.isclose(np.mean(samples), 0.0, atol=0.05)
    # Variance of U(-a, a) where a = sqrt(3)*sigma is a^2 / 3 = sigma^2
    assert np.isclose(np.std(samples), sigma, atol=0.05)


def test_cauchy_outlier_mixture():
    """Verify Cauchy mixture injects heavy-tail outliers."""
    rng = np.random.default_rng(42)
    samples = [
        NoiseModel.sample_bbob_cauchy_mixture(gamma=10.0, p_outlier=0.20, sigma_base=0.01, rng=rng)
        for _ in range(1000)
    ]
    # Extreme outliers exist
    max_abs = max(abs(s) for s in samples)
    assert max_abs > 5.0  # Cauchy produces large outliers


# =====================================================================
# 2. BBOB-Noisy Benchmark Problem Tests
# =====================================================================

@pytest.mark.parametrize("fn", ["sphere", "rosenbrock", "rastrigin", "bent_cigar", "attractive_sector"])
def test_bbob_noisy_known_optimum(fn):
    """Test that analytical functions evaluate to 0.0 at their known global optimum."""
    problem = BBOBNoisyProblem(func_name=fn, dimension=2, noise_model="gaussian", seed=42)
    assert problem.metadata.x_optimum is not None
    x_opt = problem.metadata.x_optimum
    val_true = problem.evaluate_true(x_opt)
    assert np.isclose(val_true, 0.0, atol=1e-5)


def test_bbob_noisy_evaluation_structure():
    """Verify complete evaluation result and ConfigSpace generation."""
    problem = BBOBNoisyProblem(func_name="sphere", dimension=3, noise_model="gaussian", seed=10)
    cs = problem.configspace
    assert len(cs.values()) == 3
    
    cfg = cs.sample_configuration()
    x = problem.config_to_vector(cfg)
    res = problem.evaluate(x, trial_idx=1)
    
    assert isinstance(res, EvaluationResult)
    assert len(res.x) == 3
    assert res.sigma_true > 0
    assert np.isclose(res.y_noisy, res.y_true + res.noise_residual)


# =====================================================================
# 3. hetGP Heteroscedastic Benchmark Problem Tests
# =====================================================================

def test_hetgp_yuan_wahba():
    """Test Yuan-Wahba 1D mathematical formula and input-dependent noise."""
    problem = HetGPProblem(func_name="yuan_wahba", seed=42)
    assert problem.metadata.dimension == 1
    
    # f(0) = 0, sigma(0) = 0.5 + 1.0 * 0.5 * (1 + 1) = 1.5
    f_0 = problem.evaluate_true(np.array([0.0]))
    sigma_0 = problem.evaluate_noise_std(np.array([0.0]))
    assert np.isclose(f_0, 0.0)
    assert np.isclose(sigma_0, 1.5)


def test_hetgp_branin_varying_noise():
    """Test that Branin's global minima reside in distinct noise regimes."""
    problem = HetGPProblem(func_name="branin", seed=42)
    assert problem.metadata.dimension == 2
    
    # Optimum 1: (-pi, 12.275), Optimum 2: (pi, 2.275), Optimum 3: (9.42, 2.47)
    opt1 = np.array([-np.pi, 12.275])
    opt2 = np.array([np.pi, 2.275])
    opt3 = np.array([9.42478, 2.475])
    
    f1 = problem.evaluate_true(opt1)
    f2 = problem.evaluate_true(opt2)
    f3 = problem.evaluate_true(opt3)
    
    # All 3 are global optima with f(x*) = 0.397887
    assert np.isclose(f1, 0.397887, atol=1e-3)
    assert np.isclose(f2, 0.397887, atol=1e-3)
    assert np.isclose(f3, 0.397887, atol=1e-3)
    
    # Noise standard deviation strictly varies across the 3 minima
    s1 = problem.evaluate_noise_std(opt1)
    s2 = problem.evaluate_noise_std(opt2)
    s3 = problem.evaluate_noise_std(opt3)
    
    assert s1 > s2 > s3  # Heteroscedastic noise hierarchy verified!


def test_hetgp_goldstein_price_noise_peak():
    """Test that Goldstein-Price has a noise peak directly at the global optimum."""
    problem = HetGPProblem(func_name="goldstein_price", seed=42)
    opt = np.array([0.0, -1.0])
    off_opt = np.array([1.5, 1.5])
    
    sigma_opt = problem.evaluate_noise_std(opt)
    sigma_off = problem.evaluate_noise_std(off_opt)
    assert sigma_opt > sigma_off


# =====================================================================
# 4. Registry & Telemetry Tests
# =====================================================================

def test_registry_discovery_and_instantiation():
    """Verify registry listing and retrieval."""
    available = NoisyBenchmarkRegistry.list_available_problems()
    assert len(available) > 10
    assert "hetgp_branin_2d" in available
    assert "bbob_noisy_sphere_2d_gaussian" in available
    
    p1 = NoisyBenchmarkRegistry.get_problem("hetgp_branin_2d")
    p2 = NoisyBenchmarkRegistry.get_problem("bbob_noisy_rosenbrock_4d_cauchy")
    assert isinstance(p1, HetGPProblem)
    assert isinstance(p2, BBOBNoisyProblem)


def test_noisy_telemetry_logger(tmp_path):
    """Test telemetry records and exports to JSON, CSV, and Parquet."""
    problem = HetGPProblem(func_name="yuan_wahba", seed=1)
    logger = NoisyTelemetryLogger(problem=problem, optimizer_name="test_opt", output_dir=str(tmp_path))
    
    res1 = problem.evaluate(np.array([0.5]), trial_idx=0)
    res2 = problem.evaluate(np.array([0.81395]), trial_idx=1)
    
    logger.record_evaluation(res1)
    logger.record_evaluation(res2)
    
    df = logger.to_dataframe()
    assert len(df) == 2
    assert "instantaneous_regret" in df.columns
    assert "sampled_incumbent_regret" in df.columns
    
    paths = logger.save("test_run")
    assert os.path.exists(paths["json"])
    assert os.path.exists(paths["csv"])
    assert os.path.exists(paths["parquet"])


# =====================================================================
# 5. Universal SMAC3 & Additive Epistemic Runner Integration Tests
# =====================================================================

def test_smac3_noisy_harness_smoke(tmp_path):
    """Smoke test running SMAC3 standard baseline on noisy Branin."""
    problem = HetGPProblem(func_name="branin", seed=42)
    telemetry = NoisyBOHarness.run_smac3(
        problem=problem,
        approach_name="smac3_baseline",
        n_trials=5,
        seed=42,
        output_dir=str(tmp_path),
    )
    df = telemetry.to_dataframe()
    assert len(df) == 5
    assert df["sampled_incumbent_regret"].iloc[-1] >= 0.0


def test_smac3_custom_uncertainty_smoke(tmp_path):
    """Smoke test running CustomUncertaintyRandomForest (proximity_bc) on BBOB-Noisy Sphere."""
    problem = BBOBNoisyProblem(func_name="sphere", dimension=2, noise_model="gaussian", seed=1)
    telemetry = NoisyBOHarness.run_smac3(
        problem=problem,
        approach_name="proximity_bc",
        n_trials=5,
        seed=1,
        output_dir=str(tmp_path),
    )
    df = telemetry.to_dataframe()
    assert len(df) == 5


def test_additive_epistemic_bo_smoke(tmp_path):
    """Smoke test running Decoupled Additive Epistemic BO on noisy Yuan-Wahba."""
    problem = HetGPProblem(func_name="yuan_wahba", seed=1)
    telemetry = NoisyBOHarness.run_additive_epistemic_bo(
        problem=problem,
        extractor_name="shaker_entropy",
        n_trials=6,
        n_init=3,
        seed=1,
        output_dir=str(tmp_path),
    )
    df = telemetry.to_dataframe()
    assert len(df) == 6
    assert "beta_t" in df.columns
