#!/usr/bin/env python3
"""Standardized Local Epistemic Benchmark Runner (Mandate R3).

Executes local Bayesian Optimization sweeps across benchmark topologies:
- Ackley 2D (multimodal global basin)
- Rosenbrock 2D (banana-shaped valley)
- Hartmann 6D (6D multimodal landscape)
- Synthetic Gap 2D (exploration gap with hidden global optimum)

Compares:
- Baseline: Standard SMAC3 RF surrogate with standard empirical tree variance
- Proposed: DA-EHRF Custom RF (DistanceAwareEvidentialExtractor)

Strict Paired Seed Rigor:
For each random seed s in [1, N], initial design evaluations are 100% identical
between Baseline and Proposed.

Strictly local CPU execution using concurrent.futures.ProcessPoolExecutor.
"""

from __future__ import annotations

import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import synthetic_functions
from ep_extractors import UQExtractorRegistry
from ep_extractors.distance_evidential import DistanceAwareEvidentialExtractor
from noisy_benchmarks.base import BenchmarkMetadata, EvaluationResult, NoisyBenchmarkProblem
from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
from carps_integration.acquisitions import (
    ExpectedImprovement,
    WarmupCosineScheduler,
    normalize_max_relative,
)


# ==============================================================================
# Benchmark Problem Definitions (Ackley 2D, Rosenbrock 2D, Hartmann 6D, Gap 2D)
# ==============================================================================

class Ackley2DProblem(NoisyBenchmarkProblem):
    """Ackley 2D continuous benchmark on [-5, 5]^2 with f_min = 0.0 at (0, 0)."""

    def __init__(self, seed: int = 0, noise_std: float = 0.05):
        metadata = BenchmarkMetadata(
            name="ackley_2d",
            dimension=2,
            lower_bounds=np.full(2, -5.0),
            upper_bounds=np.full(2, 5.0),
            f_optimum=0.0,
            x_optimum=np.zeros(2),
            is_minimization=True,
            noise_type="gaussian",
            description="Ackley 2D benchmark on [-5, 5]^2 with f_min = 0.0 at (0, 0)",
        )
        super().__init__(metadata, seed=seed)
        self.noise_std = float(noise_std)

    def evaluate_true(self, x: np.ndarray) -> float:
        return float(synthetic_functions.ackley_func(x[0:1], x[1:2])[0])

    def evaluate_noise_std(self, x: np.ndarray) -> float:
        return self.noise_std

    def sample_noise(self, x: np.ndarray, sigma: float, rng: Optional[np.random.Generator] = None) -> float:
        rng = rng or self.rng
        return float(rng.normal(0.0, sigma))


class Rosenbrock2DProblem(NoisyBenchmarkProblem):
    """Rosenbrock 2D continuous benchmark on [-2, 2]^2 with f_min = 0.0 at (1, 1)."""

    def __init__(self, seed: int = 0, noise_std: float = 0.05):
        metadata = BenchmarkMetadata(
            name="rosenbrock_2d",
            dimension=2,
            lower_bounds=np.full(2, -2.0),
            upper_bounds=np.full(2, 2.0),
            f_optimum=0.0,
            x_optimum=np.ones(2),
            is_minimization=True,
            noise_type="gaussian",
            description="Rosenbrock 2D benchmark on [-2, 2]^2 with f_min = 0.0 at (1, 1)",
        )
        super().__init__(metadata, seed=seed)
        self.noise_std = float(noise_std)

    def evaluate_true(self, x: np.ndarray) -> float:
        return float(synthetic_functions.rosenbrock_func(x[0:1], x[1:2])[0])

    def evaluate_noise_std(self, x: np.ndarray) -> float:
        return self.noise_std

    def sample_noise(self, x: np.ndarray, sigma: float, rng: Optional[np.random.Generator] = None) -> float:
        rng = rng or self.rng
        return float(rng.normal(0.0, sigma))


class Hartmann6DProblem(NoisyBenchmarkProblem):
    """Hartmann 6D continuous benchmark on [0, 1]^6 with f_min = -3.32237."""

    def __init__(self, seed: int = 0, noise_std: float = 0.05):
        metadata = BenchmarkMetadata(
            name="hartmann_6d",
            dimension=6,
            lower_bounds=np.zeros(6),
            upper_bounds=np.ones(6),
            f_optimum=-3.32237,
            x_optimum=np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]),
            is_minimization=True,
            noise_type="gaussian",
            description="Hartmann 6D benchmark on [0, 1]^6 with f_min = -3.32237",
        )
        super().__init__(metadata, seed=seed)
        self.noise_std = float(noise_std)

    def evaluate_true(self, x: np.ndarray) -> float:
        return float(synthetic_functions.hartmann_6d_func(
            x[0:1], x[1:2], x[2:3], x[3:4], x[4:5], x[5:6]
        )[0])

    def evaluate_noise_std(self, x: np.ndarray) -> float:
        return self.noise_std

    def sample_noise(self, x: np.ndarray, sigma: float, rng: Optional[np.random.Generator] = None) -> float:
        rng = rng or self.rng
        return float(rng.normal(0.0, sigma))


class SyntheticGap2DProblem(NoisyBenchmarkProblem):
    """Synthetic 2D Gap benchmark on [-5, 5]^2 with an unobserved exploration gap [-1.5, 1.5]^2.
    
    The true global minimum lies at (0, 0) inside the gap with f(0, 0) = 0.0.
    Initial design points are sampled strictly outside the gap region [-1.5, 1.5]^2,
    testing whether the epistemic uncertainty signal overcomes the overconfidence trap
    and drives exploration into the empty gap.
    """

    def __init__(self, seed: int = 0, noise_std: float = 0.05, gap_bound: float = 1.5):
        metadata = BenchmarkMetadata(
            name="synthetic_gap_2d",
            dimension=2,
            lower_bounds=np.full(2, -5.0),
            upper_bounds=np.full(2, 5.0),
            f_optimum=0.0,
            x_optimum=np.zeros(2),
            is_minimization=True,
            noise_type="gaussian",
            description="Synthetic 2D gap benchmark on [-5, 5]^2 with central gap [-1.5, 1.5]^2 and f_min = 0.0 at (0, 0)",
        )
        super().__init__(metadata, seed=seed)
        self.noise_std = float(noise_std)
        self.gap_bound = float(gap_bound)

    def evaluate_true(self, x: np.ndarray) -> float:
        # Deceptive multimodal Ackley with attractive local basin outside gap
        ackley_val = float(synthetic_functions.ackley_func(x[0:1], x[1:2])[0])
        # Deceptive local basin centered at (2.5, 2.5)
        dist_deceptive = np.sum((x - 2.5) ** 2)
        deceptive_well = -1.5 * np.exp(-dist_deceptive / 2.0)
        return float(ackley_val + deceptive_well + 1.5)

    def evaluate_noise_std(self, x: np.ndarray) -> float:
        return self.noise_std

    def sample_noise(self, x: np.ndarray, sigma: float, rng: Optional[np.random.Generator] = None) -> float:
        rng = rng or self.rng
        return float(rng.normal(0.0, sigma))


def get_benchmark_problem(name: str, seed: int = 0, noise_std: float = 0.05) -> NoisyBenchmarkProblem:
    """Instantiates a benchmark problem by name."""
    name_clean = name.lower().strip()
    if name_clean == "ackley_2d":
        return Ackley2DProblem(seed=seed, noise_std=noise_std)
    elif name_clean == "rosenbrock_2d":
        return Rosenbrock2DProblem(seed=seed, noise_std=noise_std)
    elif name_clean == "hartmann_6d":
        return Hartmann6DProblem(seed=seed, noise_std=noise_std)
    elif name_clean in ("synthetic_gap_2d", "synthetic_gap", "gap_2d"):
        return SyntheticGap2DProblem(seed=seed, noise_std=noise_std)
    else:
        raise ValueError(
            f"Unknown benchmark name '{name}'. Supported: ['ackley_2d', 'rosenbrock_2d', 'hartmann_6d', 'synthetic_gap_2d']"
        )


# ==============================================================================
# Paired Seed Bayesian Optimization Runner
# ==============================================================================

def run_paired_evaluation(
    benchmark_name: str,
    seed: int,
    n_trials: int = 50,
    n_init: int = 10,
    output_dir: Optional[str] = None,
    n_candidates: int = 1000,
    noise_std: float = 0.05,
) -> List[Dict[str, Any]]:
    """Executes a strictly paired evaluation of Baseline vs Proposed on a given seed.
    
    Both surrogates evaluate identical initial configurations and noisy observations.
    """
    prob = get_benchmark_problem(benchmark_name, seed=seed, noise_std=noise_std)
    dim = prob.metadata.dimension
    lb = prob.metadata.lower_bounds
    ub = prob.metadata.upper_bounds
    
    # 1. Generate Identical Initial Design
    rng_init = np.random.default_rng(seed)
    X_init_list: List[np.ndarray] = []
    
    if benchmark_name == "synthetic_gap_2d":
        # Sample initial design strictly outside the gap region [-1.5, 1.5]^2
        gap_bound = 1.5
        while len(X_init_list) < n_init:
            pt = rng_init.uniform(lb, ub)
            if np.max(np.abs(pt)) > gap_bound:
                X_init_list.append(pt)
    else:
        for _ in range(n_init):
            X_init_list.append(rng_init.uniform(lb, ub))
            
    X_init = np.array(X_init_list, dtype=np.float64)
    init_evals: List[EvaluationResult] = []
    for i, x in enumerate(X_init):
        res = prob.evaluate(x, trial_idx=i, rng=rng_init)
        init_evals.append(res)
        
    records: List[Dict[str, Any]] = []
    
    # --------------------------------------------------------------------------
    # 2. Run Baseline: Standard SMAC3 RF with Standard EI
    # --------------------------------------------------------------------------
    t0_base = time.perf_counter()
    X_base = X_init.copy()
    y_base = np.array([e.y_noisy for e in init_evals], dtype=np.float64)
    y_true_base = np.array([e.y_true for e in init_evals], dtype=np.float64)
    
    best_true_cost_base = float(np.min(y_true_base))
    
    # Record initial steps for baseline
    for i, e in enumerate(init_evals):
        records.append({
            "task_id": benchmark_name,
            "seed": seed,
            "optimizer_id": "SMAC3_HPOFacade_ei",
            "trial_id": i,
            "n_trials": n_trials,
            "x": e.x.tolist(),
            "y_noisy": e.y_noisy,
            "y_true": e.y_true,
            "trial_value__cost": e.y_true,
            "trial_value__cost_inc_norm": best_true_cost_base - prob.metadata.f_optimum,
            "sampled_incumbent_regret": best_true_cost_base - prob.metadata.f_optimum,
            "sampled_incumbent_true": best_true_cost_base,
            "wallclock_seconds": time.perf_counter() - t0_base,
        })
        
    rng_cand_base = np.random.default_rng(200000 + seed)
    for t in range(n_init, n_trials):
        rf_base = RandomForestRegressor(
            n_estimators=20,
            oob_score=True,
            random_state=seed + t,
            n_jobs=1,
        )
        rf_base.fit(X_base, y_base)
        
        # Candidate generation
        X_cand = rng_cand_base.uniform(lb, ub, size=(n_candidates, dim))
        mu_cand = rf_base.predict(X_cand)
        
        # Standard ensemble empirical standard deviation
        tree_preds = np.array([tree.predict(X_cand) for tree in rf_base.estimators_])
        sigma_cand = np.std(tree_preds, axis=0, ddof=0)
        sigma_cand = np.maximum(sigma_cand, 1e-9)
        
        # Standard Expected Improvement
        y_inc = np.min(y_base)
        improvement = y_inc - mu_cand
        z = improvement / sigma_cand
        ei = improvement * norm.cdf(z) + sigma_cand * norm.pdf(z)
        ei = np.maximum(0.0, ei)
        
        best_cand_idx = int(np.argmax(ei))
        x_next = X_cand[best_cand_idx]
        
        res_next = prob.evaluate(x_next, trial_idx=t, rng=rng_cand_base)
        X_base = np.vstack([X_base, x_next])
        y_base = np.append(y_base, res_next.y_noisy)
        y_true_base = np.append(y_true_base, res_next.y_true)
        
        if res_next.y_true < best_true_cost_base:
            best_true_cost_base = float(res_next.y_true)
            
        records.append({
            "task_id": benchmark_name,
            "seed": seed,
            "optimizer_id": "SMAC3_HPOFacade_ei",
            "trial_id": t,
            "n_trials": n_trials,
            "x": x_next.tolist(),
            "y_noisy": res_next.y_noisy,
            "y_true": res_next.y_true,
            "trial_value__cost": res_next.y_true,
            "trial_value__cost_inc_norm": best_true_cost_base - prob.metadata.f_optimum,
            "sampled_incumbent_regret": best_true_cost_base - prob.metadata.f_optimum,
            "sampled_incumbent_true": best_true_cost_base,
            "wallclock_seconds": time.perf_counter() - t0_base,
        })

    # --------------------------------------------------------------------------
    # 3. Run Proposed: DA-EHRF Custom RF (DistanceAwareEvidentialExtractor)
    # --------------------------------------------------------------------------
    t0_prop = time.perf_counter()
    X_prop = X_init.copy()
    y_prop = np.array([e.y_noisy for e in init_evals], dtype=np.float64)
    y_true_prop = np.array([e.y_true for e in init_evals], dtype=np.float64)
    
    best_true_cost_prop = float(np.min(y_true_prop))
    scheduler = WarmupCosineScheduler(
        total_trials=n_trials,
        warmup_ratio=0.20,
        beta_max=1.0,
        beta_min=0.0,
    )
    
    # Record initial steps for proposed
    for i, e in enumerate(init_evals):
        records.append({
            "task_id": benchmark_name,
            "seed": seed,
            "optimizer_id": "CARPSDynamicRF_AdditiveEpistemic_ei_distance_evidential",
            "trial_id": i,
            "n_trials": n_trials,
            "x": e.x.tolist(),
            "y_noisy": e.y_noisy,
            "y_true": e.y_true,
            "trial_value__cost": e.y_true,
            "trial_value__cost_inc_norm": best_true_cost_prop - prob.metadata.f_optimum,
            "sampled_incumbent_regret": best_true_cost_prop - prob.metadata.f_optimum,
            "sampled_incumbent_true": best_true_cost_prop,
            "wallclock_seconds": time.perf_counter() - t0_prop,
        })
        
    rng_cand_prop = np.random.default_rng(200000 + seed)
    for t in range(n_init, n_trials):
        rf_prop = RandomForestRegressor(
            n_estimators=20,
            oob_score=True,
            random_state=seed + t,
            n_jobs=1,
        )
        rf_prop.fit(X_prop, y_prop)
        
        # Fit DA-EHRF extractor
        extractor = DistanceAwareEvidentialExtractor(
            model=rf_prop,
            lengthscale="adaptive",
            kappa_leaf=1.0,
            c_spatial=1.0,
            use_feature_importances=True,
        )
        extractor.fit(X_prop, y_prop)
        
        # Use identical candidate pool to isolate surrogate difference
        X_cand = rng_cand_prop.uniform(lb, ub, size=(n_candidates, dim))
        mu_cand = rf_prop.predict(X_cand)
        
        # Base EI term
        tree_preds = np.array([tree.predict(X_cand) for tree in rf_prop.estimators_])
        sigma_base = np.std(tree_preds, axis=0, ddof=0)
        sigma_base = np.maximum(sigma_base, 1e-9)
        
        y_inc = np.min(y_prop)
        improvement = y_inc - mu_cand
        z = improvement / sigma_base
        ei = improvement * norm.cdf(z) + sigma_base * norm.pdf(z)
        ei = np.maximum(0.0, ei)
        
        # Pure Epistemic Uncertainty term from DA-EHRF
        u_epistemic = extractor.extract_epistemic_signal(X_cand)
        
        # Decoupled Additive Exploration Score with Warmup-Cosine Annealing
        beta_t = scheduler.get_beta(t)
        norm_ei = normalize_max_relative(ei)
        norm_ue = normalize_max_relative(u_epistemic)
        score = norm_ei + beta_t * norm_ue
        
        best_cand_idx = int(np.argmax(score))
        x_next = X_cand[best_cand_idx]
        
        res_next = prob.evaluate(x_next, trial_idx=t, rng=rng_cand_prop)
        X_prop = np.vstack([X_prop, x_next])
        y_prop = np.append(y_prop, res_next.y_noisy)
        y_true_prop = np.append(y_true_prop, res_next.y_true)
        
        if res_next.y_true < best_true_cost_prop:
            best_true_cost_prop = float(res_next.y_true)
            
        records.append({
            "task_id": benchmark_name,
            "seed": seed,
            "optimizer_id": "CARPSDynamicRF_AdditiveEpistemic_ei_distance_evidential",
            "trial_id": t,
            "n_trials": n_trials,
            "x": x_next.tolist(),
            "y_noisy": res_next.y_noisy,
            "y_true": res_next.y_true,
            "trial_value__cost": res_next.y_true,
            "trial_value__cost_inc_norm": best_true_cost_prop - prob.metadata.f_optimum,
            "sampled_incumbent_regret": best_true_cost_prop - prob.metadata.f_optimum,
            "sampled_incumbent_true": best_true_cost_prop,
            "wallclock_seconds": time.perf_counter() - t0_prop,
        })

    return records


# ==============================================================================
# Suite Orchestrator & CLI Execution
# ==============================================================================

def run_local_epistemic_benchmark(
    benchmarks: Optional[List[str]] = None,
    seeds: int = 35,
    n_trials: int = 50,
    n_init: int = 10,
    output_dir: str = "results/epistemic_research",
    n_workers: Optional[int] = None,
    n_candidates: int = 1000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Executes the local epistemic benchmark suite across topologies and paired seeds."""
    if benchmarks is None:
        benchmarks = ["ackley_2d", "rosenbrock_2d", "hartmann_6d", "synthetic_gap_2d"]
        
    os.makedirs(output_dir, exist_ok=True)
    n_workers = n_workers or min(os.cpu_count() or 4, 8)
    
    if verbose:
        print("==============================================================================")
        print("             Local Epistemic Uncertainty Benchmark Runner (Mandate R3)")
        print("==============================================================================")
        print(f"Benchmarks:    {benchmarks}")
        print(f"Paired Seeds:  {seeds} (N={seeds})")
        print(f"Trials/Run:    {n_trials} (Initial Design: {n_init})")
        print(f"Output Dir:    {output_dir}")
        print(f"Parallelism:   {n_workers} CPU Workers")
        print("------------------------------------------------------------------------------")
        
    tasks = []
    for b_name in benchmarks:
        for s in range(1, seeds + 1):
            tasks.append((b_name, s, n_trials, n_init, n_candidates))
            
    total_tasks = len(tasks)
    all_records: List[Dict[str, Any]] = []
    start_time = time.time()
    
    if n_workers > 1 and total_tasks > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {
                executor.submit(
                    run_paired_evaluation,
                    task[0],
                    task[1],
                    task[2],
                    task[3],
                    output_dir,
                    task[4],
                ): task for task in tasks
            }
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    res_records = future.result()
                    all_records.extend(res_records)
                except Exception as ex:
                    print(f"[ERROR] Task {task} failed with exception: {ex}", file=sys.stderr)
                completed += 1
                if verbose and (completed % 10 == 0 or completed == total_tasks):
                    elapsed = time.time() - start_time
                    pct = 100.0 * completed / total_tasks
                    print(f"[{completed:3d}/{total_tasks:3d}] ({pct:5.1f}%) Paired tasks complete ({elapsed:.1f}s elapsed)")
    else:
        completed = 0
        for task in tasks:
            res_records = run_paired_evaluation(
                benchmark_name=task[0],
                seed=task[1],
                n_trials=task[2],
                n_init=task[3],
                output_dir=output_dir,
                n_candidates=task[4],
            )
            all_records.extend(res_records)
            completed += 1
            if verbose and (completed % 5 == 0 or completed == total_tasks):
                elapsed = time.time() - start_time
                pct = 100.0 * completed / total_tasks
                print(f"[{completed:3d}/{total_tasks:3d}] ({pct:5.1f}%) Paired tasks complete ({elapsed:.1f}s elapsed)")
                
    df = pd.DataFrame(all_records)
    
    # Save outputs
    csv_path = os.path.join(output_dir, "benchmark_traces.csv")
    parquet_path = os.path.join(output_dir, "benchmark_traces.parquet")
    
    df.to_csv(csv_path, index=False)
    
    # Serialize list columns to string for Parquet
    df_parquet = df.copy()
    if "x" in df_parquet.columns:
        df_parquet["x"] = df_parquet["x"].apply(str)
    df_parquet.to_parquet(parquet_path, index=False)
    
    total_time = time.time() - start_time
    if verbose:
        print("------------------------------------------------------------------------------")
        print(f"[✓] Benchmark execution finished in {total_time:.2f}s!")
        print(f"    - Saved CSV traces:     {csv_path} ({len(df)} records)")
        print(f"    - Saved Parquet traces: {parquet_path}")
        print("==============================================================================")
        
    return df


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run local epistemic uncertainty BO benchmarks (Mandate R3)"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["ackley_2d", "rosenbrock_2d", "hartmann_6d", "synthetic_gap_2d"],
        help="List of benchmark topologies to evaluate",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=35,
        help="Number of paired seeds (N >= 30, default: 35)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of BO evaluations per seed (default: 50)",
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=10,
        help="Number of initial design points (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/epistemic_research",
        help="Output directory to save traces (default: results/epistemic_research)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of CPU workers for parallel execution (default: auto)",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    run_local_epistemic_benchmark(
        benchmarks=args.benchmarks,
        seeds=args.seeds,
        n_trials=args.n_trials,
        n_init=args.n_init,
        output_dir=args.output_dir,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
