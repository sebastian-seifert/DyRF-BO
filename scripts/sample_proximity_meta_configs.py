#!/usr/bin/env python3
"""Sobol Sequence Sampler for Proximity Lower Bound Meta-HPO Hyperparameters.

Samples N = 50 configurations using a quasi-random Sobol sequence (scipy.stats.qmc.Sobol)
based on the Antonov and Saleev (1979) Gray-code formulation with Joe and Kuo (2008)
direction numbers for fast bitwise generation, with a fixed seed for exact reproducibility across:
- k in [5, 30] (nearest neighbors for proximity uncertainty, integer inclusive)
- lambda in [0.2, 2.0] (decay parameter for proximity_b kernel, float)
- eps in [0.02, 0.20] (lower bound exploration floor factor, float)

Outputs configurations to JSON (default: results/sweep_proximity_meta_hpo/sobol_configs.json).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats.qmc import Sobol


def sample_proximity_meta_configs(
    n_configs: int = 50,
    seed: int = 42,
    k_bounds: Tuple[int, int] = (5, 30),
    lambda_bounds: Tuple[float, float] = (0.2, 2.0),
    eps_bounds: Tuple[float, float] = (0.02, 0.20),
    output_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Samples N configurations using scrambled Sobol quasi-random sequence.

    Args:
        n_configs: Number of hyperparameter configurations to generate (N=50).
        seed: Random seed for reproducible scrambled Sobol sequence.
        k_bounds: Tuple of (min, max) for integer neighbor count k (inclusive).
        lambda_bounds: Tuple of (min, max) for float decay lambda.
        eps_bounds: Tuple of (min, max) for float exploration floor eps.
        output_file: Optional file path to persist sampled configurations to JSON.

    Returns:
        List of configuration dictionaries.
    """
    k_min, k_max = int(k_bounds[0]), int(k_bounds[1])
    lam_min, lam_max = float(lambda_bounds[0]), float(lambda_bounds[1])
    eps_min, eps_max = float(eps_bounds[0]), float(eps_bounds[1])

    # Suppress non-power-of-2 balance property warning from scipy
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        sampler = Sobol(d=3, scramble=True, seed=seed)
        samples = sampler.random(n=n_configs)  # Shape (n_configs, 3) in [0, 1)

    configs: List[Dict[str, Any]] = []

    for i in range(n_configs):
        s_k, s_lam, s_eps = samples[i]

        # Uniform partition over [k_min, k_max] inclusive
        k_val = int(np.floor(s_k * (k_max - k_min + 1))) + k_min
        k_val = max(k_min, min(k_max, k_val))

        # Continuous scaling for lambda and eps rounded to 4 decimals
        lam_val = round(float(lam_min + s_lam * (lam_max - lam_min)), 4)
        lam_val = max(lam_min, min(lam_max, lam_val))

        eps_val = round(float(eps_min + s_eps * (eps_max - eps_min)), 4)
        eps_val = max(eps_min, min(eps_max, eps_val))

        cfg_id = f"cfg_{i + 1:02d}"
        optimizer_id = f"SMAC20_ProximityLCB_{cfg_id}"

        cfg_dict: Dict[str, Any] = {
            "config_id": cfg_id,
            "optimizer_id": optimizer_id,
            "k": k_val,
            "decay_lambda": lam_val,
            "lambda": lam_val,
            "eps": eps_val,
        }
        configs.append(cfg_dict)

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(configs, f, indent=2)

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample Sobol configurations for Proximity Lower Bound Meta-HPO."
    )
    parser.add_argument(
        "--n-configs",
        "-n",
        type=int,
        default=50,
        help="Number of configurations to sample (default: 50)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for Sobol sequence (default: 42)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/sweep_proximity_meta_hpo/sobol_configs.json",
        help="Path to output JSON file (default: results/sweep_proximity_meta_hpo/sobol_configs.json)",
    )
    args = parser.parse_args()

    configs = sample_proximity_meta_configs(
        n_configs=args.n_configs,
        seed=args.seed,
        output_file=args.output,
    )
    print(f"Successfully generated {len(configs)} Sobol configurations at {args.output}")


if __name__ == "__main__":
    main()
