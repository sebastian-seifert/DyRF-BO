"""Training and test sampling protocols for extrapolation uncertainty quantification.

Implements Latin Hypercube subdomain training samplers, uniform natural test samplers,
and 4-strata Dirichlet / ray-casting test samplers.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.stats.qmc import LatinHypercube

from .qp_convex_hull import ConvexHullProjectionSolver, ProjectionResult


def sample_subdomain_training(
    n_samples: int,
    dimension: int,
    domain_half_width: float = 0.5,
    seed: int | None = None,
    method: str = "lhs",
) -> np.ndarray:
    """Draw training samples strictly within sub-domain [-c, c]^D.

    Parameters
    ----------
    n_samples : int
        Number of training points to sample.
    dimension : int
        Feature space dimensionality D.
    domain_half_width : float, default=0.5
        Sub-domain half width c > 0 such that X in [-c, c]^D.
    seed : int | None, default=None
        Random number generator seed.
    method : str, default='lhs'
        Sampling method ('lhs', 'uniform', or 'random').

    Returns
    -------
    np.ndarray
        Training coordinates of shape (n_samples, dimension).
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    if domain_half_width <= 0:
        raise ValueError(f"domain_half_width must be positive, got {domain_half_width}")

    norm_method = method.lower()
    if norm_method == "lhs":
        sampler = LatinHypercube(d=dimension, seed=seed)
        unit_samples = sampler.random(n=n_samples)
        return -domain_half_width + 2.0 * domain_half_width * unit_samples
    elif norm_method in {"uniform", "random"}:
        rng = np.random.default_rng(seed)
        return rng.uniform(-domain_half_width, domain_half_width, size=(n_samples, dimension))
    else:
        raise ValueError(
            f"Unsupported sampling method '{method}'. Supported methods are 'lhs', 'uniform', 'random'."
        )


def sample_natural_test(
    n_samples: int,
    dimension: int,
    solver: ConvexHullProjectionSolver,
    seed: int | None = None,
) -> Tuple[np.ndarray, ProjectionResult, float]:
    """Draw natural uniform test points on [-1, 1]^D and compute interpolation rate.

    Parameters
    ----------
    n_samples : int
        Number of test points to draw.
    dimension : int
        Feature space dimensionality D.
    solver : ConvexHullProjectionSolver
        Pre-fitted solver for Conv(X_train).
    seed : int | None, default=None
        Random number generator seed.

    Returns
    -------
    tuple[np.ndarray, ProjectionResult, float]
        (X_test, projection_result, p_interp) where p_interp is empirical interpolation rate.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    if solver.D != dimension:
        raise ValueError(
            f"Solver dimension {solver.D} does not match requested dimension {dimension}."
        )

    rng = np.random.default_rng(seed)
    X_test = rng.uniform(-1.0, 1.0, size=(n_samples, dimension))
    projection_result = solver.project(X_test)
    p_interp = float(np.mean(projection_result.is_interpolating))

    return X_test, projection_result, p_interp


def sample_stratified_test(
    n_per_stratum: int,
    dimension: int,
    X_train: np.ndarray,
    solver: ConvexHullProjectionSolver,
    seed: int | None = None,
) -> Tuple[np.ndarray, ProjectionResult, np.ndarray]:
    """Generate test points partitioned into 4 strata via Dirichlet & ray-casting.

    Strata:
    - Stratum 0 (Interpolation): Dirichlet combinations on X_train (alpha = 1_N).
    - Stratum 1 (Near Extrapolation): d_norm in [1e-6, 0.15] via ray-casting.
    - Stratum 2 (Medium Extrapolation): d_norm in (0.15, 0.40] via ray-casting.
    - Stratum 3 (Far Extrapolation): d_norm in (0.40, 0.80] via ray-casting.

    Parameters
    ----------
    n_per_stratum : int
        Number of points per stratum.
    dimension : int
        Feature space dimensionality D.
    X_train : np.ndarray
        Training sample coordinates of shape (N, D).
    solver : ConvexHullProjectionSolver
        Convex hull projection solver.
    seed : int | None, default=None
        Random number generator seed.

    Returns
    -------
    tuple[np.ndarray, ProjectionResult, np.ndarray]
        (X_test, projection_result, strata_labels) where X_test has shape (4 * n_per_stratum, D)
        and strata_labels has shape (4 * n_per_stratum,) with values in {0, 1, 2, 3}.
    """
    if n_per_stratum <= 0:
        raise ValueError(f"n_per_stratum must be positive, got {n_per_stratum}")
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")

    X_train_arr = np.asarray(X_train, dtype=np.float64)
    if X_train_arr.ndim != 2:
        raise ValueError(f"X_train must be 2D array, got {X_train_arr.ndim}D.")
    if X_train_arr.shape[0] == 0:
        raise ValueError("X_train cannot be empty.")
    if X_train_arr.shape[1] != dimension:
        raise ValueError(
            f"X_train dimension {X_train_arr.shape[1]} does not match requested dimension {dimension}."
        )
    if solver.D != dimension:
        raise ValueError(
            f"Solver dimension {solver.D} does not match requested dimension {dimension}."
        )

    N_train = X_train_arr.shape[0]
    rng = np.random.default_rng(seed)

    # 1. Stratum 0 (Pure Interpolation): Dirichlet barycentric combinations
    w_dirichlet = rng.dirichlet(np.ones(N_train), size=n_per_stratum)
    X_stratum_0 = w_dirichlet @ X_train_arr

    # 2. Strata 1-3 (Near, Medium, Far Extrapolation) via ray-casting
    strata_target_bounds = {
        1: (0.00, 0.15),
        2: (0.15, 0.40),
        3: (0.40, 0.80),
    }
    strata_ranges = {
        1: (1e-6, 0.15),
        2: (0.15, 0.40),
        3: (0.40, 0.80),
    }
    collected: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}

    sqrt_D = np.sqrt(dimension)
    max_iters = 200
    iters = 0

    while any(len(collected[s]) < n_per_stratum for s in (1, 2, 3)) and iters < max_iters:
        iters += 1
        for s in (1, 2, 3):
            needed = n_per_stratum - len(collected[s])
            if needed <= 0:
                continue

            t_low, t_high = strata_target_bounds[s]
            batch_size = max(needed * 4, 100)

            # Random anchor x0 in X_train
            idx = rng.integers(0, N_train, size=batch_size)
            x0 = X_train_arr[idx]

            # Isotropic Gaussian unit ray v = z / ||z||_2
            z = rng.standard_normal((batch_size, dimension))
            norm_z = np.linalg.norm(z, axis=1, keepdims=True)
            norm_z = np.maximum(norm_z, 1e-12)
            v = z / norm_z

            # Offset by target distance and clip to domain [-1, 1]^D
            d_target = rng.uniform(t_low, t_high, size=batch_size)
            cand = np.clip(x0 + d_target[:, None] * sqrt_D * v, -1.0, 1.0)

            # Evaluate true normalized distance
            d_norm = solver.distance(cand)

            # Route candidates into matching unfilled strata
            for target_s, (low, high) in strata_ranges.items():
                if len(collected[target_s]) >= n_per_stratum:
                    continue
                if target_s == 1:
                    mask = (d_norm >= low) & (d_norm <= high)
                else:
                    mask = (d_norm > low) & (d_norm <= high)

                matches = cand[mask]
                if len(matches) > 0:
                    avail = n_per_stratum - len(collected[target_s])
                    collected[target_s].extend(matches[:avail])

    for s in (1, 2, 3):
        if len(collected[s]) < n_per_stratum:
            raise RuntimeError(
                f"Could not accumulate {n_per_stratum} points for stratum {s} "
                f"(collected {len(collected[s])}) after {max_iters} iterations."
            )

    X_stratum_1 = np.asarray(collected[1][:n_per_stratum], dtype=np.float64)
    X_stratum_2 = np.asarray(collected[2][:n_per_stratum], dtype=np.float64)
    X_stratum_3 = np.asarray(collected[3][:n_per_stratum], dtype=np.float64)

    X_test = np.vstack([X_stratum_0, X_stratum_1, X_stratum_2, X_stratum_3])
    strata_labels = np.concatenate([
        np.zeros(n_per_stratum, dtype=np.int64),
        np.ones(n_per_stratum, dtype=np.int64),
        np.full(n_per_stratum, 2, dtype=np.int64),
        np.full(n_per_stratum, 3, dtype=np.int64),
    ])

    projection_result = solver.project(X_test)
    return X_test, projection_result, strata_labels
