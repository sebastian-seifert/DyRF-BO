"""Convex Hull Projection Solver via Accelerated Proximal Gradient (FISTA) & SLSQP.

Implements Euclidean distance calculation and simplex weight extraction
for extrapolation uncertainty quantification in Bayesian optimization.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
from scipy.optimize import minimize


class ProjectionResult(dict):
    """Result of convex hull projection containing distance metrics and weights.

    Inherits from dict to allow both dictionary access (res['d_norm'])
    and attribute access (res.d_norm).
    """

    def __init__(
        self,
        weights: np.ndarray,
        x_proj: np.ndarray,
        d_raw: np.ndarray,
        d_norm: np.ndarray,
        d_rel: np.ndarray,
        d_inf: np.ndarray,
        is_interpolating: np.ndarray,
    ) -> None:
        data = {
            "weights": weights,
            "x_proj": x_proj,
            "d_raw": d_raw,
            "d_norm": d_norm,
            "d_rel": d_rel,
            "d_inf": d_inf,
            "is_interpolating": is_interpolating,
        }
        super().__init__(data)
        self.weights = weights
        self.x_proj = x_proj
        self.d_raw = d_raw
        self.d_norm = d_norm
        self.d_rel = d_rel
        self.d_inf = d_inf
        self.is_interpolating = is_interpolating


def project_simplex(V: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Vectorized Euclidean projection of rows onto standard simplex.

    Solves: min_w 0.5 * ||w - v||_2^2 s.t. sum(w) = z, w >= 0.

    Parameters
    ----------
    V : np.ndarray
        Input array of shape (N,) or (M, N).
    z : float, default=1.0
        Simplex sum constraint.

    Returns
    -------
    np.ndarray
        Projected weights of the same shape as V.
    """
    if z <= 0:
        raise ValueError(f"Simplex sum z must be strictly positive, got {z}")

    is_1d = V.ndim == 1
    v_2d = V[np.newaxis, :] if is_1d else V

    M, N = v_2d.shape
    if N == 1:
        w_proj = np.full_like(v_2d, z, dtype=np.float64)
        return w_proj[0] if is_1d else w_proj

    # Sort descending along last axis
    u = np.sort(v_2d, axis=-1)[:, ::-1]
    cssv = np.cumsum(u, axis=-1)
    rho = np.arange(1, N + 1, dtype=v_2d.dtype)

    cond = (u - (cssv - z) / rho) > 0
    num_true = np.count_nonzero(cond, axis=-1, keepdims=True)
    num_true = np.maximum(num_true, 1)
    idx = num_true - 1

    theta = (np.take_along_axis(cssv, idx, axis=-1) - z) / num_true
    W = np.maximum(v_2d - theta, 0.0)

    # Clean small floating point inaccuracies to guarantee exact sum z
    row_sums = np.sum(W, axis=-1, keepdims=True)
    nonzero_mask = row_sums > 0
    W = np.where(nonzero_mask, W * (z / np.where(nonzero_mask, row_sums, 1.0)), z / N)

    return W[0] if is_1d else W


class ConvexHullProjectionSolver:
    """High-performance solver for projecting query points onto Conv(X_train).

    Computes Euclidean, normalized RMS, relative domain, and Chebyshev distances
    from test points to the convex hull of training samples.

    Parameters
    ----------
    X_train : np.ndarray
        Training sample coordinates of shape (N, D).
    method : str, default='fista'
        Optimization method ('fista' or 'slsqp').
    """

    def __init__(self, X_train: np.ndarray, method: str = "fista") -> None:
        X_train_arr = np.asarray(X_train, dtype=np.float64)
        if X_train_arr.ndim != 2:
            raise ValueError(
                f"X_train must be a 2D array of shape (N, D), got {X_train_arr.ndim}D array."
            )
        if X_train_arr.shape[0] == 0 or X_train_arr.shape[1] == 0:
            raise ValueError(
                f"X_train cannot have empty dimensions, got shape {X_train_arr.shape}."
            )

        norm_method = method.lower()
        if norm_method not in {"fista", "slsqp"}:
            raise ValueError(
                f"Unsupported method '{method}'. Supported methods are 'fista' and 'slsqp'."
            )

        self.X_train = X_train_arr
        self.N, self.D = self.X_train.shape
        self.method = norm_method

        # Gram matrix precomputation and Lipschitz constant
        self._gram_cached: Optional[np.ndarray] = None
        if self.D < self.N:
            self.L = float(np.linalg.norm(self.X_train.T @ self.X_train, 2))
        else:
            self._gram_cached = self.X_train @ self.X_train.T
            self.L = float(np.linalg.norm(self._gram_cached, 2))
        self.step_size = 1.0 / self.L if self.L > 1e-12 else 1.0

        # Cached quantities for warm start and fast nearest-neighbor lookups
        self._train_sq_norms = np.sum(self.X_train**2, axis=1)
        self._centroid = np.mean(self.X_train, axis=0, keepdims=True)

    @property
    def _gram(self) -> np.ndarray:
        """Precomputed Gram matrix X_train @ X_train.T, built lazily if D < N."""
        if self._gram_cached is None:
            self._gram_cached = self.X_train @ self.X_train.T
        return self._gram_cached

    @staticmethod
    def project_simplex(V: np.ndarray, z: float = 1.0) -> np.ndarray:
        """Vectorized Euclidean projection of rows onto standard simplex."""
        return project_simplex(V, z=z)

    def _init_weights(self, X_test: np.ndarray) -> np.ndarray:
        """Construct high-quality initial simplex weights for FISTA.

        Chooses per query between nearest training vertex and centroid.
        """
        M = X_test.shape[0]
        if self.N == 1:
            return np.ones((M, 1), dtype=np.float64)

        # Distances to all training vertices: ||x - y||^2 = ||y||^2 - 2 x.y + ||x||^2
        dist_sq = self._train_sq_norms - 2.0 * (X_test @ self.X_train.T)
        nn_idx = np.argmin(dist_sq, axis=1)

        W = np.zeros((M, self.N), dtype=np.float64)
        W[np.arange(M), nn_idx] = 1.0

        # Check if centroid barycenter is closer
        d_centroid_sq = np.sum((X_test - self._centroid) ** 2, axis=1)
        d_nn_sq = (
            np.take_along_axis(dist_sq, nn_idx[:, None], axis=1).squeeze(-1)
            + np.sum(X_test**2, axis=1)
        )
        use_centroid = d_centroid_sq < d_nn_sq
        W[use_centroid] = 1.0 / self.N

        return W

    def _solve_fista(
        self,
        X_test: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Solve convex hull projection using Accelerated Proximal Gradient (FISTA) with adaptive restart."""
        M = X_test.shape[0]
        if self.N == 1:
            return np.ones((M, 1), dtype=np.float64)

        W = self._init_weights(X_test)
        Y = W.copy()
        t = np.ones((M, 1), dtype=np.float64)
        effective_tol = min(tol, max(tol * 1e-2, 1e-10))

        for _ in range(max_iter):
            # Low-rank gradient factorization: (Y @ X_train - X_test) @ X_train.T
            grad = (Y @ self.X_train - X_test) @ self.X_train.T
            W_next = project_simplex(Y - self.step_size * grad, z=1.0)

            diff = np.max(np.abs(W_next - W))
            if diff < effective_tol:
                W = W_next
                break

            # Adaptive restart: zero out momentum if gradient opposes displacement
            restart = np.sum((Y - W_next) * (W_next - W), axis=-1, keepdims=True) > 0
            t_next = np.where(restart, 1.0, 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t)))
            momentum = np.where(restart, 0.0, (t - 1.0) / t_next)

            Y = W_next + momentum * (W_next - W)
            W = W_next
            t = t_next

        return W

    def _solve_slsqp(
        self,
        X_test: np.ndarray,
        max_iter: int = 200,
        tol: float = 1e-7,
    ) -> np.ndarray:
        """Solve convex hull projection point-by-point via scipy SLSQP."""
        M = X_test.shape[0]
        if self.N == 1:
            return np.ones((M, 1), dtype=np.float64)

        P = self._gram
        w_init_all = self._init_weights(X_test)
        bounds = [(0.0, 1.0)] * self.N
        constraints = {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
            "jac": lambda w: np.ones_like(w),
        }
        slsqp_ftol = min(tol, 1e-12)

        W = np.zeros((M, self.N), dtype=np.float64)
        for i in range(M):
            q = -(self.X_train @ X_test[i])
            w0 = w_init_all[i]

            def objective(w: np.ndarray) -> float:
                return float(0.5 * (w @ P @ w) + (q @ w))

            def jacobian(w: np.ndarray) -> np.ndarray:
                return P @ w + q

            res = minimize(
                objective,
                w0,
                method="SLSQP",
                jac=jacobian,
                bounds=bounds,
                constraints=constraints,
                options={"ftol": slsqp_ftol, "maxiter": max_iter},
            )
            W[i] = res.x

        return W

    def project(
        self,
        X_test: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-6,
        method: Optional[str] = None,
    ) -> ProjectionResult:
        """Project query points onto the convex hull of training samples.

        Parameters
        ----------
        X_test : np.ndarray
            Query points of shape (M, D) or (D,).
        max_iter : int, default=50
            Maximum iterations for the optimizer.
        tol : float, default=1e-6
            Convergence tolerance.
        method : Optional[str], default=None
            Optimization backend override ('fista' or 'slsqp').

        Returns
        -------
        ProjectionResult
            Dictionary subclass containing weights, x_proj, d_raw, d_norm,
            d_rel, d_inf, and is_interpolating.
        """
        X_test_arr = np.asarray(X_test, dtype=np.float64)
        if X_test_arr.ndim == 1:
            X_test_2d = X_test_arr[np.newaxis, :]
        elif X_test_arr.ndim == 2:
            X_test_2d = X_test_arr
        else:
            raise ValueError(
                f"X_test must be 1D (D,) or 2D (M, D), got {X_test_arr.ndim}D array."
            )

        if X_test_2d.shape[1] != self.D:
            raise ValueError(
                f"Expected feature dimension {self.D}, got {X_test_2d.shape[1]}."
            )

        active_method = (method or self.method).lower()
        if active_method == "fista":
            weights = self._solve_fista(X_test_2d, max_iter=max_iter, tol=tol)
        elif active_method == "slsqp":
            weights = self._solve_slsqp(X_test_2d, max_iter=max(max_iter, 200), tol=tol)
        else:
            raise ValueError(
                f"Unsupported projection method '{active_method}'. Must be 'fista' or 'slsqp'."
            )

        # Projected points: (M, N) @ (N, D) -> (M, D)
        x_proj = weights @ self.X_train

        # Coordinate residuals: (M, D)
        diff = X_test_2d - x_proj

        # Metric evaluations
        d_raw = np.linalg.norm(diff, axis=1)
        d_norm = d_raw / np.sqrt(self.D)
        d_rel = d_raw / (2.0 * np.sqrt(self.D))
        d_inf = np.max(np.abs(diff), axis=1)
        is_interpolating = d_norm < 1e-6

        return ProjectionResult(
            weights=weights,
            x_proj=x_proj,
            d_raw=d_raw,
            d_norm=d_norm,
            d_rel=d_rel,
            d_inf=d_inf,
            is_interpolating=is_interpolating,
        )

    def distance(self, X_test: np.ndarray) -> np.ndarray:
        """Convenience method returning normalized RMS distance d_norm.

        Parameters
        ----------
        X_test : np.ndarray
            Query points of shape (M, D) or (D,).

        Returns
        -------
        np.ndarray
            Normalized RMS distances d_norm of shape (M,).
        """
        return self.project(X_test)["d_norm"]
