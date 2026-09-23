"""Dual Uncertainty Quantification Inference Engine.

Evaluates both Standard SMAC3 LCB (via Hutter Law of Total Variance) and
Proximity LCB (Pure Extracted Exploration Term via k-NN OOB residuals) over
the exact same underlying tree ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ


def create_smac_default_rf(
    seed: Optional[int] = None,
    n_trees: int = 10,
    model: Optional[Any] = None,
    use_extra_trees: bool = True,
    **kwargs: Any,
) -> Any:
    """Create an ensemble regressor with native SMAC3 defaults or reuse existing.

    SMAC3 native defaults:
    - n_estimators = n_trees (default 10)
    - max_features = 5/6 (~0.8333)
    - min_samples_split = 3
    - min_samples_leaf = 3
    - criterion = 'squared_error'
    - bootstrap = True
    - oob_score = True
    - random_state = seed

    Parameters
    ----------
    seed : Optional[int], default=None
        Random state seed.
    n_trees : int, default=10
        Number of trees in ensemble.
    model : Optional[Any], default=None
        Existing fitted or unfitted surrogate model. If provided, copies
        hyperparameters or reuses the model.
    use_extra_trees : bool, default=True
        If True, instantiates ExtraTreesRegressor (splitter='random');
        otherwise RandomForestRegressor.
    **kwargs : Any
        Additional hyperparameter overrides.

    Returns
    -------
    Ensemble regressor model instance.
    """
    if model is not None:
        if hasattr(model, "estimators_"):
            return model
        try:
            cloned = clone(model)
            if seed is not None and hasattr(cloned, "random_state"):
                cloned.random_state = seed
            return cloned
        except Exception:
            return model

    model_cls = ExtraTreesRegressor if use_extra_trees else RandomForestRegressor
    params: Dict[str, Any] = {
        "n_estimators": n_trees,
        "max_features": 5.0 / 6.0,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "criterion": "squared_error",
        "bootstrap": True,
        "oob_score": True,
        "random_state": seed,
    }
    params.update(kwargs)
    return model_cls(**params)


@dataclass
class DualUQResult:
    """Container for predictions and dual uncertainty quantifications.

    Supports both dataclass attribute access (e.g. `res.u_slcb`) and
    dictionary-style mapping access (e.g. `res['u_slcb']`).

    Attributes
    ----------
    y_hat : np.ndarray
        Ensemble mean prediction \\hat{\\mu}(x) of shape (M,).
    u_slcb : np.ndarray
        Standard SMAC3 LCB uncertainty (1.96 * \\sqrt{\\sigma^2_{total}}) of shape (M,).
    u_plcb : np.ndarray
        Proximity LCB exploration term max(\\delta_{floor}, |q_{0.025}|) of shape (M,).
    var_between : np.ndarray
        Between-tree prediction variance \\sigma^2_{between}(x) of shape (M,).
    var_within : np.ndarray
        Mean within-tree leaf impurity variance \\sigma^2_{within}(x) of shape (M,).
    var_total : np.ndarray
        Total variance \\sigma^2_{total}(x) = var_between + var_within of shape (M,).
    q_lower : np.ndarray
        Lower residual quantile q_{0.025}^{(k)}(x) of shape (M,).
    delta_floor : np.ndarray
        Point-adaptive exploration floor \\epsilon * 1.96 * local_mae of shape (M,).
    local_mae : np.ndarray
        Local Mean Absolute Error from proximity neighborhood of shape (M,).
    """

    y_hat: np.ndarray
    u_slcb: np.ndarray
    u_plcb: np.ndarray
    var_between: np.ndarray
    var_within: np.ndarray
    var_total: np.ndarray
    q_lower: np.ndarray
    delta_floor: np.ndarray
    local_mae: np.ndarray

    def __getitem__(self, key: str) -> np.ndarray:
        if hasattr(self, key) and not key.startswith("_"):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in DualUQResult.")

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        if hasattr(self, key) and not key.startswith("_"):
            setattr(self, key, value)
        else:
            raise KeyError(f"Cannot set unknown field '{key}' in DualUQResult.")

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key) and not key.startswith("_")

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def keys(self) -> Tuple[str, ...]:
        return (
            "y_hat",
            "u_slcb",
            "u_plcb",
            "var_between",
            "var_within",
            "var_total",
            "q_lower",
            "delta_floor",
            "local_mae",
        )

    def values(self) -> Tuple[np.ndarray, ...]:
        return tuple(getattr(self, k) for k in self.keys())

    def items(self) -> Tuple[Tuple[str, np.ndarray], ...]:
        return tuple((k, getattr(self, k)) for k in self.keys())

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return getattr(self, key)
        return default

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in self.keys()}


class DualUQEvaluator:
    """Dual Uncertainty Quantification Inference Engine.

    Unifies Standard SMAC3 LCB (Hutter Law of Total Variance) and Proximity LCB
    (Pure Extracted Exploration Term) on the identical underlying random forest.

    Parameters
    ----------
    model : Optional[Any], default=None
        Pre-configured or pre-fitted ensemble model. If None, instantiates
        a model with SMAC3 defaults.
    seed : Optional[int], default=None
        Random state seed.
    n_trees : int, default=10
        Number of trees in ensemble.
    k : int, default=28
        Number of nearest neighbors under ensemble Laplacian tree-path metric.
    epsilon : float, default=0.080791
        Exploration floor scaling coefficient.
    topological_decay_lambda : float, default=0.20486
        Exponential decay lambda for topological tree path distance.
    level : float, default=0.95
        Confidence level for quantile estimation (0.95 corresponds to kappa=1.96).
    kappa : float, default=1.96
        Standard deviation / error multiplier for LCB exploration bounds.
    device : str, default='cpu'
        Computation backend for proximity engine ('cpu', 'gpu', or 'auto').
    use_extra_trees : bool, default=True
        Whether to use ExtraTreesRegressor for random splitting.
    **kwargs : Any
        Additional parameters passed to model creation.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        seed: Optional[int] = None,
        n_trees: int = 10,
        k: int = 28,
        epsilon: float = 0.080791,
        topological_decay_lambda: float = 0.20486,
        level: float = 0.95,
        kappa: float = 1.96,
        device: str = "cpu",
        use_extra_trees: bool = True,
        **kwargs: Any,
    ) -> None:
        self.seed = seed
        self.n_trees = n_trees
        self.k = k
        self.epsilon = epsilon
        self.topological_decay_lambda = topological_decay_lambda
        self.level = level
        self.kappa = kappa
        self.device = device
        self.use_extra_trees = use_extra_trees

        if model is not None:
            self.model = create_smac_default_rf(
                seed=seed,
                n_trees=n_trees,
                model=model,
                use_extra_trees=use_extra_trees,
                **kwargs,
            )
        else:
            self.model = create_smac_default_rf(
                seed=seed,
                n_trees=n_trees,
                use_extra_trees=use_extra_trees,
                **kwargs,
            )

        self.uq_model: Optional[GPUProximityRegressionUQ] = None
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        refit: bool = False,
    ) -> "DualUQEvaluator":
        """Fit the underlying forest and initialize the proximity UQ engine.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix of shape (N, D).
        y_train : np.ndarray
            Training target array of shape (N,).
        refit : bool, default=False
            If True, refits the underlying model even if already fitted.

        Returns
        -------
        Self instance for method chaining.
        """
        X_arr = np.asarray(X_train, dtype=np.float64)
        y_arr = np.asarray(y_train, dtype=np.float64).flatten()

        if X_arr.ndim != 2:
            raise ValueError(
                f"X_train must be a 2D array of shape (N, D), got {X_arr.ndim}D array."
            )
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError(
                f"X_train cannot have empty dimensions, got shape {X_arr.shape}."
            )
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Sample size mismatch: X_train has {X_arr.shape[0]} samples, "
                f"but y_train has {y_arr.shape[0]} samples."
            )

        # Fit model if not yet fitted or if refit is requested
        if not hasattr(self.model, "estimators_") or refit:
            self.model.fit(X_arr, y_arr)

        # Build proximity UQ engine wrapping the identical forest
        self.uq_model = GPUProximityRegressionUQ(
            model=self.model,
            X_train=X_arr,
            y_train=y_arr,
            device=self.device,
            topological_decay_lambda=self.topological_decay_lambda,
        )
        self.uq_model.fit()

        self.X_train_ = X_arr
        self.y_train_ = y_arr
        self._is_fitted = True
        return self

    def evaluate(self, X_test: np.ndarray) -> DualUQResult:
        """Evaluate both Standard SMAC3 LCB and Proximity LCB.

        Parameters
        ----------
        X_test : np.ndarray
            Query points of shape (M, D) or (D,).

        Returns
        -------
        DualUQResult
            Container with predictions, variances, quantiles, and uncertainties.
        """
        if not self._is_fitted or self.uq_model is None or self.X_train_ is None:
            raise RuntimeError("DualUQEvaluator must be fitted before calling evaluate().")

        X_test_arr = np.asarray(X_test, dtype=np.float64)
        if X_test_arr.ndim == 1:
            X_test_2d = X_test_arr[np.newaxis, :]
        elif X_test_arr.ndim == 2:
            X_test_2d = X_test_arr
        else:
            raise ValueError(
                f"X_test must be a 1D or 2D array, got {X_test_arr.ndim}D array."
            )

        if X_test_2d.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: query X_test has {X_test_2d.shape[1]} features, "
                f"but training data had {self.X_train_.shape[1]} features."
            )

        # 1. Standard SMAC3 LCB Quantification via Hutter Law of Total Variance
        estimators = self.model.estimators_
        tree_preds = np.column_stack([t.predict(X_test_2d) for t in estimators])
        y_hat = np.mean(tree_preds, axis=1)

        # Between-tree variance: \sigma^2_{between}(x) = 1/B \sum (T_b(x) - \hat{\mu}(x))^2
        var_between = np.mean((tree_preds - y_hat[:, None]) ** 2, axis=1)

        # Within-tree leaf variance: \sigma^2_{within}(x) = 1/B \sum tree_b.tree_.impurity[leaf]
        tree_impurities = np.column_stack([
            np.maximum(0.0, t.tree_.impurity[t.apply(X_test_2d)]) for t in estimators
        ])
        var_within = np.mean(tree_impurities, axis=1)

        # Total variance: \sigma^2_{total}(x) = var_between + var_within
        var_total = np.maximum(0.0, var_between + var_within)

        # U_{SLCB}(x) = kappa * \sqrt{\sigma^2_{total}(x)}
        u_slcb = self.kappa * np.sqrt(var_total)

        # 2. Proximity LCB Quantification (Pure Extracted Exploration Term)
        k_eff = min(self.k, len(self.X_train_))
        y_pred_lwr, _, _, local_mae = self.uq_model.predict_with_intervals(
            X_test_2d,
            n_neighbors=k_eff,
            level=self.level,
            return_mae=True,
        )

        # Lower residual quantile: q_{0.025}^{(k)}(x) = y_{pred_lwr}(x) - \hat{\mu}(x) <= 0
        q_lower = (y_pred_lwr - y_hat).astype(np.float64)
        local_mae_f64 = np.asarray(local_mae, dtype=np.float64)

        # Floor: \delta_{floor}(x) = \epsilon * \kappa * local_mae(x)
        delta_floor = self.epsilon * self.kappa * local_mae_f64

        # Extracted Exploration Uncertainty: U_{PLCB}(x) = max(\delta_{floor}(x), |q_{0.025}^{(k)}(x)|)
        u_plcb = np.maximum(delta_floor, np.abs(q_lower))

        return DualUQResult(
            y_hat=y_hat,
            u_slcb=u_slcb,
            u_plcb=u_plcb,
            var_between=var_between,
            var_within=var_within,
            var_total=var_total,
            q_lower=q_lower,
            delta_floor=delta_floor,
            local_mae=local_mae_f64,
        )
