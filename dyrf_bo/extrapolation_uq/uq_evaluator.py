"""Multi-UQ and Dual Uncertainty Quantification Inference Engine.

Evaluates candidate uncertainty quantification methods across tree ensembles:
1. Hutter Total: 1.96 * sqrt(var_between + var_within) [Alias: u_slcb]
2. Hutter Between: 1.96 * sqrt(var_between) (ddof=1)
3. Hutter Within: 1.96 * sqrt(var_within)
4. Shaker Epistemic: 1.96 * sigma_aleatoric * sqrt(4^MI - 1) via Huber closed-form entropy
5. Shaker Total: 1.96 * sqrt( 2^(2*H_total) / (2*pi*e) ) via Huber GMM entropy-power
6. RF-FIRE Lower: Standard Proximity lower quantile exploration term
7. Proximity A Lower: Pure unweighted topological path distance (fixed-k lower quantile)
8. Proximity B Lower: Pure unweighted topological continuous weighted lower quantile
9. Proximity AC Lower: Proximity A lower quantile combined with Density Scaling C
10. Proximity BC Lower: Proximity B lower quantile combined with Density Scaling C
11. Proximity LCB Lower: Pure Extracted Exploration Term with floor [Alias: u_plcb]
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from Epistemic_Quantifier import EpistemicQuantifier
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ


SURROGATE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "smac_default": {
        "estimator_class": ExtraTreesRegressor,
        "model_cls": ExtraTreesRegressor,
        "model_class": ExtraTreesRegressor,
        "n_estimators": 10,
        "max_depth": None,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "max_features": 5.0 / 6.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
    "mature": {
        "estimator_class": ExtraTreesRegressor,
        "model_cls": ExtraTreesRegressor,
        "model_class": ExtraTreesRegressor,
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "max_features": 5.0 / 6.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
    "shallow": {
        "estimator_class": ExtraTreesRegressor,
        "model_cls": ExtraTreesRegressor,
        "model_class": ExtraTreesRegressor,
        "n_estimators": 25,
        "max_depth": 4,
        "min_samples_split": 3,
        "min_samples_leaf": 5,
        "max_features": 5.0 / 6.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
    "coarse": {
        "estimator_class": ExtraTreesRegressor,
        "model_cls": ExtraTreesRegressor,
        "model_class": ExtraTreesRegressor,
        "n_estimators": 25,
        "max_depth": None,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": 5.0 / 6.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
    "breiman": {
        "estimator_class": RandomForestRegressor,
        "model_cls": RandomForestRegressor,
        "model_class": RandomForestRegressor,
        "n_estimators": 25,
        "max_depth": None,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "max_features": 5.0 / 6.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
}


def create_surrogate_rf(
    surrogate_type: str = "smac_default",
    seed: Optional[int] = None,
    model: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Create an ensemble regressor based on one of the 5 surrogate configurations or reuse existing.

    Supported surrogate types:
    - 'smac_default': ExtraTreesRegressor, 10 trees, SMAC3 defaults.
    - 'mature': ExtraTreesRegressor, 100 trees, SMAC3 defaults.
    - 'shallow': ExtraTreesRegressor, 25 trees, max_depth=4, min_samples_leaf=5.
    - 'coarse': ExtraTreesRegressor, 25 trees, min_samples_split=20, min_samples_leaf=10.
    - 'breiman': RandomForestRegressor, 25 trees, standard Breiman random forest.

    Parameters
    ----------
    surrogate_type : str, default='smac_default'
        Name of surrogate configuration.
    seed : Optional[int], default=None
        Random state seed.
    model : Optional[Any], default=None
        Existing fitted or unfitted surrogate model. If provided, copies
        hyperparameters or reuses the model.
    **kwargs : Any
        Additional hyperparameter overrides.

    Returns
    -------
    Ensemble regressor model instance.

    Raises
    ------
    ValueError
        If surrogate_type is not recognized in SURROGATE_CONFIGS.
    """
    if surrogate_type not in SURROGATE_CONFIGS:
        raise ValueError(
            f"Unknown surrogate_type '{surrogate_type}'. "
            f"Supported surrogates: {list(SURROGATE_CONFIGS.keys())}"
        )

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

    config = dict(SURROGATE_CONFIGS[surrogate_type])

    # Determine estimator class
    estimator_cls = config.get("estimator_class", ExtraTreesRegressor)
    if "estimator_class" in kwargs:
        estimator_cls = kwargs.pop("estimator_class")
    elif "model_class" in kwargs:
        estimator_cls = kwargs.pop("model_class")
    elif "model_cls" in kwargs:
        estimator_cls = kwargs.pop("model_cls")
    elif "use_extra_trees" in kwargs:
        use_extra = kwargs.pop("use_extra_trees")
        estimator_cls = ExtraTreesRegressor if use_extra else RandomForestRegressor

    # Clean out class metadata keys
    params = {
        k: v
        for k, v in config.items()
        if k not in ("estimator_class", "model_class", "model_cls")
    }
    params["random_state"] = seed
    params.update(kwargs)
    return estimator_cls(**params)


def create_smac_default_rf(
    seed: Optional[int] = None,
    n_trees: int = 10,
    model: Optional[Any] = None,
    use_extra_trees: bool = True,
    **kwargs: Any,
) -> Any:
    """Create an ensemble regressor with native SMAC3 defaults or reuse existing.

    Delegates to create_surrogate_rf for unified surrogate construction.

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
    overrides = dict(kwargs)
    overrides["use_extra_trees"] = use_extra_trees
    if n_trees != 10 or "n_estimators" not in overrides:
        overrides["n_estimators"] = n_trees
    return create_surrogate_rf(
        surrogate_type="smac_default",
        seed=seed,
        model=model,
        **overrides,
    )


@dataclass
class MultiUQResult:
    """Container for predictions and multi-method uncertainty quantifications.

    Supports both dataclass attribute access (e.g. `res.u_hutter_total`, `res.u_slcb`)
    and dictionary-style mapping access (e.g. `res['u_plcb_lower']`).

    Attributes
    ----------
    y_hat : np.ndarray
        Ensemble mean prediction of shape (M,).
    u_slcb : np.ndarray
        Backwards compatibility alias for u_hutter_total of shape (M,).
    u_plcb : np.ndarray
        Backwards compatibility alias for u_plcb_lower of shape (M,).
    var_between : np.ndarray
        Between-tree prediction variance of shape (M,).
    var_within : np.ndarray
        Mean within-tree leaf impurity variance of shape (M,).
    var_total : np.ndarray
        Total variance (var_between + var_within) of shape (M,).
    q_lower : np.ndarray
        Lower residual quantile from topological proximity of shape (M,).
    delta_floor : np.ndarray
        Adaptive exploration floor epsilon * kappa * local_mae of shape (M,).
    local_mae : np.ndarray
        Local Mean Absolute Error from proximity neighborhood of shape (M,).
    u_hutter_total : np.ndarray
        SMAC3 LCB total uncertainty: 1.96 * sqrt(var_total) of shape (M,).
    u_hutter_between : np.ndarray
        SMAC3 between-tree disagreement uncertainty: 1.96 * sqrt(var_between) of shape (M,).
    u_hutter_within : np.ndarray
        SMAC3 within-tree leaf impurity uncertainty: 1.96 * sqrt(var_within) of shape (M,).
    u_shaker_epistemic : np.ndarray
        Shaker epistemic uncertainty via Huber MI: 1.96 * sigma_aleatoric * sqrt(4^MI - 1) of shape (M,).
    u_shaker_total : np.ndarray
        Shaker total uncertainty via Huber GMM entropy power of shape (M,).
    shaker_mi : np.ndarray
        Mutual information in bits of shape (M,).
    shaker_total_entropy : np.ndarray
        GMM total differential entropy in bits of shape (M,).
    u_rf_fire_lower : np.ndarray
        RF-FIRE lower quantile exploration term of shape (M,).
    u_prox_a_lower : np.ndarray
        Proximity A (Topological Path Distance, unweighted tree walk) lower quantile exploration term of shape (M,).
    u_prox_b_lower : np.ndarray
        Proximity B (Topological Continuous Distance, unweighted tree walk) lower quantile exploration term of shape (M,).
    u_prox_ac_lower : np.ndarray
        Proximity AC (Proximity A combined with Density Scaling C) lower quantile exploration term of shape (M,).
    u_prox_bc_lower : np.ndarray
        Proximity BC (Proximity B combined with Density Scaling C) lower quantile exploration term of shape (M,).
    u_plcb_lower : np.ndarray
        Proximity LCB lower quantile exploration term with floor of shape (M,).
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
    u_hutter_total: np.ndarray
    u_hutter_between: np.ndarray
    u_hutter_within: np.ndarray
    u_shaker_epistemic: np.ndarray
    u_shaker_total: np.ndarray
    shaker_mi: np.ndarray
    shaker_total_entropy: np.ndarray
    u_rf_fire_lower: np.ndarray
    u_prox_a_lower: np.ndarray
    u_prox_b_lower: np.ndarray
    u_prox_ac_lower: np.ndarray
    u_prox_bc_lower: np.ndarray
    u_plcb_lower: np.ndarray

    def __getitem__(self, key: str) -> np.ndarray:
        if hasattr(self, key) and not key.startswith("_"):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}.")

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        if hasattr(self, key) and not key.startswith("_"):
            setattr(self, key, value)
        else:
            raise KeyError(f"Cannot set unknown field '{key}' in {self.__class__.__name__}.")

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key) and not key.startswith("_")

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def keys(self) -> Tuple[str, ...]:
        return (
            "y_hat",
            "u_hutter_total",
            "u_hutter_between",
            "u_hutter_within",
            "u_shaker_epistemic",
            "u_shaker_total",
            "shaker_mi",
            "shaker_total_entropy",
            "u_rf_fire_lower",
            "u_prox_a_lower",
            "u_prox_b_lower",
            "u_prox_ac_lower",
            "u_prox_bc_lower",
            "u_plcb_lower",
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

    def get_uncertainties_dict(self) -> Dict[str, np.ndarray]:
        """Returns dictionary mapping all estimator metric names and aliases to arrays."""
        return {
            "u_hutter_total": self.u_hutter_total,
            "u_hutter_between": self.u_hutter_between,
            "u_hutter_within": self.u_hutter_within,
            "u_shaker_epistemic": self.u_shaker_epistemic,
            "u_shaker_total": self.u_shaker_total,
            "u_rf_fire_lower": self.u_rf_fire_lower,
            "u_prox_a_lower": self.u_prox_a_lower,
            "u_prox_b_lower": self.u_prox_b_lower,
            "u_prox_ac_lower": self.u_prox_ac_lower,
            "u_prox_bc_lower": self.u_prox_bc_lower,
            "u_plcb_lower": self.u_plcb_lower,
            "u_slcb": self.u_slcb,
            "u_plcb": self.u_plcb,
            # Active estimator names for lower quantiles
            "hutter_total": self.u_hutter_total,
            "hutter_between": self.u_hutter_between,
            "hutter_within": self.u_hutter_within,
            "shaker_epistemic": self.u_shaker_epistemic,
            "shaker_total": self.u_shaker_total,
            "rf_fire": self.u_rf_fire_lower,
            "prox_a": self.u_prox_a_lower,
            "prox_b": self.u_prox_b_lower,
            "prox_ac": self.u_prox_ac_lower,
            "prox_bc": self.u_prox_bc_lower,
            "plcb": self.u_plcb_lower,
            "slcb": self.u_slcb,
        }


@dataclass
class DualUQResult(MultiUQResult):
    """Backwards-compatible subclass of MultiUQResult preserving legacy 9-key mapping interface."""

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


class MultiUQEvaluator:
    """Unified Multi-UQ Inference Engine.

    Evaluates 10 candidate uncertainty quantification methods across tree ensembles
    over the identical underlying forest and test query sample points.

    Parameters
    ----------
    model : Optional[Any], default=None
        Pre-configured or pre-fitted ensemble model. If None, instantiates
        ExtraTreesRegressor with native SMAC3 defaults.
    seed : Optional[int], default=None
        Random state seed.
    n_trees : int, default=10
        Number of trees in ensemble.
    k : int, default=28
        Number of nearest neighbors under proximity metric.
    epsilon : float, default=0.080791
        Exploration floor scaling coefficient.
    topological_decay_lambda : float, default=0.20486
        Exponential decay lambda for topological tree path distance.
    density_scaling_alpha : float, default=1.0
        Power exponent for topological density scaling in Proximity B+C.
    level : float, default=0.95
        Confidence level for quantile estimation (0.95 corresponds to kappa=1.96).
    kappa : float, default=1.96
        Standard deviation / error multiplier for LCB exploration bounds.
    device : str, default='cpu'
        Computation backend ('cpu', 'gpu', or 'auto').
    use_extra_trees : Optional[bool], default=None
        Whether to use ExtraTreesRegressor for random splitting. If None,
        respects surrogate_type default.
    batch_size : Union[int, str], default=512
        Chunk size for query batching.
    surrogate_type : str, default='smac_default'
        Surrogate configuration identifier ('smac_default', 'mature', 'shallow',
        'coarse', 'breiman').
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
        density_scaling_alpha: float = 1.0,
        level: float = 0.95,
        kappa: float = 1.96,
        device: str = "cpu",
        use_extra_trees: Optional[bool] = None,
        batch_size: Union[int, str] = 512,
        surrogate_type: str = "smac_default",
        eval_mode: str = "all",
        weighted_proximity: bool = False,
        **kwargs: Any,
    ) -> None:
        if eval_mode not in ("all", "unweighted_proximity_only"):
            raise ValueError(
                f"Unknown eval_mode '{eval_mode}'. Must be 'all' or 'unweighted_proximity_only'."
            )
        self.eval_mode = eval_mode
        self.surrogate_type = surrogate_type
        self.seed = seed
        self.weighted_proximity = weighted_proximity

        # Resolve n_trees: if explicitly overridden (kwargs or n_trees != 10), use override.
        # Otherwise respect surrogate defaults.
        if "n_estimators" in kwargs:
            resolved_n_trees = kwargs["n_estimators"]
        elif n_trees != 10:
            resolved_n_trees = n_trees
        else:
            resolved_n_trees = SURROGATE_CONFIGS.get(surrogate_type, {}).get(
                "n_estimators", n_trees
            )

        self.n_trees = resolved_n_trees
        self.n_estimators = resolved_n_trees
        self.k = k
        self.epsilon = epsilon
        self.topological_decay_lambda = topological_decay_lambda
        self.density_scaling_alpha = density_scaling_alpha
        self.level = level
        self.kappa = kappa
        self.device = device
        self.batch_size = batch_size

        rf_kwargs = dict(kwargs)
        if "n_estimators" not in rf_kwargs:
            rf_kwargs["n_estimators"] = resolved_n_trees
        if use_extra_trees is not None:
            rf_kwargs["use_extra_trees"] = use_extra_trees

        if model is not None:
            self.model = create_surrogate_rf(
                surrogate_type=surrogate_type,
                seed=seed,
                model=model,
                **rf_kwargs,
            )
            if hasattr(self.model, "n_estimators"):
                self.n_trees = self.model.n_estimators
                self.n_estimators = self.model.n_estimators
        else:
            self.model = create_surrogate_rf(
                surrogate_type=surrogate_type,
                seed=seed,
                **rf_kwargs,
            )

        self.use_extra_trees = isinstance(self.model, ExtraTreesRegressor)

        self.uq_model: Optional[GPUProximityRegressionUQ] = None
        self.shaker: Optional[EpistemicQuantifier] = None
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        refit: bool = False,
    ) -> "MultiUQEvaluator":
        """Fit the underlying forest and initialize proximity and Shaker engines.

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

        # 1. Proximity UQ engine wrapping identical forest with precomputed distances and baseline density
        self.uq_model = GPUProximityRegressionUQ(
            model=self.model,
            X_train=X_arr,
            y_train=y_arr,
            device=self.device,
            topological_decay_lambda=self.topological_decay_lambda,
            density_scaling_alpha=self.density_scaling_alpha,
            use_density_scaling=True,
            weighting="leaf_normalized" if self.weighted_proximity else "unweighted_all",
            weighted=self.weighted_proximity,
        )
        self.uq_model.fit()

        # 2. EpistemicQuantifier for Shaker GMM entropy family
        if self.eval_mode == "unweighted_proximity_only":
            self.shaker = None
        else:
            self.shaker = EpistemicQuantifier(
                model=self.model,
                X_train=X_arr,
                y_train=y_arr,
            )

        self.X_train_ = X_arr
        self.y_train_ = y_arr
        self._is_fitted = True
        return self

    def evaluate(self, X_test: np.ndarray) -> MultiUQResult:
        """Evaluate all 10 uncertainty quantification methods.

        Parameters
        ----------
        X_test : np.ndarray
            Query points of shape (M, D) or (D,).

        Returns
        -------
        MultiUQResult
            Container with predictions, variances, and 15 uncertainty estimators.
        """
        if not self._is_fitted or self.X_train_ is None:
            raise RuntimeError("MultiUQEvaluator must be fitted before calling evaluate().")

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

        M = X_test_2d.shape[0]
        N_train = len(self.X_train_)
        estimators = self.model.estimators_
        B = len(estimators)

        # =========================================================================
        # Group 1: Hutter Law of Total Variance Family
        # =========================================================================
        tree_preds = np.column_stack([t.predict(X_test_2d) for t in estimators])
        y_hat = np.mean(tree_preds, axis=1).astype(np.float64)

        var_between = (
            np.var(tree_preds, axis=1, ddof=1).astype(np.float64)
            if B > 1
            else np.zeros_like(y_hat)
        )

        tree_impurities = np.column_stack([
            np.maximum(0.0, t.tree_.impurity[t.apply(X_test_2d)]) for t in estimators
        ])
        var_within = np.mean(tree_impurities, axis=1).astype(np.float64)

        var_total = np.maximum(0.0, var_between + var_within)

        u_hutter_between = (self.kappa * np.sqrt(np.maximum(0.0, var_between))).astype(np.float64)
        u_hutter_within = (self.kappa * np.sqrt(np.maximum(0.0, var_within))).astype(np.float64)
        u_hutter_total = (self.kappa * np.sqrt(var_total)).astype(np.float64)
        u_slcb = u_hutter_total

        # =========================================================================
        # Group 2: Shaker Information-Theoretic Entropy Family
        # =========================================================================
        if self.eval_mode == "unweighted_proximity_only":
            u_shaker_epistemic = np.zeros(M, dtype=np.float64)
            u_shaker_total = np.zeros(M, dtype=np.float64)
            shaker_mi = np.zeros(M, dtype=np.float64)
            shaker_total_entropy = np.zeros(M, dtype=np.float64)
        else:
            if self.shaker is None:
                self.shaker = EpistemicQuantifier(
                    model=self.model,
                    X_train=self.X_train_,
                    y_train=self.y_train_,
                )

            shaker_mi = np.asarray(
                self.shaker.shaker_get_epistemic_entropy(
                    X_test_2d,
                    method="huber",
                    backend=self.device,
                ),
                dtype=np.float64,
            )

            shaker_total_entropy = np.asarray(
                self.shaker._shaker_calc_total_entropy(
                    X_test_2d,
                    method="huber",
                    backend=self.device,
                ),
                dtype=np.float64,
            )

            aleatoric_var = np.asarray(
                self.shaker.base_get_aleatoric_variance(X_test_2d),
                dtype=np.float64,
            )
            sigma_aleatoric = np.sqrt(np.maximum(aleatoric_var, 1e-6))

            safe_mi_exp = np.clip(2.0 * shaker_mi, 0.0, 50.0)
            u_shaker_epistemic = (
                self.kappa * sigma_aleatoric * np.sqrt(np.maximum(2.0 ** safe_mi_exp - 1.0, 0.0))
            ).astype(np.float64)

            safe_tot_exp = np.clip(2.0 * shaker_total_entropy, -50.0, 50.0)
            entropy_power_var = (2.0 ** safe_tot_exp) / (2.0 * np.pi * np.e)
            u_shaker_total = (
                self.kappa * np.sqrt(np.maximum(entropy_power_var, 0.0))
            ).astype(np.float64)

        # =========================================================================
        # Group 3: Proximity / RF-FIRE / RF-GAP Family
        # =========================================================================
        k_eff = min(self.k, N_train)
        alpha_lwr = (1.0 - self.level) / 2.0
        alpha_upr = 1.0 - alpha_lwr

        has_precomputed_topo = (
            self.topological_decay_lambda is not None
            and self.uq_model is not None
            and hasattr(self.uq_model, "tree_leaf_distances")
            and self.uq_model.tree_leaf_distances is not None
            and len(self.uq_model.tree_leaf_distances) > 0
        )

        if has_precomputed_topo:
            u_rf_fire_lower = np.zeros(M, dtype=np.float64)
            u_prox_a_lower = np.zeros(M, dtype=np.float64)
            u_prox_b_lower = np.zeros(M, dtype=np.float64)
            u_prox_ac_lower = np.zeros(M, dtype=np.float64)
            u_prox_bc_lower = np.zeros(M, dtype=np.float64)
            u_plcb_lower = np.zeros(M, dtype=np.float64)
            q_lower = np.zeros(M, dtype=np.float64)
            delta_floor = np.zeros(M, dtype=np.float64)
            local_mae = np.zeros(M, dtype=np.float64)

            oob_res = np.asarray(self.uq_model.oob_residuals, dtype=np.float64)
            valid_oob_mask = (
                np.asarray(self.uq_model.valid_oob_mask)
                if hasattr(self.uq_model, "valid_oob_mask") and self.uq_model.valid_oob_mask is not None
                else None
            )
            n_baseline = float(getattr(self.uq_model, "N_baseline", 1.0))

            chunk_size = 512 if isinstance(self.batch_size, str) else int(self.batch_size)

            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)
                b_len = end - start
                X_chunk = X_test_2d[start:end, :]
                leaf_batch = self.model.apply(X_chunk)

                prox_topo_unweighted = np.zeros((b_len, N_train), dtype=np.float32)
                density_chunk = np.zeros(b_len, dtype=np.float32)
                prox_fire = np.zeros((b_len, N_train), dtype=np.float32)

                for t in range(B):
                    id_to_dense = self.uq_model.tree_leaf_id_to_dense[t]
                    if hasattr(id_to_dense, "get"):
                        id_to_dense = id_to_dense.get()
                    dense_test = id_to_dense[leaf_batch[:, t]]
                    dense_train_inbag = id_to_dense[self.uq_model.in_bag_leaves[:, t]]
                    dense_train_all = id_to_dense[self.uq_model.leaf_matrix_train[:, t]]

                    tree_dists = self.uq_model.tree_leaf_distances[t]
                    if hasattr(tree_dists, "get"):
                        tree_dists = tree_dists.get()

                    # Unweighted (Option A: pure Breiman topological across all trees)
                    d_t_unweighted = tree_dists[dense_test[:, None], dense_train_all[None, :]]
                    decay_t_unweighted = np.exp(-self.topological_decay_lambda * d_t_unweighted)
                    prox_topo_unweighted += decay_t_unweighted

                    # Walked density sum for density scaling gamma
                    d_t_inbag = tree_dists[dense_test[:, None], dense_train_inbag[None, :]]
                    decay_t_inbag = np.exp(-self.topological_decay_lambda * d_t_inbag)
                    in_bag_c = self.uq_model.in_bag_counts[:, t]
                    density_chunk += np.sum(decay_t_inbag * in_bag_c[None, :], axis=1)

                    # RF-FIRE co-occurrence
                    train_w = self.uq_model.train_weights[:, t]
                    matches_t = (leaf_batch[:, t, None] == self.uq_model.in_bag_leaves[None, :, t])
                    prox_fire += matches_t * train_w[None, :]

                prox_topo_unweighted /= B
                prox_fire /= B

                if valid_oob_mask is not None:
                    prox_topo_unweighted[:, ~valid_oob_mask] = 0.0
                    prox_fire[:, ~valid_oob_mask] = 0.0

                # 1. Topological fixed-k unweighted (Proximity A & PLCB)
                if k_eff < N_train:
                    partition_idx_topo_u = np.flip(np.argsort(prox_topo_unweighted, axis=1), axis=1)[:, :k_eff]
                    k_residuals_topo_u = oob_res[partition_idx_topo_u]
                else:
                    k_residuals_topo_u = np.broadcast_to(oob_res[None, :], (b_len, N_train))

                q_lwr_topo_u = np.quantile(k_residuals_topo_u, alpha_lwr, axis=1)
                q_upr_topo_u = np.quantile(k_residuals_topo_u, alpha_upr, axis=1)

                in_int_topo_u = (k_residuals_topo_u >= q_lwr_topo_u[:, None]) & (k_residuals_topo_u <= q_upr_topo_u[:, None])
                cnt_topo_u = np.sum(in_int_topo_u, axis=1)
                sum_abs_topo_u = np.sum(np.abs(k_residuals_topo_u) * in_int_topo_u, axis=1)
                b_local_mae_u = np.where(
                    cnt_topo_u > 0,
                    sum_abs_topo_u / np.maximum(cnt_topo_u, 1),
                    float(self.uq_model.oob_mae),
                )
                b_delta_floor_u = self.epsilon * self.kappa * b_local_mae_u

                # Proximity A
                b_u_prox_a_lower = np.maximum(0.0, -q_lwr_topo_u)

                # Continuous weighted quantile for Proximity B using unweighted tree-walk proximities
                q_lwr_b_unw = self.uq_model._compute_weighted_quantile(oob_res, prox_topo_unweighted, alpha_lwr)
                if hasattr(q_lwr_b_unw, "get"):
                    q_lwr_b_unw = q_lwr_b_unw.get()
                b_u_prox_b_lower = np.maximum(0.0, -q_lwr_b_unw)

                # Density scaling gamma
                avg_test_leaf_sizes = density_chunk / B
                avg_test_leaf_sizes = np.maximum(avg_test_leaf_sizes, 1e-5)
                gamma = np.maximum(1.0, (n_baseline / avg_test_leaf_sizes) ** self.density_scaling_alpha)

                # Proximity AC (A combined with C)
                b_u_prox_ac_lower = gamma * b_u_prox_a_lower

                # Proximity BC (B combined with C)
                b_u_prox_bc_lower = gamma * b_u_prox_b_lower

                # PLCB
                b_u_plcb_lower = np.maximum(b_delta_floor_u, -q_lwr_topo_u)

                # RF-FIRE lower quantile
                if k_eff < N_train:
                    partition_idx_fire = np.flip(np.argsort(prox_fire, axis=1), axis=1)[:, :k_eff]
                    k_residuals_fire = oob_res[partition_idx_fire]
                else:
                    k_residuals_fire = np.broadcast_to(oob_res[None, :], (b_len, N_train))
                q_lwr_fire = np.quantile(k_residuals_fire, alpha_lwr, axis=1)
                b_u_rf_fire_lower = np.maximum(0.0, -q_lwr_fire)

                # Store into output arrays
                u_prox_a_lower[start:end] = b_u_prox_a_lower
                u_prox_b_lower[start:end] = b_u_prox_b_lower
                u_prox_ac_lower[start:end] = b_u_prox_ac_lower
                u_prox_bc_lower[start:end] = b_u_prox_bc_lower
                u_plcb_lower[start:end] = b_u_plcb_lower
                u_rf_fire_lower[start:end] = b_u_rf_fire_lower

                q_lower[start:end] = q_lwr_topo_u
                delta_floor[start:end] = b_delta_floor_u
                local_mae[start:end] = b_local_mae_u
        else:
            # Fallback for mocked or non-topological UQ model
            y_pred_lwr, _, _, b_local_mae = self.uq_model.predict_with_intervals(
                X_test_2d,
                n_neighbors=k_eff,
                level=self.level,
                return_mae=True,
            )
            q_lower = (y_pred_lwr - y_hat).astype(np.float64)
            local_mae = np.asarray(b_local_mae, dtype=np.float64)
            delta_floor = self.epsilon * self.kappa * local_mae

            u_plcb_lower = np.maximum(delta_floor, -q_lower)
            u_prox_a_lower = np.maximum(0.0, -q_lower)
            u_prox_b_lower = u_prox_a_lower
            u_prox_ac_lower = u_prox_a_lower
            u_prox_bc_lower = u_prox_a_lower
            u_rf_fire_lower = u_prox_a_lower

        u_slcb = u_hutter_total
        u_plcb = u_plcb_lower

        return MultiUQResult(
            y_hat=y_hat,
            u_slcb=u_slcb,
            u_plcb=u_plcb,
            var_between=var_between,
            var_within=var_within,
            var_total=var_total,
            q_lower=q_lower,
            delta_floor=delta_floor,
            local_mae=local_mae,
            u_hutter_total=u_hutter_total,
            u_hutter_between=u_hutter_between,
            u_hutter_within=u_hutter_within,
            u_shaker_epistemic=u_shaker_epistemic,
            u_shaker_total=u_shaker_total,
            shaker_mi=shaker_mi,
            shaker_total_entropy=shaker_total_entropy,
            u_rf_fire_lower=u_rf_fire_lower,
            u_prox_a_lower=u_prox_a_lower,
            u_prox_b_lower=u_prox_b_lower,
            u_prox_ac_lower=u_prox_ac_lower,
            u_prox_bc_lower=u_prox_bc_lower,
            u_plcb_lower=u_plcb_lower,
        )


class DualUQEvaluator(MultiUQEvaluator):
    """Dual Uncertainty Quantification Inference Engine.

    Unifies Standard SMAC3 LCB (Hutter Law of Total Variance) and Proximity LCB
    (Pure Extracted Exploration Term) on the identical underlying random forest.
    Maintains 100% backwards compatibility with legacy callers and test assertions.
    """

    def __init__(
        self,
        *args: Any,
        eval_mode: str = "all",
        weighted_proximity: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, eval_mode=eval_mode, weighted_proximity=weighted_proximity, **kwargs)

    def evaluate(self, X_test: np.ndarray) -> DualUQResult:
        """Evaluate both Standard SMAC3 LCB and Proximity LCB.

        Returns
        -------
        DualUQResult
            Container with predictions, variances, quantiles, and uncertainties.
        """
        multi_res = super().evaluate(X_test)
        return DualUQResult(
            y_hat=multi_res.y_hat,
            u_slcb=multi_res.u_slcb,
            u_plcb=multi_res.u_plcb,
            var_between=multi_res.var_between,
            var_within=multi_res.var_within,
            var_total=multi_res.var_total,
            q_lower=multi_res.q_lower,
            delta_floor=multi_res.delta_floor,
            local_mae=multi_res.local_mae,
            u_hutter_total=multi_res.u_hutter_total,
            u_hutter_between=multi_res.u_hutter_between,
            u_hutter_within=multi_res.u_hutter_within,
            u_shaker_epistemic=multi_res.u_shaker_epistemic,
            u_shaker_total=multi_res.u_shaker_total,
            shaker_mi=multi_res.shaker_mi,
            shaker_total_entropy=multi_res.shaker_total_entropy,
            u_rf_fire_lower=multi_res.u_rf_fire_lower,
            u_prox_a_lower=multi_res.u_prox_a_lower,
            u_prox_b_lower=multi_res.u_prox_b_lower,
            u_prox_ac_lower=multi_res.u_prox_ac_lower,
            u_prox_bc_lower=multi_res.u_prox_bc_lower,
            u_plcb_lower=multi_res.u_plcb_lower,
        )
