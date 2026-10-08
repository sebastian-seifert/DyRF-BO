from __future__ import annotations
import warnings
from typing import Any
import numpy as np
from scipy.stats import norm
from smac.acquisition.function import AbstractAcquisitionFunction

class ProximityLowerBoundAcquisition(AbstractAcquisitionFunction):
    """
    Direct Empirical Lower Bound Acquisition Function (Proximity LCB) for SMAC3.
    
    Evaluates:
        Acq(x) = -y_pred_lwr_floored(x)
        
    where:
        delta_floor(x) = eps * kappa(alpha) * local_mae(x)
        y_pred_lwr_floored(x) = min(y_pred_lwr(x), y_pred(x) - delta_floor(x))
        
    SMAC maximizes acquisition functions, so argmax -y_pred_lwr_floored(x)
    selects candidates with minimal lower bound (exploitation + exploration),
    with guaranteed minimum exploration delta_floor(x).
    """
    def __init__(
        self,
        eps: float = 0.16,
        level: float = 0.95,
        k: int | str = 25,
        k_warmup: int = 25,
        weighting: str = "unweighted",
        use_leaf_weights: bool | None = None,
        weighted: bool = False,
        method: str | None = None,
        fallback_on_low_support: bool = False,
    ) -> None:
        super().__init__()
        self._eps = float(eps)
        self._level = float(level)
        self._k = k
        self._k_warmup = int(k_warmup)
        self._fallback_on_low_support = bool(fallback_on_low_support)
        if weighted is True or use_leaf_weights is True or weighting == "weighted":
            self._weighting = "weighted"
            self._weighted = True
        else:
            self._weighting = "unweighted"
            self._weighted = False
        self._use_leaf_weights = use_leaf_weights
        self._method = method.lower() if isinstance(method, str) else None
        alpha = 1.0 - self._level
        self._kappa = float(norm.ppf(1.0 - alpha / 2.0)) if self._level > 0.0 else 0.0
        self._num_data: int | None = None

    @property
    def name(self) -> str:
        return f"Proximity Lower Bound (level={self._level}, eps={self._eps}, k={self._k})"

    @property
    def meta(self) -> dict[str, Any]:
        meta = super().meta
        meta.update({
            "eps": self._eps,
            "level": self._level,
            "k": self._k,
            "k_warmup": self._k_warmup,
            "kappa": self._kappa,
            "weighting": self._weighting,
            "weighted": self._weighted,
            "method": self._method,
            "fallback_on_low_support": self._fallback_on_low_support,
        })
        return meta

    def _is_continuous_method(self) -> bool:
        """
        Determines whether the acquisition operates in continuous mode (methods B or BC)
        versus discrete neighbor mode (methods A or AC).
        Continuous mode bypasses k-warmup and computes intervals using continuous tree-walk proximities.
        """
        if self._method is not None:
            m = self._method.lower()
            return m.startswith("proximity_b") or m in (
                "b", "bc", "b_cv", "bc_cv", "proximity_b", "proximity_bc", "proximity_b_cv", "proximity_bc_cv"
            )

        if self._model is not None:
            # 1. Check model.uncertainty_func
            u_func = getattr(self._model, "uncertainty_func", None)
            if isinstance(u_func, str):
                u_str = u_func.lower()
                if u_str.startswith("proximity_b") or u_str in (
                    "b", "bc", "b_cv", "bc_cv", "proximity_b", "proximity_bc", "proximity_b_cv", "proximity_bc_cv"
                ):
                    return True

            # 2. Check attached extractor or the model object itself
            extractor = getattr(self._model, "uq_extractor", None) or getattr(self._model, "extractor", None)
            candidates = [extractor, self._model]
            for obj in candidates:
                if obj is not None:
                    cls_name = type(obj).__name__
                    if (
                        cls_name.startswith("ProximityB")
                        or cls_name.startswith("ProximityBC")
                        or "proximityb" in cls_name.lower()
                        or "proximity_b" in cls_name.lower()
                    ):
                        return True
                    if hasattr(obj, "method") and isinstance(obj.method, str):
                        m_obj = obj.method.lower()
                        if m_obj.startswith("proximity_b") or m_obj in (
                            "b", "bc", "b_cv", "bc_cv", "proximity_b", "proximity_bc", "proximity_b_cv", "proximity_bc_cv"
                        ):
                            return True

        return False

    def _update(self, **kwargs: Any) -> None:
        self._num_data = int(kwargs["num_data"]) if "num_data" in kwargs else None

    def _compute(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Surrogate model must be set before computing acquisition values.")

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        # 1. Determine sample size N
        N = self._num_data
        if N is None:
            if hasattr(self._model, "last_X") and self._model.last_X is not None:
                N = len(self._model.last_X)
            elif hasattr(self._model, "n_train"):
                N = int(self._model.n_train)
            else:
                N = 0

        # 2. Determine mode and effective neighbor parameter
        is_continuous = self._is_continuous_method()
        if is_continuous:
            # Continuous mode (B, BC): bypass k-warmup when N > 0, using continuous tree-walk proximities
            should_warmup = (N is None or N <= 0)
            n_neighbors = "auto"
        else:
            # Discrete mode (A, AC): retain discrete k-warmup
            if isinstance(self._k, (int, np.integer)):
                k_eff = int(self._k)
            else:
                k_eff = self._k_warmup
            should_warmup = (N is None or N <= k_eff)
            n_neighbors = self._k

        # 3. Phase 1: Pure Native SMAC3 Random Forest LCB during Warm Start
        if should_warmup:
            if hasattr(self._model, "predict_standard_rf"):
                mean_rf, var_rf = self._model.predict_standard_rf(X)
            else:
                mean_rf, var_rf = self._model.predict_marginalized(X)
            mean_rf = np.asarray(mean_rf, dtype=np.float64).flatten()
            std_rf = np.sqrt(np.maximum(np.asarray(var_rf, dtype=np.float64).flatten(), 1e-10))
            # Standard RF LCB for minimization: -(mean - kappa * std) = -mean + kappa * std
            return (-mean_rf + self._kappa * std_rf).reshape(-1, 1)

        # 4. Phase 2: Floored Proximity Lower Bound
        # Obtain intervals and local in-interval MAE
        if hasattr(self._model, "predict_with_intervals"):
            try:
                res = self._model.predict_with_intervals(
                    X,
                    n_neighbors=n_neighbors,
                    level=self._level,
                    return_mae=True,
                    weighting=self._weighting,
                    weighted=self._weighted,
                    fallback_on_low_support=self._fallback_on_low_support,
                )
            except TypeError:
                try:
                    res = self._model.predict_with_intervals(
                        X,
                        n_neighbors=n_neighbors,
                        level=self._level,
                        return_mae=True,
                        weighting=self._weighting,
                        fallback_on_low_support=self._fallback_on_low_support,
                    )
                except TypeError:
                    try:
                        res = self._model.predict_with_intervals(
                            X,
                            n_neighbors=n_neighbors,
                            level=self._level,
                            return_mae=True,
                            fallback_on_low_support=self._fallback_on_low_support,
                        )
                    except TypeError:
                        res = self._model.predict_with_intervals(
                            X, n_neighbors=n_neighbors, level=self._level, return_mae=True
                        )
            if len(res) == 4:
                y_pred_lwr, y_pred, _, local_mae = res
            else:
                y_pred_lwr, y_pred, _ = res
                local_mae = getattr(self._model, "oob_mae", 1.0)
        else:
            # Fallback to standard Gaussian prediction
            warnings.warn(
                f"Surrogate model '{type(self._model).__name__}' does not have 'predict_with_intervals'; "
                "falling back to standard Gaussian prediction for Proximity LCB.",
                UserWarning,
                stacklevel=2,
            )
            mean, var = self._model.predict_marginalized(X)
            y_pred = mean.flatten()
            std = np.sqrt(np.maximum(var.flatten(), 1e-10))
            y_pred_lwr = y_pred - self._kappa * std
            local_mae = np.full_like(y_pred, getattr(self._model, "oob_mae", 1.0))

        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
        y_pred_lwr = np.asarray(y_pred_lwr, dtype=np.float64).flatten()
        local_mae = np.asarray(local_mae, dtype=np.float64).flatten()
        if local_mae.size == 1 and y_pred.size > 1:
            local_mae = np.full_like(y_pred, local_mae.item())

        # Compute point-adaptive exploration floor if eps > 0 and kappa > 0
        if self._eps > 0.0 and self._kappa > 0.0:
            delta_floor = self._eps * self._kappa * local_mae
            y_pred_lwr_floored = np.minimum(y_pred_lwr, y_pred - delta_floor)
        else:
            y_pred_lwr_floored = y_pred_lwr

        # Negate for SMAC maximizer (minimizing lower bound <=> maximizing -lower bound)
        return (-y_pred_lwr_floored).reshape(-1, 1)
