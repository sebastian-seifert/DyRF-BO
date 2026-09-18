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
        k_warmup: int = 25
    ) -> None:
        super().__init__()
        self._eps = float(eps)
        self._level = float(level)
        self._k = k
        self._k_warmup = int(k_warmup)
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
        })
        return meta

    def _update(self, **kwargs: Any) -> None:
        if "num_data" in kwargs:
            self._num_data = int(kwargs["num_data"])

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

        # 2. Determine effective k
        if isinstance(self._k, (int, np.integer)):
            k_eff = int(self._k)
        else:
            k_eff = self._k_warmup

        # 3. Phase 1: Pure Native SMAC3 Random Forest LCB during Warm Start (N <= k_eff)
        if N <= k_eff:
            if hasattr(self._model, "predict_standard_rf"):
                mean_rf, var_rf = self._model.predict_standard_rf(X)
            else:
                mean_rf, var_rf = self._model.predict_marginalized(X)
            mean_rf = np.asarray(mean_rf, dtype=np.float64).flatten()
            std_rf = np.sqrt(np.maximum(np.asarray(var_rf, dtype=np.float64).flatten(), 1e-10))
            # Standard RF LCB for minimization: -(mean - kappa * std) = -mean + kappa * std
            return (-mean_rf + self._kappa * std_rf).reshape(-1, 1)

        # 4. Phase 2: Floored Proximity Lower Bound when N > k_eff
        # Obtain intervals and local in-interval MAE
        if hasattr(self._model, "predict_with_intervals"):
            res = self._model.predict_with_intervals(
                X, n_neighbors=self._k, level=self._level, return_mae=True
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
            std = np.sqrt(var.flatten())
            y_pred_lwr = y_pred - self._kappa * std
            local_mae = np.full_like(y_pred, getattr(self._model, "oob_mae", 1.0))

        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
        y_pred_lwr = np.asarray(y_pred_lwr, dtype=np.float64).flatten()
        local_mae = np.asarray(local_mae, dtype=np.float64).flatten()

        # Compute point-adaptive exploration floor if eps > 0 and kappa > 0
        if self._eps > 0.0 and self._kappa > 0.0:
            delta_floor = self._eps * self._kappa * local_mae
            y_pred_lwr_floored = np.minimum(y_pred_lwr, y_pred - delta_floor)
        else:
            y_pred_lwr_floored = y_pred_lwr

        # Negate for SMAC maximizer (minimizing lower bound <=> maximizing -lower bound)
        return (-y_pred_lwr_floored).reshape(-1, 1)
