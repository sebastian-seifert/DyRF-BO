from __future__ import annotations
from typing import Callable, Union, Any
import numpy as np

from smac.model.random_forest import RandomForest
from ep_extractors import UQExtractorRegistry

class CustomUncertaintyRandomForest(RandomForest):
    """
    SMAC3-native Random Forest surrogate that replaces empirical tree variance 
    with a pluggable custom uncertainty function (either a registry key or a custom Python callable),
    extracting tree statistics directly from SMAC3's native EPMRandomForest (self._rf).
    """
    def __init__(
        self,
        uncertainty_func: Union[str, Callable[[Any, np.ndarray, np.ndarray], np.ndarray]] = "standard_disagreement",
        oob_score: bool = True,
        extractor_kwargs: dict | None = None,
        **kwargs
    ):
        super().__init__(oob_score=oob_score, **kwargs)
        self.uncertainty_func = uncertainty_func
        self.extractor_kwargs = extractor_kwargs
        self.uq_extractor = None
        self.last_X = None
        self.last_y = None

    @property
    def extractor(self):
        """Alias property for uq_extractor providing unified surrogate interface."""
        return self.uq_extractor

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        Y: np.ndarray | None = None,
        **kwargs
    ) -> "CustomUncertaintyRandomForest":
        """Fits standard SMAC3 Random Forest and the custom uncertainty extractor/function."""
        y_target = Y if y is None else y
        if y_target is None:
            raise ValueError("Target values must be provided via 'y' or 'Y'.")
        super().train(X, y_target)
        X_clean = self._impute_inactive(X)
        self.last_X = X_clean
        self.last_y = y_target.flatten()
        
        # SMAC3's self._rf is an EPMRandomForest (subclass of sklearn RandomForestRegressor).
        # We pass self._rf directly into UQExtractorRegistry with zero secondary model retraining overhead!
        if isinstance(self.uncertainty_func, str) and self._rf is not None:
            self.uq_extractor = UQExtractorRegistry.get(
                self.uncertainty_func, self._rf, **(self.extractor_kwargs or {})
            )
            self.uq_extractor.fit(X_clean, self.last_y)
        return self

    def _predict(self, X: np.ndarray, covariance_type: str | None = "diagonal") -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean using SMAC3 Random Forest and computes custom uncertainty signal U(X).
        Returns (mean, U(X)^2) so SMAC3's native acquisition functions (EI, PI, LCB) use U(X)!
        """
        mean, _ = super()._predict(X, covariance_type=covariance_type)
        X_clean = self._impute_inactive(X)
        
        if isinstance(self.uncertainty_func, str):
            if self.uq_extractor is None and self._rf is not None:
                self.uq_extractor = UQExtractorRegistry.get(
                    self.uncertainty_func, self._rf, **(self.extractor_kwargs or {})
                )
                self.uq_extractor.fit(self.last_X if self.last_X is not None else X_clean, self.last_y)
            unc_signal = self.uq_extractor.extract_epistemic_signal(X_clean)
        elif callable(self.uncertainty_func):
            unc_signal = self.uncertainty_func(self._rf or self, X_clean, self.last_y)
        else:
            raise ValueError(f"Unsupported uncertainty_func: {self.uncertainty_func}")
            
        var = (unc_signal ** 2).reshape(-1, 1)
        var = np.maximum(var, 1e-10)
        return mean, var

    def predict_standard_rf(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the pure, authentic SMAC3 Random Forest (mean, tree_variance)
        directly from underlying EPMRandomForest (super()._predict),
        strictly bypassing any custom uncertainty extractors.
        """
        X_clean = self._impute_inactive(X)
        return super()._predict(X_clean)

    @property
    def oob_mae(self) -> float:
        """Exposes global OOB MAE from active extractor or underlying EPMRandomForest."""
        if self.uq_extractor is not None and hasattr(self.uq_extractor, "oob_mae"):
            return self.uq_extractor.oob_mae
        if hasattr(self._rf, "oob_prediction_") and self.last_y is not None:
            return float(np.mean(np.abs(self.last_y - self._rf.oob_prediction_)))
        return 1.0

    def predict_with_intervals(
        self,
        X: np.ndarray,
        n_neighbors: int | str = "auto",
        level: float = 0.95,
        return_mae: bool = False
    ):
        """
        Generates point predictions and empirical prediction intervals.
        Delegates directly to uq_extractor if supported.
        """
        X_clean = self._impute_inactive(X)
        if self.uq_extractor is None and self._rf is not None and isinstance(self.uncertainty_func, str):
            self.uq_extractor = UQExtractorRegistry.get(
                self.uncertainty_func, self._rf, **(self.extractor_kwargs or {})
            )
            self.uq_extractor.fit(self.last_X if self.last_X is not None else X_clean, self.last_y)

        if self.uq_extractor is not None and hasattr(self.uq_extractor, "predict_with_intervals"):
            return self.uq_extractor.predict_with_intervals(
                X_clean, n_neighbors=n_neighbors, level=level, return_mae=return_mae
            )

        # Fallback to Gaussian interval via _predict
        mean, var = self._predict(X_clean)
        mean = mean.flatten()
        std = np.sqrt(var.flatten())
        from scipy.stats import norm
        alpha = 1.0 - level
        kappa = float(norm.ppf(1.0 - alpha / 2.0)) if level > 0.0 else 0.0
        y_lwr = mean - kappa * std
        y_upr = mean + kappa * std
        if return_mae:
            mae = np.full_like(mean, self.oob_mae)
            return y_lwr, mean, y_upr, mae
        return y_lwr, mean, y_upr
