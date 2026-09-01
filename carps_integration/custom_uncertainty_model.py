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
        **kwargs
    ):
        super().__init__(oob_score=oob_score, **kwargs)
        self.uncertainty_func = uncertainty_func
        self.uq_extractor = None
        self.last_X = None
        self.last_y = None

    def train(self, X: np.ndarray, y: np.ndarray) -> "CustomUncertaintyRandomForest":
        """Fits standard SMAC3 Random Forest and the custom uncertainty extractor/function."""
        super().train(X, y)
        X_clean = self._impute_inactive(X)
        self.last_X = X_clean
        self.last_y = y.flatten() if y is not None else None
        
        # SMAC3's self._rf is an EPMRandomForest (subclass of sklearn RandomForestRegressor).
        # We pass self._rf directly into UQExtractorRegistry with zero secondary model retraining overhead!
        if isinstance(self.uncertainty_func, str) and self._rf is not None:
            self.uq_extractor = UQExtractorRegistry.get(self.uncertainty_func, self._rf)
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
                self.uq_extractor = UQExtractorRegistry.get(self.uncertainty_func, self._rf)
                self.uq_extractor.fit(self.last_X if self.last_X is not None else X_clean, self.last_y)
            unc_signal = self.uq_extractor.extract_epistemic_signal(X_clean)
        elif callable(self.uncertainty_func):
            unc_signal = self.uncertainty_func(self._rf or self, X_clean, self.last_y)
        else:
            raise ValueError(f"Unsupported uncertainty_func: {self.uncertainty_func}")
            
        var = (unc_signal ** 2).reshape(-1, 1)
        return mean, var
