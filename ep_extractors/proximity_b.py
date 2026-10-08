from __future__ import annotations
from typing import Optional
import numpy as np
from ep_extractors.base import BaseEpistemicExtractor
from ep_extractors import UQExtractorRegistry
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ

@UQExtractorRegistry.register("proximity_b")
class ProximityBExtractor(BaseEpistemicExtractor):
    def __init__(
        self,
        model,
        device="auto",
        decay_lambda=1.345,
        weighting: str = "unweighted_all",
        use_leaf_weights: Optional[bool] = None,
        weighted: bool = False,
        residual_mode: str = "oob",
        cv_folds: int = 5,
        **kwargs
    ):
        """
        Proximity B: Topological Tree Path Distance Proximity.
        """
        super().__init__(model)
        self.device = device
        self.decay_lambda = decay_lambda
        self.weighting = weighting
        self.use_leaf_weights = use_leaf_weights
        self.weighted = weighted
        self.residual_mode = residual_mode
        self.cv_folds = cv_folds
        self.uq_model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.uq_model = GPUProximityRegressionUQ(
            self.model,
            X_train,
            y_train,
            device=self.device,
            use_density_scaling=False,
            topological_decay_lambda=self.decay_lambda,
            weighting=self.weighting,
            use_leaf_weights=self.use_leaf_weights,
            weighted=self.weighted,
            residual_mode=self.residual_mode,
            cv_folds=self.cv_folds,
        )
        self.uq_model.fit()

    def extract_epistemic_signal(self, X: np.ndarray) -> np.ndarray:
        if self.uq_model is None:
            raise RuntimeError("Extractor must be fitted before extracting signal.")
        return self.uq_model.compute_uq(X)

    @property
    def oob_mae(self) -> float:
        if self.uq_model is not None and hasattr(self.uq_model, "oob_mae"):
            return self.uq_model.oob_mae
        return 1.0

    def predict_with_intervals(
        self,
        X: np.ndarray,
        n_neighbors: int | str = "auto",
        level: float = 0.95,
        return_mae: bool = False,
        weighting: Optional[str] = None,
        use_leaf_weights: Optional[bool] = None,
        weighted: Optional[bool] = None,
        **kwargs,
    ):
        if self.uq_model is None:
            raise RuntimeError("Extractor must be fitted before computing intervals.")
        return self.uq_model.predict_with_intervals(
            X,
            n_neighbors=n_neighbors,
            level=level,
            return_mae=return_mae,
            weighting=weighting if weighting is not None else self.weighting,
            use_leaf_weights=use_leaf_weights if use_leaf_weights is not None else self.use_leaf_weights,
            weighted=weighted if weighted is not None else self.weighted,
            **kwargs,
        )


@UQExtractorRegistry.register("proximity_b_unweighted")
class ProximityBUnweightedExtractor(ProximityBExtractor):
    def __init__(
        self,
        model,
        device="auto",
        decay_lambda=1.345,
        weighting: str = "unweighted_all",
        use_leaf_weights: Optional[bool] = None,
        weighted: bool = False,
        **kwargs,
    ):
        super().__init__(
            model,
            device=device,
            decay_lambda=decay_lambda,
            weighting=weighting,
            use_leaf_weights=use_leaf_weights,
            weighted=weighted,
            **kwargs,
        )


@UQExtractorRegistry.register("proximity_b_weighted")
class ProximityBWeightedExtractor(ProximityBExtractor):
    def __init__(
        self,
        model,
        device="auto",
        decay_lambda=1.345,
        weighting: str = "leaf_normalized",
        use_leaf_weights: Optional[bool] = None,
        weighted: bool = True,
        **kwargs,
    ):
        super().__init__(
            model,
            device=device,
            decay_lambda=decay_lambda,
            weighting=weighting,
            use_leaf_weights=use_leaf_weights,
            weighted=weighted,
            **kwargs,
        )


@UQExtractorRegistry.register("proximity_b_cv")
class ProximityBCVExtractor(ProximityBExtractor):
    def __init__(
        self,
        model,
        device="auto",
        decay_lambda=1.345,
        weighting: str = "unweighted_all",
        use_leaf_weights: Optional[bool] = None,
        weighted: bool = False,
        residual_mode: str = "cv",
        cv_folds: int = 5,
        **kwargs,
    ):
        super().__init__(
            model,
            device=device,
            decay_lambda=decay_lambda,
            weighting=weighting,
            use_leaf_weights=use_leaf_weights,
            weighted=weighted,
            residual_mode=residual_mode,
            cv_folds=cv_folds,
            **kwargs,
        )

