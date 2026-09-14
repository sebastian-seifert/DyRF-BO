import numpy as np
from ep_extractors.base import BaseEpistemicExtractor
from ep_extractors import UQExtractorRegistry
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ

@UQExtractorRegistry.register("proximity_b")
class ProximityBExtractor(BaseEpistemicExtractor):
    def __init__(self, model, device="auto", decay_lambda=1.0, **kwargs):
        """
        Proximity B: Topological Tree Path Distance Proximity.
        """
        super().__init__(model)
        self.device = device
        self.decay_lambda = decay_lambda
        self.uq_model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.uq_model = GPUProximityRegressionUQ(
            self.model,
            X_train,
            y_train,
            device=self.device,
            use_density_scaling=False,
            topological_decay_lambda=self.decay_lambda
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
        return_mae: bool = False
    ):
        if self.uq_model is None:
            raise RuntimeError("Extractor must be fitted before computing intervals.")
        return self.uq_model.predict_with_intervals(
            X, n_neighbors=n_neighbors, level=level, return_mae=return_mae
        )
