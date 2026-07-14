import numpy as np
from ep_extractors.base import BaseEpistemicExtractor
from ep_extractors import UQExtractorRegistry
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ

@UQExtractorRegistry.register("standard_proximity")
class StandardProximityExtractor(BaseEpistemicExtractor):
    def __init__(self, model, device="cpu", **kwargs):
        """
        Proximity A: Standard Leaf Incidence Proximity.
        Co-occurrence index (topological_decay_lambda = None, use_density_scaling = False).
        """
        super().__init__(model)
        self.device = device
        self.uq_model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.uq_model = GPUProximityRegressionUQ(
            self.model,
            X_train,
            y_train,
            device=self.device,
            use_density_scaling=False,
            topological_decay_lambda=None
        )
        self.uq_model.fit()

    def extract_epistemic_signal(self, X: np.ndarray) -> np.ndarray:
        if self.uq_model is None:
            raise RuntimeError("Extractor must be fitted before extracting signal.")
        return self.uq_model.compute_uq(X)
