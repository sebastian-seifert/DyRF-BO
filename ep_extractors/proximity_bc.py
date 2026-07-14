import numpy as np
from ep_extractors.base import BaseEpistemicExtractor
from ep_extractors import UQExtractorRegistry
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ

@UQExtractorRegistry.register("proximity_bc")
class ProximityBCExtractor(BaseEpistemicExtractor):
    def __init__(self, model, device="cpu", decay_lambda=1.0, alpha=1.0, **kwargs):
        """
        Proximity B+C: Topological Decay Proximity with Density Scaling.
        """
        super().__init__(model)
        self.device = device
        self.decay_lambda = decay_lambda
        self.alpha = alpha
        self.uq_model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.uq_model = GPUProximityRegressionUQ(
            self.model,
            X_train,
            y_train,
            device=self.device,
            use_density_scaling=True,
            density_scaling_alpha=self.alpha,
            topological_decay_lambda=self.decay_lambda
        )
        self.uq_model.fit()

    def extract_epistemic_signal(self, X: np.ndarray) -> np.ndarray:
        if self.uq_model is None:
            raise RuntimeError("Extractor must be fitted before extracting signal.")
        return self.uq_model.compute_uq(X)
