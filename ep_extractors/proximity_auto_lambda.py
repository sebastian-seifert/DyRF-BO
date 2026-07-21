import numpy as np
from ep_extractors.base import BaseEpistemicExtractor
from ep_extractors import UQExtractorRegistry
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ

@UQExtractorRegistry.register("proximity_auto_lambda")
class ProximityAutoLambdaExtractor(BaseEpistemicExtractor):
    def __init__(self, model, device="auto", alpha=1.0, bounds=(0.001, 20.0), **kwargs):
        """
        Proximity Auto Lambda: Dynamic continuous lambda optimization via OOB NLL minimization.
        """
        super().__init__(model)
        self.device = device
        self.alpha = alpha
        self.bounds = bounds
        self.uq_model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.uq_model = GPUProximityRegressionUQ(
            self.model,
            X_train,
            y_train,
            device=self.device,
            use_density_scaling=True,
            density_scaling_alpha=self.alpha,
            topological_decay_lambda=1.0
        )
        self.uq_model.fit()
        best_lambda = self.uq_model.tune_lambda_oob(bounds=self.bounds)
        # Refit N_baseline with the tuned lambda
        self.uq_model.topological_decay_lambda = best_lambda

    def extract_epistemic_signal(self, X: np.ndarray) -> np.ndarray:
        if self.uq_model is None:
            raise RuntimeError("Extractor must be fitted before extracting signal.")
        return self.uq_model.compute_uq(X)
