import numpy as np
from ep_extractors.base import BaseEpistemicExtractor
from ep_extractors import UQExtractorRegistry
from Credal_Regression_UQ import CredalRegressionUQ

@UQExtractorRegistry.register("likelihood_credal")
class LikelihoodCredalExtractor(BaseEpistemicExtractor):
    def __init__(self, model, backend="auto", n_grid=100, integration_method="gauss_legendre", **kwargs):
        super().__init__(model)
        self.backend = backend
        self.n_grid = n_grid
        self.integration_method = integration_method
        self.cruq = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.cruq = CredalRegressionUQ(self.model, X_train, y_train)

    def extract_epistemic_signal(self, X: np.ndarray) -> np.ndarray:
        if self.cruq is None:
            raise RuntimeError("Extractor must be fitted before extracting signal.")
        epistemic_var, _ = self.cruq.compute_uq(
            X,
            backend=self.backend,
            n_grid=self.n_grid,
            integration_method=self.integration_method
        )
        return epistemic_var
