import numpy as np
from ep_extractors.base import BaseEpistemicExtractor
from ep_extractors import UQExtractorRegistry
from Epistemic_Quantifier import EpistemicQuantifier

@UQExtractorRegistry.register("chen_variance")
class ChenVarianceExtractor(BaseEpistemicExtractor):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.eq = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.eq = EpistemicQuantifier(self.model, X_train, y_train)

    def extract_epistemic_signal(self, X: np.ndarray) -> np.ndarray:
        if self.eq is None:
            raise RuntimeError("Extractor must be fitted before extracting signal.")
        # Convert paired tree variance \sigma^2 to standard deviation \sigma (linear target units)
        return np.sqrt(self.eq.chen_get_epistemic_variance(X))

