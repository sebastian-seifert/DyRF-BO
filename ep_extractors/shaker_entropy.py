import numpy as np
from ep_extractors.base import BaseEpistemicExtractor
from ep_extractors import UQExtractorRegistry
from Epistemic_Quantifier import EpistemicQuantifier

@UQExtractorRegistry.register("shaker_entropy")
class ShakerEntropyExtractor(BaseEpistemicExtractor):
    def __init__(self, model, num_samples=10000, batch_size="auto", backend="auto", **kwargs):
        super().__init__(model)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.backend = backend
        self.eq = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.eq = EpistemicQuantifier(self.model, X_train, y_train)

    def extract_epistemic_signal(self, X: np.ndarray) -> np.ndarray:
        if self.eq is None:
            raise RuntimeError("Extractor must be fitted before extracting signal.")
        return self.eq.shaker_get_epistemic_entropy(
            X,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            backend=self.backend
        )
