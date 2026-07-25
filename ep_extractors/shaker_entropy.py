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
            
        # CONVERSION & THEORETICAL IMPLICATIONS:
        # Instead of returning raw Mutual Information in base-2 Bits (MI = H_total - H_aleatoric),
        # we convert the information-theoretic entropy into physical target standard deviation units (sigma):
        #
        #   1. Variance Inversion Formula:
        #      MI = 0.5 * log2(sigma_total^2 / sigma_aleatoric^2)
        #      ==> sigma_epistemic^2 = sigma_aleatoric^2 * (2^(2 * MI) - 1)
        #
        #   2. Standard Deviation Conversion:
        #      sigma_epistemic = sqrt(sigma_epistemic^2) = sigma_aleatoric * sqrt(4^MI - 1)
        #
        # CRITICAL THEORETICAL IMPLICATIONS:
        #  - Noise Coupling: MI is a unitless relative entropy ratio. Scaling to target units [y]
        #    requires multiplying by the baseline aleatoric noise scale sigma_aleatoric.
        #    Notice: If aleatoric noise sigma_aleatoric -> 0, sigma_epistemic -> 0 regardless of MI bits.
        #  - Exponential Inflation: For large epistemic ignorance (MI > 2 bits), 4^MI grows exponentially,
        #    creating a strong global exploration incentive in Bayesian Optimization acquisition functions (EI_epistemic).
        #  - Gaussian Inversion Assumption: The conversion assumes a single Gaussian equivalent density
        #    for the underlying multi-modal GMM entropy.
        
        epistemic_var = self.eq.shaker_get_epistemic_variance(
            X,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            backend=self.backend
        )
        return np.sqrt(epistemic_var)

