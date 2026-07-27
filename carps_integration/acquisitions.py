from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm

class BaseAcquisitionFunction(ABC):
    @abstractmethod
    def compute(self, preds: np.ndarray, unc: np.ndarray, y_best: float) -> np.ndarray:
        """Returns acquisition scores for candidates (higher score = better candidate to sample)."""
        pass

class ExpectedImprovement(BaseAcquisitionFunction):
    def __init__(self, xi: float = 0.0):
        self.xi = float(xi)

    def compute(self, preds: np.ndarray, unc: np.ndarray, y_best: float) -> np.ndarray:
        sigma = np.where(unc > 1e-9, unc, 1e-9)
        diff = (y_best - self.xi) - preds
        z = diff / sigma
        ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(unc > 1e-9, ei, np.maximum(0.0, diff))

class LowerConfidenceBound(BaseAcquisitionFunction):
    def __init__(self, beta: float = 1.96):
        self.beta = float(beta)

    def compute(self, preds: np.ndarray, unc: np.ndarray, y_best: float) -> np.ndarray:
        # For minimization, we negate (mean - beta * unc) so argmax selects lowest LCB
        return -preds + self.beta * unc

class ProbabilityOfImprovement(BaseAcquisitionFunction):
    def __init__(self, xi: float = 0.0):
        self.xi = float(xi)

    def compute(self, preds: np.ndarray, unc: np.ndarray, y_best: float) -> np.ndarray:
        sigma = np.where(unc > 1e-9, unc, 1e-9)
        z = ((y_best - self.xi) - preds) / sigma
        return norm.cdf(z)

class AcquisitionRegistry:
    _REGISTRY = {
        "ei": ExpectedImprovement,
        "lcb": LowerConfidenceBound,
        "pi": ProbabilityOfImprovement,
    }

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseAcquisitionFunction:
        name_clean = name.lower().strip()
        if name_clean not in cls._REGISTRY:
            raise ValueError(
                f"Unknown acquisition function '{name}'. Supported choices: {list(cls._REGISTRY.keys())}"
            )
        return cls._REGISTRY[name_clean](**kwargs)
