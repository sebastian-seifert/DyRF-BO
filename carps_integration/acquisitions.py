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

def normalize_max_relative(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalizes an array by its maximum value: arr / (max(arr) + eps)."""
    arr_clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = np.max(arr_clean) if len(arr_clean) > 0 else 0.0
    if max_val <= 0.0:
        return np.zeros_like(arr_clean)
    return arr_clean / (max_val + eps)

class WarmupCosineScheduler:
    """
    Schedules exploration weight beta_t over budget T using warmup + cosine annealing decay.
    """
    def __init__(
        self,
        total_trials: int = 50,
        warmup_ratio: float = 0.20,
        beta_max: float = 1.0,
        beta_min: float = 0.0
    ) -> None:
        self.total_trials = max(1, int(total_trials))
        self.warmup_ratio = float(warmup_ratio)
        self.beta_max = float(beta_max)
        self.beta_min = float(beta_min)
        self.t_warmup = int(np.floor(self.warmup_ratio * self.total_trials))

    def get_beta(self, t: int) -> float:
        if t <= self.t_warmup:
            return self.beta_max
        if t >= self.total_trials:
            return self.beta_min
        
        remaining_budget = self.total_trials - self.t_warmup
        if remaining_budget <= 0:
            return self.beta_min
            
        progress = (t - self.t_warmup) / remaining_budget
        cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(self.beta_min + (self.beta_max - self.beta_min) * cosine_factor)

class AdditiveEpistemicAcquisition(BaseAcquisitionFunction):
    """
    Decoupled Additive Epistemic Acquisition Function.
    Combines normalized base acquisition score (computed from surrogate total uncertainty)
    with normalized epistemic uncertainty bonus scaled by beta_t.
    """
    def __init__(self, base_acq: BaseAcquisitionFunction, eps: float = 1e-9):
        self.base_acq = base_acq
        self.eps = eps

    def compute(self, preds: np.ndarray, unc: np.ndarray, y_best: float) -> np.ndarray:
        return self.base_acq.compute(preds, unc, y_best)

    def compute_additive(
        self,
        preds: np.ndarray,
        unc_tot: np.ndarray,
        u_epistemic: np.ndarray,
        y_best: float,
        beta_t: float = 1.0
    ) -> np.ndarray:
        raw_base = self.base_acq.compute(preds, unc_tot, y_best)
        # Shift LCB unconditionally so min is 0.0, preserving translation invariance under max_relative normalization
        if isinstance(self.base_acq, LowerConfidenceBound):
            min_base = np.min(raw_base) if len(raw_base) > 0 else 0.0
            raw_base = raw_base - min_base
                
        norm_base = normalize_max_relative(raw_base, eps=self.eps)
        norm_ep = normalize_max_relative(u_epistemic, eps=self.eps)
        
        return norm_base + float(beta_t) * norm_ep

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

