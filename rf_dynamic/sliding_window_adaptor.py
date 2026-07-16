import numpy as np
from collections import deque

class SlidingWindowRFAdaptor:
    """
    Manages sliding-window statistics of epistemic uncertainty signals
    and dynamically maps them to Random Forest hyperparameters (min_samples_leaf, max_features).
    """
    def __init__(
        self,
        window_size: int = 5,
        min_samples_leaf_base: int = 2,
        min_samples_leaf_min: int = 1,
        min_samples_leaf_max: int = 15,
        alpha: float = 1.0,
        max_features_base: float = 0.5,
        max_features_min: float = 0.1,
        max_features_max: float = 0.8,
        eta: float = 0.5
    ):
        self.window_size = window_size
        
        self.min_samples_leaf_base = min_samples_leaf_base
        self.min_samples_leaf_min = min_samples_leaf_min
        self.min_samples_leaf_max = min_samples_leaf_max
        self.alpha = alpha
        
        self.max_features_base = max_features_base
        self.max_features_min = max_features_min
        self.max_features_max = max_features_max
        self.eta = eta

        # FIFO queues for sliding window history
        self.q95_history = deque(maxlen=window_size)
        self.mean_scaled_history = deque(maxlen=window_size)

        # Current parameter values (initialized to base values)
        self.current_min_samples_leaf = min_samples_leaf_base
        self.current_max_features = max_features_base

    def update_and_normalize(self, raw_signals: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Updates the window statistics with raw signals from a new candidate pool,
        normalizes the signals using the moving Hybrid Normalization Scheme,
        and computes the updated RF parameters for the next fit.
        
        Parameters:
        -----------
        raw_signals : np.ndarray
            Epistemic signals of shape (n_candidates,).
        n_samples : int
            Current number of training samples (to dynamically cap min_samples_leaf).
            
        Returns:
        --------
        scaled_signals : np.ndarray
            Normalized signals of shape (n_candidates,) clipped to [0, 1].
        """
        if len(raw_signals) == 0:
            return raw_signals

        # 1. Compute 95th percentile of the current step
        q95 = float(np.percentile(raw_signals, 95))
        self.q95_history.append(q95)

        # 2. Hybrid Normalization Scheme (divide by max q95 in window history)
        max_q95 = max(self.q95_history)
        if max_q95 > 1e-9:
            scaled_signals = np.minimum(1.0, raw_signals / max_q95)
        else:
            scaled_signals = np.minimum(1.0, raw_signals)

        # Ensure non-negative
        scaled_signals = np.maximum(0.0, scaled_signals)

        # 3. Store iteration-level mean of scaled signals
        mean_scaled = float(np.mean(scaled_signals))
        self.mean_scaled_history.append(mean_scaled)

        # 4. Calculate moving average over the window
        mu_scaled = float(np.mean(self.mean_scaled_history))

        # 5. Adapt parameters using moving window statistics
        # min_samples_leaf scales with average uncertainty
        # Dynamically cap the max leaf size at 25% of training samples (min 1) to prevent stump collapse
        leaf_upper_bound = min(self.min_samples_leaf_max, max(1, n_samples // 4))
        
        raw_leaf = np.floor(self.min_samples_leaf_base * (1.0 + self.alpha * mu_scaled))
        self.current_min_samples_leaf = int(np.clip(
            raw_leaf,
            self.min_samples_leaf_min,
            leaf_upper_bound
        ))

        # max_features scales inversely with average uncertainty
        raw_features = self.max_features_base * (1.0 - self.eta * mu_scaled)
        self.current_max_features = float(np.clip(
            raw_features,
            self.max_features_min,
            self.max_features_max
        ))

        return scaled_signals

    def get_next_parameters(self):
        """
        Returns the adapted Random Forest parameters for the next iteration.
        
        Returns:
        --------
        min_samples_leaf : int
        max_features : float
        """
        return self.current_min_samples_leaf, self.current_max_features
