import numpy as np
from collections import deque

class SlidingWindowRFAdaptor:
    """
    Manages sliding-window statistics of epistemic uncertainty signals
    and dynamically maps them to Random Forest hyperparameters.
    """
    def __init__(
        self,
        window_size: int = 5,
        n_base: int = 100,
        n_min: int = 10,
        n_max: int = 200,
        gamma: float = 1.0,
        depth_base: int = 12,
        depth_min: int = 5,
        depth_max: int = 30,
        beta: float = 5.0
    ):
        self.window_size = window_size
        self.n_base = n_base
        self.n_min = n_min
        self.n_max = n_max
        self.gamma = gamma
        self.depth_base = depth_base
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.beta = beta

        # FIFO queues for sliding window history
        self.q95_history = deque(maxlen=window_size)
        self.mean_scaled_history = deque(maxlen=window_size)

        # Current parameter values (initialized to base values)
        self.current_n_trees = n_base
        self.current_max_depth = depth_base

    def update_and_normalize(self, raw_signals: np.ndarray) -> np.ndarray:
        """
        Updates the window statistics with raw signals from a new candidate pool,
        normalizes the signals using the moving Hybrid Normalization Scheme,
        and computes the updated RF parameters for the next fit.
        
        Parameters:
        -----------
        raw_signals : np.ndarray
            Epistemic signals of shape (n_candidates,).
            
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

        # 4. Calculate moving average and dispersion over the window
        mu_scaled = float(np.mean(self.mean_scaled_history))
        if len(self.mean_scaled_history) > 1:
            sigma_scaled = float(np.std(self.mean_scaled_history))
        else:
            sigma_scaled = 0.0

        # 5. Adapt parameters using moving window statistics
        self.current_n_trees = int(np.clip(
            np.floor(self.n_base * (1.0 + self.gamma * mu_scaled)),
            self.n_min,
            self.n_max
        ))

        self.current_max_depth = int(np.clip(
            np.floor(self.depth_base + self.beta * sigma_scaled),
            self.depth_min,
            self.depth_max
        ))

        return scaled_signals

    def get_next_parameters(self):
        """
        Returns the adapted Random Forest parameters for the next iteration.
        
        Returns:
        --------
        n_estimators : int
        max_depth : int
        """
        return self.current_n_trees, self.current_max_depth
