import os
import sys
import warnings
import numpy as np

# Reconfigure stdout and stderr to UTF-8 to prevent encoding issues in cluster environments
if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
if sys.stderr is not None:
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

class GPUProximityRegressionUQ:
    def __init__(self, model, X_train, y_train, device="auto", batch_size=256):
        """
        GPU-Accelerated Wrapper for Localized Uncertainty Quantification in Random Forests
        via Proximities (RF-FIRE / RF-GAP). Supports dynamic NumPy and CuPy backends.
        
        Args:
            model: A fitted or unfitted RandomForestRegressor.
            X_train: array-like of shape (n_samples, n_features), training inputs.
            y_train: array-like of shape (n_samples,), training targets.
            device: str, "cpu", "gpu" (or "cuda"), or "auto".
                "auto" enables CuPy if a GPU and CuPy are available.
            batch_size: int, size of chunked batches for test point processing.
                Controls peak memory consumption during vectorized proximity computation.
        """
        self.model = model
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.n_train = len(self.X_train)
        self.batch_size = batch_size
        
        # Configure backend dynamically
        self._init_backend(device)
        
        # If the model is already fitted, we can run fit() to extract necessary parameters
        if hasattr(self.model, "estimators_"):
            self.fit()

    def _init_backend(self, device):
        """Initializes the backend (NumPy or CuPy) dynamically."""
        device = device.lower()
        self.using_gpu = False
        
        if device in ("gpu", "cuda"):
            if HAS_CUPY:
                try:
                    # Quick check to verify GPU visibility/accessibility
                    device_count = cp.cuda.runtime.getDeviceCount()
                    if device_count > 0:
                        self.using_gpu = True
                    else:
                        warnings.warn("CuPy is installed but no GPUs were detected. Falling back to CPU.")
                except Exception as e:
                    warnings.warn(f"Failed to initialize CUDA device via CuPy: {e}. Falling back to CPU.")
            else:
                warnings.warn("GPU backend requested but CuPy is not installed. Falling back to CPU.")
        elif device == "auto":
            if HAS_CUPY:
                try:
                    if cp.cuda.runtime.getDeviceCount() > 0:
                        self.using_gpu = True
                except Exception:
                    pass
        
        if self.using_gpu:
            self.xp = cp
            # Check for nanquantile support in the current CuPy version
            try:
                _ = cp.nanquantile(cp.array([1.0, cp.nan]), 0.5)
                self.nanquantile_supported = True
            except (AttributeError, NotImplementedError):
                self.nanquantile_supported = False
        else:
            self.xp = np
            self.nanquantile_supported = True

    def fit(self):
        """
        Fits the underlying Random Forest model if not already fitted,
        extracts OOB residual statistics, and prepares internal structures.
        """
        if not hasattr(self.model, "estimators_"):
            # Ensure oob_score=True is enabled to collect out-of-bag statistics
            self.model.set_params(oob_score=True)
            self.model.fit(self.X_train, self.y_train)
            
        if not getattr(self.model, "oob_score", False):
            raise ValueError("The provided RandomForestRegressor must have oob_score=True.")
            
        self.estimators = self.model.estimators_
        self.n_estimators = len(self.estimators)
        
        # Extract out-of-bag predictions and compute residuals
        self.oob_prediction_ = self.model.oob_prediction_
        self.oob_residuals = self.y_train - self.oob_prediction_
        
        # Import underlying scikit-learn helper functions to reconstruct OOB indices and in-bag counts
        from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices
        
        self.oob_indices = np.zeros((self.n_train, self.n_estimators), dtype=np.int32)
        self.in_bag_counts = np.zeros((self.n_train, self.n_estimators), dtype=np.int32)
        
        for t, tree in enumerate(self.estimators):
            # 1. Unsampled indices = Out-of-Bag (OOB) samples
            oob_idx = _generate_unsampled_indices(tree.random_state, self.n_train, self.n_train)
            self.oob_indices[oob_idx, t] = 1
            
            # 2. Sampled indices = In-bag samples (with frequency counts)
            ib_idx = _generate_sample_indices(tree.random_state, self.n_train, self.n_train)
            idx, counts = np.unique(ib_idx, return_counts=True)
            self.in_bag_counts[idx, t] = counts
            
        self.in_bag_indices = 1 - self.oob_indices
        self.leaf_matrix_train = self.model.apply(self.X_train)
        
        # in_bag_leaves keeps the leaf ID for in-bag samples, and sets to 0 for OOB samples
        self.in_bag_leaves = self.in_bag_indices * self.leaf_matrix_train
        
        # Transfer training structures to the active backend (numpy or cupy)
        self.oob_residuals_xp = self.xp.asarray(self.oob_residuals)
        self.in_bag_leaves_xp = self.xp.asarray(self.in_bag_leaves)
        self.in_bag_counts_xp = self.xp.asarray(self.in_bag_counts)

    def compute_uq(self, X_test, n_neighbors="auto", level=0.95):
        """
        Computes localized uncertainty quantification (interval width) for test query points.
        
        Args:
            X_test: array-like of shape (n_test_samples, n_features), query points.
            n_neighbors: int, 'auto', or 'all'. Number of nearest neighbors to consider.
            level: float, confidence level of the prediction interval.
            
        Returns:
            uncertainty: np.ndarray of shape (n_test_samples,), the prediction interval width.
        """
        # Ensure model is fitted and structures are prepared
        if not hasattr(self, "estimators"):
            self.fit()
            
        X_test = np.asarray(X_test)
        n_test = len(X_test)
        
        # Apply the trees to test points to get leaf IDs
        leaf_matrix_test = self.model.apply(X_test)
        leaf_matrix_test_xp = self.xp.asarray(leaf_matrix_test)
        
        # Pre-allocate output arrays on the backend
        resid_lwr = self.xp.zeros(n_test, dtype=self.xp.float32)
        resid_upr = self.xp.zeros(n_test, dtype=self.xp.float32)
        
        alpha_lwr = (1.0 - level) / 2.0
        alpha_upr = 1.0 - alpha_lwr
        
        # Process in batches to control GPU/CPU memory consumption
        for start in range(0, n_test, self.batch_size):
            end = min(start + self.batch_size, n_test)
            batch_len = end - start
            
            leaf_batch = leaf_matrix_test_xp[start:end, :]  # (batch_size, n_estimators)
            
            # Vectorized RF-GAP proximity calculation:
            # 1. Check if test point falls in the same leaf as in-bag training samples
            # broadcast comparison: (batch_size, 1, n_estimators) == (1, n_train, n_estimators)
            matches = leaf_batch[:, None, :] == self.in_bag_leaves_xp[None, :, :]  # (batch_size, n_train, n_estimators)
            
            # 2. Scale matches by the training sample's frequency count in that tree
            matched_counts = self.xp.where(matches, self.in_bag_counts_xp[None, :, :], 0.0)
            
            # 3. Sum total in-bag counts in each leaf (partition size)
            ks = self.xp.sum(matched_counts, axis=1, keepdims=True)  # (batch_size, 1, n_estimators)
            ks = self.xp.where(ks == 0, 1.0, ks)  # Avoid division by zero
            
            # 4. Compute batch proximity matrix: shape (batch_size, n_train)
            prox_batch = self.xp.sum(matched_counts / ks, axis=2) / self.n_estimators
            
            # 5. Extract residual quantiles based on proximity neighbors
            tiled_residuals = self.xp.tile(self.oob_residuals_xp[None, :], (batch_len, 1))
            
            if n_neighbors == "auto":
                # Mask out training samples with proximity < 1e-10
                tiled_residuals[prox_batch < 1e-10] = self.xp.nan
                
                # Perform nanquantile estimation
                if self.using_gpu and not self.nanquantile_supported:
                    # Fall back to NumPy CPU for nanquantile if CuPy version lacks it
                    tiled_cpu = cp.asnumpy(tiled_residuals)
                    lwr_cpu = np.nanquantile(tiled_cpu, alpha_lwr, axis=1)
                    upr_cpu = np.nanquantile(tiled_cpu, alpha_upr, axis=1)
                    resid_lwr[start:end] = cp.asarray(lwr_cpu)
                    resid_upr[start:end] = cp.asarray(upr_cpu)
                else:
                    resid_lwr[start:end] = self.xp.nanquantile(tiled_residuals, alpha_lwr, axis=1)
                    resid_upr[start:end] = self.xp.nanquantile(tiled_residuals, alpha_upr, axis=1)
            else:
                k = self.n_train if n_neighbors == "all" else int(n_neighbors)
                # Sort neighbors by proximity (descending) to match RFGAP tie-breaking exactly
                # Use argsort to match the reference implementation's tie-breaking behavior exactly
                partition_idx = self.xp.flip(self.xp.argsort(prox_batch, axis=1), axis=1)[:, :k]
                
                # Extract corresponding residuals
                k_residuals = self.xp.take_along_axis(tiled_residuals, partition_idx, axis=1)
                
                resid_lwr[start:end] = self.xp.quantile(k_residuals, alpha_lwr, axis=1)
                resid_upr[start:end] = self.xp.quantile(k_residuals, alpha_upr, axis=1)
                
        # Calculate localized prediction interval width (uncertainty)
        uq = resid_upr - resid_lwr
        
        # If using GPU, convert result back to a standard NumPy array for Scikit-Learn compatibility
        if self.using_gpu:
            uq = uq.get()
            
        return uq
