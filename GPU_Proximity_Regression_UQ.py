import os
import sys
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# Reconfigure stdout and stderr to UTF-8 for cluster stdout logs
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


class GPUProximityRegressionUQ:
    def __init__(self, model, X_train, y_train, use_gpu=True):
        """
        GPU-Accelerated & Vectorized version of Proximity-based UQ (RF-FIRE).
        Calculates out-of-bag (OOB) matching leaf proximities and local residuals.
        
        Args:
            model: A fitted RandomForestRegressor.
            X_train: np.ndarray of shape (n_samples, n_features) representing training inputs.
            y_train: np.ndarray of shape (n_samples,) representing training targets.
            use_gpu: bool, set to True to use CuPy for GPU execution.
        """
        self.model = model
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.use_gpu = use_gpu and HAS_CUPY
        
        # Ensure model is fitted
        check_is_fitted(model)
        
        # Retrieve OOB predictions and residuals
        if not hasattr(model, "oob_prediction_"):
            raise ValueError(
                "Base model must be fitted with oob_score=True. "
                "Please configure RandomForestRegressor(oob_score=True)."
            )
            
        self.y_train_oob_pred = model.oob_prediction_
        self.oob_residuals = self.y_train - self.y_train_oob_pred
        
        self.n_train = self.X_train.shape[0]
        self.n_estimators = len(model.estimators_)
        
        # Precompute leaf matrices
        self.leaf_matrix_train = model.apply(self.X_train)
        
        # Precompute bootstrap OOB/in-bag matrices
        self.oob_indices = np.zeros((self.n_train, self.n_estimators), dtype=np.int32)
        self.in_bag_counts = np.zeros((self.n_train, self.n_estimators), dtype=np.float32)
        
        for t in range(self.n_estimators):
            tree = model.estimators_[t]
            
            # Out-of-bag samples
            oob_idx = _generate_unsampled_indices(tree.random_state, self.n_train, self.n_train)
            self.oob_indices[oob_idx, t] = 1
            
            # In-bag samples
            in_bag_idx = _generate_sample_indices(tree.random_state, self.n_train, self.n_train)
            matches, counts = np.unique(in_bag_idx, return_counts=True)
            self.in_bag_counts[matches, t] = counts

    def compute_uq(self, X_test, n_neighbors=50, level=0.95):
        """
        Computes localized uncertainty for the test set using GPU/vectorized prediction intervals.
        
        Args:
            X_test: np.ndarray of shape (n_test, n_features) representing query points.
            n_neighbors: int or 'auto', number of closest neighbors to consider.
            level: float, confidence level of prediction interval (e.g. 0.95).
            
        Returns:
            uncertainty: np.ndarray of shape (n_test,) representing interval width.
        """
        X_test = np.asarray(X_test)
        n_test = X_test.shape[0]
        leaf_matrix_test = self.model.apply(X_test)
        
        # Choose backend (CuPy or NumPy)
        xp = cp if self.use_gpu else np
        
        # Move arrays to GPU / Target device
        L = xp.asarray(self.leaf_matrix_train)
        T = xp.asarray(leaf_matrix_test)
        W = xp.asarray(self.in_bag_counts)
        oob_res = xp.asarray(self.oob_residuals)
        
        # Compute test proximities (n_test, n_train) using C-level array broadcast and reduction
        P = xp.zeros((n_test, self.n_train), dtype=xp.float32)
        
        for t in range(self.n_estimators):
            L_t = L[:, t]
            T_t = T[:, t]
            W_t = W[:, t]
            
            # Sum of in-bag counts per leaf in tree t
            leaf_sums = xp.bincount(L_t, weights=W_t)
            leaf_sums = xp.where(leaf_sums == 0, 1.0, leaf_sums)
            
            # K_t[i] is the leaf sum for leaf T_t[i]
            K_t = leaf_sums[T_t]
            
            # Proximity ratio: W[j] / K_t[i]
            ratio = W_t[xp.newaxis, :] / K_t[:, xp.newaxis]
            
            # Matching mask: T_t[i] == L_t[j]
            mask = T_t[:, xp.newaxis] == L_t[xp.newaxis, :]
            
            # Accumulate proximity contribution
            P += ratio * mask
            
        P /= self.n_estimators
        
        # Sort training samples by proximity (highest proximity first) for each test sample
        neighbor_indices = xp.argsort(-P, axis=1)
        
        # Align residuals to the proximity sorting
        tiled_res = xp.tile(oob_res, (n_test, 1))
        res_sorted = xp.take_along_axis(tiled_res, neighbor_indices, axis=1)
        
        if n_neighbors == 'auto':
            # Zero out residuals with no proximity support
            P_sorted = xp.take_along_axis(P, neighbor_indices, axis=1)
            res_sorted = xp.where(P_sorted < 1e-10, xp.nan, res_sorted)
            
            if self.use_gpu:
                # CuPy compatibility fallback to numpy for nanquantile
                res_sorted_np = cp.asnumpy(res_sorted)
                resid_lwr_np = np.nanquantile(res_sorted_np, (1 - level) / 2, axis=1)
                resid_upr_np = np.nanquantile(res_sorted_np, 1 - (1 - level) / 2, axis=1)
                resid_lwr = cp.asarray(resid_lwr_np)
                resid_upr = cp.asarray(resid_upr_np)
            else:
                resid_lwr = np.nanquantile(res_sorted, (1 - level) / 2, axis=1)
                resid_upr = np.nanquantile(res_sorted, 1 - (1 - level) / 2, axis=1)
        else:
            k = int(n_neighbors)
            res_top_k = res_sorted[:, :k]
            
            q_lwr = (1 - level) / 2
            q_upr = 1 - q_lwr
            
            if self.use_gpu:
                resid_lwr = cp.percentile(res_top_k, q_lwr * 100, axis=1)
                resid_upr = cp.percentile(res_top_k, q_upr * 100, axis=1)
            else:
                resid_lwr = np.percentile(res_top_k, q_lwr * 100, axis=1)
                resid_upr = np.percentile(res_top_k, q_upr * 100, axis=1)
                
        uncertainty = resid_upr - resid_lwr
        
        if self.use_gpu:
            return cp.asnumpy(uncertainty)
        else:
            return uncertainty
