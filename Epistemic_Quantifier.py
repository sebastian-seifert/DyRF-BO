import os
import sys
import numpy as np
from scipy.special import logsumexp

try:
    import cupy as cp
    import cupyx
    from cupyx.scipy.special import logsumexp as cp_logsumexp
    HAS_CUPY = True
    try:
        if cp.cuda.runtime.getDeviceCount() > 0:
            smoke_test = cp.asarray([1.0])
            smoke_test = smoke_test + 1.0
            cp.cuda.Stream.null.synchronize()
            HAS_GPU = bool(cp.asnumpy(smoke_test)[0] == 2.0)
        else:
            HAS_GPU = False
    except Exception:
        HAS_GPU = False
except ImportError:
    cp = None
    cupyx = None
    cp_logsumexp = None
    HAS_CUPY = False
    HAS_GPU = False

class LeafCache:
    def __init__(self, model, X_test, means=None, variances=None, counts=None, leaf_ids=None):
        self.model = model
        if leaf_ids is not None:
            self.all_test_leaf_ids = leaf_ids
            self.means = means
            self.variances = variances
            self.counts = counts
        else:
            X_test_2d = np.atleast_2d(np.asarray(X_test))
            self.all_test_leaf_ids = model.apply(X_test_2d)
            n_samples, n_trees = self.all_test_leaf_ids.shape
            
            self.means = np.zeros((n_trees, n_samples))
            self.variances = np.zeros((n_trees, n_samples))
            self.counts = np.zeros((n_trees, n_samples))
            
            for i, estimator in enumerate(model.estimators_):
                test_leaf_ids = self.all_test_leaf_ids[:, i]
                node_means = estimator.tree_.value[:, 0, 0]
                node_impurities = estimator.tree_.impurity
                node_samples = estimator.tree_.n_node_samples
                
                self.means[i, :] = node_means[test_leaf_ids]
                n_samples_node = node_samples[test_leaf_ids]
                scale = np.where(n_samples_node > 1, n_samples_node / (n_samples_node - 1), 0.0)
                self.variances[i, :] = node_impurities[test_leaf_ids] * scale + 1e-6
                self.counts[i, :] = n_samples_node

    def get_slice(self, start, end):
        return LeafCache(
            self.model,
            None,
            means=self.means[:, start:end],
            variances=self.variances[:, start:end],
            counts=self.counts[:, start:end],
            leaf_ids=self.all_test_leaf_ids[start:end, :]
        )

class EpistemicQuantifier:
    def __init__(self, model, X_train, y_train, leaf_cache=None):
        self.model = model
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.leaf_cache = leaf_cache

    # ==========================================
    # BASE / SHARED METHODS
    # ==========================================
    def _base_calc_per_tree_variance(self, X_test, min_var=1e-6, all_test_leaf_ids=None):
        """
        Calculates the per-tree unbiased variances (sigma^2) for each sample in X_test.
        Optimized by directly retrieving pre-computed tree node impurities (MSE) 
        and scaling them to unbiased variances, bypassing slow nested CPU loops.
        
        Returns: np.array of shape (n_trees, n_samples_test)
        """
        if self.leaf_cache is not None:
            return self.leaf_cache.variances
            
        X_test = np.atleast_2d(X_test)
        n_trees = len(self.model.estimators_)
        n_samples = X_test.shape[0]
        
        # Get all leaf assignments for test data: shape (n_samples, n_trees)
        if all_test_leaf_ids is None:
            all_test_leaf_ids = self.model.apply(X_test)
        
        variances = np.zeros((n_trees, n_samples))
        
        for t, estimator in enumerate(self.model.estimators_):
            test_leaf_ids = all_test_leaf_ids[:, t]
            
            # Scikit-learn precomputes node impurity (MSE) during training
            impurity = estimator.tree_.impurity[test_leaf_ids]
            n_node_samples = estimator.tree_.n_node_samples[test_leaf_ids]
            
            # Compute unbiased variance: s^2 = impurity * (N / (N - 1))
            # If N <= 1, variance is 0.0
            scale = np.where(n_node_samples > 1, n_node_samples / (n_node_samples - 1), 0.0)
            variances[t, :] = impurity * scale + min_var
            
        return variances

    def _get_tree_predictions(self, X_test, all_test_leaf_ids=None):
        """
        Retrieves the prediction for each individual tree for each test point in a fully vectorized way.
        
        Returns: np.array of shape (n_trees, n_samples)
        """
        if self.leaf_cache is not None:
            return self.leaf_cache.means
            
        X_test = np.atleast_2d(X_test)
        if all_test_leaf_ids is None:
            all_test_leaf_ids = self.model.apply(X_test)
        n_samples = X_test.shape[0]
        n_trees = len(self.model.estimators_)
        
        tree_preds = np.zeros((n_trees, n_samples))
        for t, estimator in enumerate(self.model.estimators_):
            tree_preds[t, :] = estimator.tree_.value[all_test_leaf_ids[:, t], 0, 0]
        return tree_preds

    def base_get_aleatoric_variance(self, X_test, all_test_leaf_ids=None):
        """
        Returns the mean of the per-tree variances (E[sigma^2]).
        This is the baseline aleatoric uncertainty in terms of variance.
        """
        return np.mean(self._base_calc_per_tree_variance(X_test, all_test_leaf_ids=all_test_leaf_ids), axis=0)

    # ==========================================
    # STANDARD DISAGREEMENT
    # ==========================================
    def standard_get_epistemic_variance(self, X_test):
        """
        Approach 1: Standard Disagreement (Tree Variance)
        Captures how much the different trees disagree on the prediction.
        Higher disagreement usually indicates regions with less training data.
        
        Formula (from lecture): Var_b(E[Y|B]) = E[X^2] - (E[X])^2
        """
        # Get the prediction for each individual tree for each test point
        # tree_preds shape: (n_trees, n_test_points)
        tree_preds = self._get_tree_predictions(X_test)
        
        # Calculate E[X^2]: The mean of the squared predictions
        mean_of_squares = np.mean(tree_preds**2, axis=0)
        
        # Calculate (E[X])^2: The square of the mean prediction
        square_of_mean = (np.mean(tree_preds, axis=0))**2
        
        # Note: Mathematically identical to np.var(tree_preds, axis=0).
        # We use the expanded form for theoretical consistency with lectures.
        variance = mean_of_squares - square_of_mean
        
        # Ensure no negative variances due to floating point precision errors
        return np.maximum(variance, 0.0)

    # ==========================================
    # CHEN STABILITY
    # ==========================================
    def chen_get_epistemic_variance(self, X_test):
        """
        Approach 3: Chen 2025 (Paired Stability)
        Uses paired tree differences as a proxy for the 'safety' of the prediction.
        Directly related to generalization stability.
        
        Formula: V_chen = (1/M) * sum_{j=1}^{M/2} (h_{2j-1} - h_{2j})^2
        """
        # tree_preds shape: (n_trees, n_test_points)
        tree_preds = self._get_tree_predictions(X_test)
        M = tree_preds.shape[0]
        
        # We need an even number of trees for pairs
        if M % 2 != 0:
            tree_preds = tree_preds[:-1]
            M -= 1
            
        # Split into two sets of pairs (0,2,4... vs 1,3,5...)
        set1 = tree_preds[0::2]
        set2 = tree_preds[1::2]
        
        # Squared differences between pairs
        squared_diffs = (set1 - set2)**2
        
        # The sum is over M/2 pairs, and we divide by M total trees
        return np.sum(squared_diffs, axis=0) / M

    # ==========================================
    # SHAKER METHOD
    # ==========================================
    def shaker_get_epistemic_entropy(self, X_test, num_samples=10000, batch_size="auto", random_state=None, backend="auto"):
        """
        Approach 2: Shaker 2020 (Epistemic Component)
        Calculated as: Total Uncertainty (GMM Entropy) - Aleatoric Uncertainty.
        
        Total Uncertainty is the entropy of the Gaussian Mixture Model formed by the trees.
        Aleatoric is the mean entropy of the individual tree distributions.
        """
        X_test = np.atleast_2d(X_test)
        all_test_leaf_ids = self.model.apply(X_test)
        
        total_unc = self._shaker_calc_total_entropy(
            X_test,
            num_samples=num_samples,
            batch_size=batch_size,
            random_state=random_state,
            backend=backend,
            all_test_leaf_ids=all_test_leaf_ids
        )
        aleatoric_unc = self._shaker_calc_aleatoric_entropy(X_test, all_test_leaf_ids=all_test_leaf_ids)
        
        # Epistemic = Total - Aleatoric
        return np.maximum(total_unc - aleatoric_unc, 0.0)

    def shaker_get_epistemic_variance(self, X_test, num_samples=10000, batch_size="auto", random_state=None, backend="auto"):
        """
        Returns a Shaker-inspired epistemic proxy in variance units.

        Shaker's native decomposition is entropy-based, while Standard and Chen
        return variance-like quantities. We therefore keep the native entropy
        decomposition as mutual information in bits and map that information
        to a local variance increase relative to the aleatoric variance:

            MI = 0.5 * log2(total_var / aleatoric_var)
            epistemic_var = aleatoric_var * (2 ** (2 * MI) - 1)
        """
        X_test = np.atleast_2d(X_test)
        all_test_leaf_ids = self.model.apply(X_test)
        
        mi_bits = self.shaker_get_epistemic_entropy(
            X_test,
            num_samples=num_samples,
            batch_size=batch_size,
            random_state=random_state,
            backend=backend
        )
        aleatoric_var = self.base_get_aleatoric_variance(X_test, all_test_leaf_ids=all_test_leaf_ids)

        # Safe exponent clipping to prevent IEEE 754 float64 overflow even with large aleatoric variance
        safe_exponent = np.clip(2.0 * mi_bits, 0.0, 50.0)
        return aleatoric_var * np.maximum(2.0 ** safe_exponent - 1.0, 0.0)

    def shaker_get_total_variance(self, X_test, num_samples=10000, batch_size="auto", random_state=None, backend="auto"):
        """Converts Shaker's total GMM entropy into entropy-power variance units."""
        total_entropy = self._shaker_calc_total_entropy(
            X_test,
            num_samples=num_samples,
            batch_size=batch_size,
            random_state=random_state,
            backend=backend,
        )
        return self._shaker_convert_entropy_to_var(total_entropy)

    # --- Shaker Internals ---
    def _shaker_calc_aleatoric_entropy(self, X_test, all_test_leaf_ids=None):
        """
        Calculates the closed-form aleatoric uncertainty (mean differential entropy).
        Formula from Slide 4: (1/M) * sum( 0.5 * log2(2 * pi * e * sigma_hat^2) )
        """
        if all_test_leaf_ids is None:
            all_test_leaf_ids = self.model.apply(X_test)
            
        vars2 = self._base_calc_per_tree_variance(X_test, all_test_leaf_ids=all_test_leaf_ids) 
        individual_entropies = 0.5 * np.log2(2 * np.pi * np.e * vars2)
        return np.mean(individual_entropies, axis=0)

    def _shaker_convert_entropy_to_var(self, entropy_bits):
        """Converts differential entropy in bits to the variance of a Gaussian."""
        # Safe exponent clipping to prevent IEEE 754 float64 overflow
        safe_exponent = np.clip(2.0 * entropy_bits, -50.0, 50.0)
        return (2.0 ** safe_exponent) / (2.0 * np.pi * np.e)

    def _shaker_convert_var_to_entropy(self, var):
        """Converts variance of a Gaussian to differential entropy in bits."""
        return 0.5 * np.log2(2.0 * np.pi * np.e * var)

    def _shaker_calc_total_entropy(self, X_test, num_samples=10000, batch_size="auto", random_state=None, backend="auto", all_test_leaf_ids=None):
        r"""
        Calculates the Total Uncertainty (Entropy of the GMM) via fully vectorized
        1D deterministic trapezoidal quadrature over batches of test query points.
        
        Formula: H = \int -p(y) \log_2 p(y) dy
        """
        import time
        debug_timing = os.environ.get("PROXIMITY_DEBUG") == "1"
        if debug_timing:
            t0 = time.time()
            
        X_test = np.atleast_2d(X_test)
        n_samples = X_test.shape[0]
        n_trees = len(self.model.estimators_)
        
        backend = self._mc_resolve_backend(backend)
        is_gpu = backend == "gpu"
        xp = cp if is_gpu else np
        
        n_grid = 128
        
        if batch_size == "auto":
            batch_size = self._get_dynamic_shaker_batch_size(n_grid, n_trees, backend)
            if debug_timing:
                print(f"Dynamically resolved Shaker batch size: {batch_size}")
        
        if all_test_leaf_ids is None:
            if self.leaf_cache is not None:
                all_test_leaf_ids = self.leaf_cache.all_test_leaf_ids
            else:
                all_test_leaf_ids = self.model.apply(X_test)
            
        if self.leaf_cache is not None:
            mu_all = self.leaf_cache.means
            vars_all = self.leaf_cache.variances
        else:
            mu_all = self._get_tree_predictions(X_test, all_test_leaf_ids=all_test_leaf_ids)
            vars_all = self._base_calc_per_tree_variance(X_test, all_test_leaf_ids=all_test_leaf_ids)
            
        sigmas_all = np.sqrt(vars_all)
        
        total_entropy = np.zeros(n_samples)
        
        if is_gpu:
            mu_g = cp.asarray(mu_all)
            sigmas_g = cp.asarray(sigmas_all)
        else:
            mu_g = mu_all
            sigmas_g = sigmas_all
            
        start = 0
        while start < n_samples:
            end = min(start + batch_size, n_samples)
            try:
                mu_batch = mu_g[:, start:end]
                sigma_batch = sigmas_g[:, start:end]
                B = end - start
                
                # Define integration range per sample: y_min, y_max of shape (B,)
                y_min = xp.min(mu_batch - 6.0 * sigma_batch, axis=0)
                y_max = xp.max(mu_batch + 6.0 * sigma_batch, axis=0)
                
                # Setup 1D grid per sample: shape (B, n_grid)
                grid_steps = xp.linspace(0.0, 1.0, n_grid)
                y_grid = y_min[:, xp.newaxis] + grid_steps[xp.newaxis, :] * (y_max - y_min)[:, xp.newaxis]
                dy = (y_max - y_min) / (n_grid - 1)
                
                # Reshape for multi-dimensional broadcasting:
                # y_b: (1, B, n_grid)
                # mu_b: (n_trees, B, 1)
                # sigma_b: (n_trees, B, 1)
                y_b = y_grid[xp.newaxis, :, :]
                mu_b = mu_batch[:, :, xp.newaxis]
                sigma_b = sigma_batch[:, :, xp.newaxis]
                
                # Compute component PDF: shape (n_trees, B, n_grid)
                z = (y_b - mu_b) / sigma_b
                inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
                pdf_comp = (xp.exp(-0.5 * z**2) * inv_sqrt_2pi) / sigma_b
                
                # Mixture PDF p(y): shape (B, n_grid)
                p_y = xp.mean(pdf_comp, axis=0)
                p_y_safe = xp.maximum(p_y, 1e-300)
                
                # Integrand: -p(y) * log2(p(y))
                integrand = -p_y * xp.log2(p_y_safe)
                
                # Trapezoidal integration
                trapz_weights = xp.ones(n_grid)
                trapz_weights[0] = 0.5
                trapz_weights[-1] = 0.5
                
                batch_entropy = xp.sum(integrand * trapz_weights[xp.newaxis, :], axis=1) * dy
                
                if is_gpu:
                    total_entropy[start:end] = cp.asnumpy(batch_entropy)
                else:
                    total_entropy[start:end] = batch_entropy
                    
                start += B
            except Exception as e:
                # Catch GPU/CPU memory errors
                is_oom = False
                if is_gpu and cp is not None:
                    if isinstance(e, cp.cuda.memory.OutOfMemoryError):
                        is_oom = True
                if isinstance(e, MemoryError):
                    is_oom = True
                    
                if is_oom:
                    if is_gpu and cp is not None:
                        cp.get_default_memory_pool().free_all_blocks()
                    if batch_size <= 10:
                        raise RuntimeError("OOM even with batch size <= 10")
                    batch_size = max(10, batch_size // 2)
                    if debug_timing:
                        print(f"   [GMM Shaker Profile] OOM encountered. Halving batch size to {batch_size}")
                else:
                    raise e
                    
        if debug_timing:
            print(f"   [GMM Shaker Profile] Total total_entropy calculation took: {time.time() - t0:.6f}s")
            
        return total_entropy

    def _mc_is_cupy_available(self):
        return HAS_GPU

    def _mc_resolve_backend(self, backend):
        if backend not in {"auto", "cpu", "gpu"}:
            raise ValueError("backend must be one of: 'auto', 'cpu', 'gpu'")
        if backend == "auto":
            return "gpu" if HAS_GPU else "cpu"
        if backend == "gpu" and not HAS_GPU:
            return "cpu"
        return backend

    def _get_dynamic_shaker_batch_size(self, n_grid, n_trees, backend):
        if backend == "gpu" and cp is not None:
            try:
                free_mem, _ = cp.cuda.Device().mem_info
                # Target using 35% of free memory
                target_mem_bytes = free_mem * 0.35
                # We need ~4 tensors of shape (n_trees, B, n_grid) of float32
                bytes_per_sample = n_trees * n_grid * 4 * 4
                batch_size = int(target_mem_bytes / bytes_per_sample)
                # Restrict to a safe range: [100, 30000]
                return int(np.clip(batch_size, 100, 30000))
            except Exception:
                return 1000
        else:
            return 2000
