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

class EpistemicQuantifier:
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

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

        return aleatoric_var * np.maximum(2.0 ** (2.0 * mi_bits) - 1.0, 0.0)

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
        vars2 = self._base_calc_per_tree_variance(X_test, all_test_leaf_ids=all_test_leaf_ids) 
        individual_entropies = 0.5 * np.log2(2 * np.pi * np.e * vars2)
        return np.mean(individual_entropies, axis=0)

    def _shaker_convert_entropy_to_var(self, entropy_bits):
        """Converts differential entropy in bits to the variance of a Gaussian."""
        return (2.0 ** (2.0 * entropy_bits)) / (2.0 * np.pi * np.e)

    def _shaker_calc_total_entropy(self, X_test, num_samples=10000, batch_size="auto", random_state=None, backend="auto", all_test_leaf_ids=None):
        """
        Calculates the Total Uncertainty (Entropy of the GMM) via fully vectorized
        Monte Carlo estimation over batches of test query points.
        
        Formula: E[-log2(p(y|x))], with y sampled from the tree GMM.
        """
        import time
        debug_timing = os.environ.get("PROXIMITY_DEBUG") == "1"
        if debug_timing:
            t0 = time.time()
            
        X_test = np.atleast_2d(X_test)
        n_samples = X_test.shape[0]
        n_trees = len(self.model.estimators_)
        
        backend = self._mc_resolve_backend(backend)
        print(f"Vectorized Monte Carlo backend: {backend}")
        
        if batch_size == "auto":
            batch_size = self._get_dynamic_shaker_batch_size(num_samples, n_trees, backend)
            print(f"Dynamically resolved Shaker batch size: {batch_size}")
        
        if all_test_leaf_ids is None:
            all_test_leaf_ids = self.model.apply(X_test)
            
        # 1. Get predictions and variances for all trees: shapes (n_samples, n_trees)
        if debug_timing:
            t_pred_start = time.time()
        mu_all = self._get_tree_predictions(X_test, all_test_leaf_ids=all_test_leaf_ids).T # (n_samples, n_trees)
        vars_all = self._base_calc_per_tree_variance(X_test, all_test_leaf_ids=all_test_leaf_ids).T # (n_samples, n_trees)
        sigmas_all = np.sqrt(vars_all)
        if debug_timing:
            t_pred_end = time.time()
            print(f"   [GMM Shaker Profile] Tree predictions and variance retrieval took: {t_pred_end - t_pred_start:.6f}s")
        
        total_entropy = np.zeros(n_samples)
        
        if backend == "gpu":
            if debug_timing:
                t_prep_start = time.time()
            # Copy to pinned memory for fast DMA transfers
            mu_all_pinned = cupyx.empty_pinned(mu_all.shape, dtype=np.float32)
            sigmas_all_pinned = cupyx.empty_pinned(sigmas_all.shape, dtype=np.float32)
            mu_all_pinned[...] = mu_all
            sigmas_all_pinned[...] = sigmas_all

            cp.get_default_memory_pool().free_all_blocks()
            rng = self._mc_make_gpu_rng(random_state)
            if debug_timing:
                print(f"   [GMM Shaker Profile] GPU/DMA buffer setups took: {time.time() - t_prep_start:.6f}s")
                t_loop_start = time.time()
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                B = end - start
                
                # Move this batch of mu and sigma to the GPU in float32 from pinned memory
                mu_batch = cp.asarray(mu_all_pinned[start:end, :], dtype=cp.float32)
                sigma_batch = cp.asarray(sigmas_all_pinned[start:end, :], dtype=cp.float32)
                
                # Sample tree components: shape (B, num_samples)
                if hasattr(rng, "integers"):
                    components = rng.integers(0, n_trees, size=(B, num_samples))
                else:
                    components = rng.randint(0, n_trees, size=(B, num_samples))
                
                # Sample standard normals: shape (B, num_samples)
                eps = self._mc_gpu_standard_normal(rng, (B, num_samples)).astype(cp.float32)
                
                # Select the mean and std dev corresponding to the sampled components
                batch_indices = cp.arange(B)[:, None]
                mu_sampled = mu_batch[batch_indices, components]
                sigma_sampled = sigma_batch[batch_indices, components]
                
                # Generate sample targets: shape (B, num_samples)
                y_samples = mu_sampled + sigma_sampled * eps
                
                # Evaluate log GMM probability in float32 to reduce memory footprint:
                y_expanded = y_samples[:, :, None] # (B, num_samples, 1)
                mu_expanded = mu_batch[:, None, :] # (B, 1, n_trees)
                sigma_expanded = sigma_batch[:, None, :] # (B, 1, n_trees)
                
                # Precompute constant log factor
                log_const = cp.log(sigma_batch * cp.sqrt(2 * cp.pi)) # (B, n_trees)
                log_const_expanded = log_const[:, None, :] # (B, 1, n_trees)
                
                # Perform in-place operations to avoid massive memory allocations
                diff = y_expanded - mu_expanded # (B, num_samples, n_trees)
                diff /= sigma_expanded
                diff **= 2
                diff *= -0.5
                diff -= log_const_expanded
                
                log_p_y = cp_logsumexp(diff, axis=2) - cp.log(n_trees)
                batch_entropy = -cp.mean(log_p_y, axis=1) / cp.log(2)
                
                total_entropy[start:end] = cp.asnumpy(batch_entropy)
            if debug_timing:
                print(f"   [GMM Shaker Profile] GPU MC Batch execution loop took: {time.time() - t_loop_start:.6f}s")
                
        else: # CPU backend
            rng = self._mc_make_cpu_rng(random_state)
            if debug_timing:
                t_loop_start = time.time()
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                B = end - start
                
                mu_batch = mu_all[start:end, :].astype(np.float32)
                sigma_batch = sigmas_all[start:end, :].astype(np.float32)
                
                components = rng.integers(0, n_trees, size=(B, num_samples))
                eps = rng.normal(0, 1, size=(B, num_samples)).astype(np.float32)
                
                batch_indices = np.arange(B)[:, None]
                mu_sampled = mu_batch[batch_indices, components]
                sigma_sampled = sigma_batch[batch_indices, components]
                
                y_samples = mu_sampled + sigma_sampled * eps
                
                y_expanded = y_samples[:, :, None]
                mu_expanded = mu_batch[:, None, :]
                sigma_expanded = sigma_batch[:, None, :]
                
                # Precompute constant log factor
                log_const = np.log(sigma_batch * np.sqrt(2 * np.pi)) # (B, n_trees)
                log_const_expanded = log_const[:, None, :] # (B, 1, n_trees)
                
                # Perform in-place operations to avoid massive memory allocations
                diff = y_expanded - mu_expanded # (B, num_samples, n_trees)
                diff /= sigma_expanded
                diff **= 2
                diff *= -0.5
                diff -= log_const_expanded
                
                log_p_y = logsumexp(diff, axis=2) - np.log(n_trees)
                batch_entropy = -np.mean(log_p_y, axis=1) / np.log(2)
                
                total_entropy[start:end] = batch_entropy
                
            if debug_timing:
                print(f"   [GMM Shaker Profile] CPU MC Batch execution loop took: {time.time() - t_loop_start:.6f}s")
                
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
            print("CuPy/CUDA is not available. Falling back to CPU Monte Carlo.")
            return "cpu"
        return backend

    def _mc_make_gpu_rng(self, random_state):
        if hasattr(cp.random, "default_rng"):
            return cp.random.default_rng(random_state)
        return cp.random.RandomState(random_state)

    def _mc_make_cpu_rng(self, random_state):
        return np.random.default_rng(random_state)

    def _mc_gpu_standard_normal(self, rng, size):
        if hasattr(rng, "standard_normal"):
            return rng.standard_normal(size=size)
        return cp.random.standard_normal(size=size)

    def _get_dynamic_shaker_batch_size(self, num_samples, n_trees, backend):
        if backend == "gpu" and cp is not None:
            try:
                free_mem, _ = cp.cuda.Device().mem_info
                # We target using no more than 1% of free memory to be extremely safe in multi-job/shared environments
                target_mem_bytes = free_mem * 0.01
                # Each element in float32 takes 4 bytes. We need ~10 tensors of size (B, num_samples, n_trees)
                # due to intermediate arrays created in expressions and cached by CuPy's memory pool:
                # 1. z_samples (B, num_samples, n_trees)
                # 2. log_prob_components (B, num_samples, n_trees)
                # 3. intermediate arithmetic allocations (sub, div, pow, mul)
                bytes_per_sample = num_samples * n_trees * 4 * 10
                batch_size = int(target_mem_bytes / bytes_per_sample)
                # Restrict to a safe range: [10, 1000]
                return int(np.clip(batch_size, 10, 1000))
            except Exception:
                return 200
        else:
            # CPU backend has L3 cache limitations. For 10,000 samples, keeping the batch small
            # (e.g. 100 to 500) fits the memory footprint within L3 cache slices, avoiding page thrashing.
            return 200
