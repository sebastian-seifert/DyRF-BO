import os
import sys
import numpy as np
from scipy.special import logsumexp, roots_hermite

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
            denom = np.maximum(n_node_samples - 1.0, 1.0)
            scale = np.where(n_node_samples > 1, n_node_samples / denom, 0.0)
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
    # Approach 2: Shaker 2020 (Entropy-based)
    # ==========================================
    def shaker_get_epistemic_entropy(self, X_test, num_samples=10000, batch_size="auto", random_state=None, backend="auto", n_quadrature_points=32, method="gauss_hermite"):
        """
        Approach 2: Shaker 2020 (Epistemic Component)
        Calculated as: Total Uncertainty (GMM Entropy) - Aleatoric Uncertainty.
        
        Total Uncertainty is the entropy of the Gaussian Mixture Model formed by the trees.
        Aleatoric is the mean entropy of the individual tree distributions.
        """
        X_test = np.atleast_2d(X_test)
        all_test_leaf_ids = self.model.apply(X_test)
        n_trees = len(self.model.estimators_)
        max_mi_bound = float(np.log2(n_trees))
        
        total_unc = self._shaker_calc_total_entropy(
            X_test,
            num_samples=num_samples,
            batch_size=batch_size,
            random_state=random_state,
            backend=backend,
            all_test_leaf_ids=all_test_leaf_ids,
            n_quadrature_points=n_quadrature_points,
            method=method
        )
        aleatoric_unc = self._shaker_calc_aleatoric_entropy(X_test, all_test_leaf_ids=all_test_leaf_ids)
        
        raw_mi = total_unc - aleatoric_unc

        # Enforce theoretical bounds: 0 <= MI <= log2(n_trees)
        if np.any(raw_mi > max_mi_bound + 1e-4):
            import warnings
            max_viol = float(np.max(raw_mi - max_mi_bound))
            warnings.warn(
                f"Numerical integration exceeded information-theoretic MI upper bound log2({n_trees})={max_mi_bound:.4f} by {max_viol:.4f} bits. Clamping to bound.",
                RuntimeWarning
            )

        return np.clip(raw_mi, 0.0, max_mi_bound)

    def shaker_get_epistemic_variance(self, X_test, num_samples=10000, batch_size="auto", random_state=None, backend="auto", n_quadrature_points=32, method="gauss_hermite"):
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
            backend=backend,
            n_quadrature_points=n_quadrature_points,
            method=method
        )
        aleatoric_var = self.base_get_aleatoric_variance(X_test, all_test_leaf_ids=all_test_leaf_ids)

        # Safe exponent clipping to prevent IEEE 754 float64 overflow even with large aleatoric variance
        safe_exponent = np.clip(2.0 * mi_bits, 0.0, 50.0)
        return aleatoric_var * np.maximum(2.0 ** safe_exponent - 1.0, 0.0)

    def shaker_get_total_variance(self, X_test, num_samples=10000, batch_size="auto", random_state=None, backend="auto", n_quadrature_points=32, method="gauss_hermite"):
        """Converts Shaker's total GMM entropy into entropy-power variance units."""
        total_entropy = self._shaker_calc_total_entropy(
            X_test,
            num_samples=num_samples,
            batch_size=batch_size,
            random_state=random_state,
            backend=backend,
            n_quadrature_points=n_quadrature_points,
            method=method
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

    def _shaker_calc_total_entropy(self, X_test, num_samples=10000, batch_size="auto", random_state=None, backend="auto", all_test_leaf_ids=None, n_quadrature_points=32, method="gauss_hermite"):
        r"""
        Calculates the Total Uncertainty (Entropy of the GMM) via component-resolving
        Gauss-Hermite quadrature or Monte Carlo sampling over batches of test query points.
        
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
            
        vars_all = np.maximum(vars_all, 1e-6)
        sigmas_all = np.sqrt(vars_all)
        
        total_entropy = np.zeros(n_samples)
        
        if method == "monte_carlo":
            rng = np.random.default_rng(random_state)
            mc_batch_size = 50 if batch_size == "auto" else batch_size
            for start in range(0, n_samples, mc_batch_size):
                end = min(start + mc_batch_size, n_samples)
                B = end - start
                mu_batch = mu_all[:, start:end]
                sigma_batch = sigmas_all[:, start:end]
                
                tree_idx = rng.integers(0, n_trees, size=(num_samples, B))
                m_chosen = np.take_along_axis(mu_batch, tree_idx, axis=0)
                s_chosen = np.take_along_axis(sigma_batch, tree_idx, axis=0)
                
                y_samples = m_chosen + s_chosen * rng.standard_normal(size=(num_samples, B))
                
                u = (y_samples[:, None, :] - mu_batch[None, :, :]) / sigma_batch[None, :, :]
                log_comp = -0.5 * u**2 - np.log(sigma_batch[None, :, :]) - 0.5 * np.log(2.0 * np.pi)
                log_py = logsumexp(log_comp, axis=1) - np.log(n_trees)
                log2_py = log_py / np.log(2.0)
                
                total_entropy[start:end] = - np.mean(log2_py, axis=0)
            return total_entropy

        # Gauss-Hermite component-resolving quadrature
        K = n_quadrature_points
        z_gh, w_gh = roots_hermite(K)
        
        if batch_size == "auto":
            bytes_per_sample = n_trees * K * n_trees * 8
            calc_b = max(1, min(64, int(50_000_000 / (bytes_per_sample + 1))))
            batch_size = calc_b
            if debug_timing:
                print(f"Dynamically resolved Shaker GH batch size: {batch_size}")
                
        if is_gpu and cp is not None and cp_logsumexp is not None:
            z_g = cp.asarray(z_gh)
            w_g = cp.asarray(w_gh)
            norm_w = (w_g / float(np.sqrt(np.pi)))[None, :, None]
            mu_g = cp.asarray(mu_all)
            sigmas_g = cp.asarray(sigmas_all)
            sqrt2 = float(np.sqrt(2.0))
            half_log_2pi = float(0.5 * np.log(2.0 * np.pi))
            log_ntrees = float(np.log(n_trees))
            ln2 = float(np.log(2.0))
            
            start = 0
            while start < n_samples:
                end = min(start + batch_size, n_samples)
                B = end - start
                try:
                    mu_b = mu_g[:, start:end]
                    sig_b = sigmas_g[:, start:end]
                    
                    y = mu_b[:, None, :] + sqrt2 * sig_b[:, None, :] * z_g[None, :, None]
                    y_exp = y[:, :, None, :]
                    mu_exp = mu_b[None, None, :, :]
                    sig_exp = sig_b[None, None, :, :]
                    
                    u = (y_exp - mu_exp) / sig_exp
                    log_comp = -0.5 * u**2 - cp.log(sig_exp) - half_log_2pi
                    log_py = cp_logsumexp(log_comp, axis=2) - log_ntrees
                    log2_py = log_py / ln2
                    
                    h_per_tree = cp.sum(norm_w * (-log2_py), axis=1)
                    batch_entropy = cp.mean(h_per_tree, axis=0)
                    total_entropy[start:end] = cp.asnumpy(batch_entropy)
                    start += B
                except (MemoryError, Exception) as e:
                    if is_gpu and cp is not None:
                        cp.get_default_memory_pool().free_all_blocks()
                    if batch_size <= 1:
                        raise e
                    batch_size = max(1, batch_size // 2)
                    if debug_timing:
                        print(f"OOM in GPU Shaker GH. Halving batch size to {batch_size}")
        else:
            norm_w = (w_gh / np.sqrt(np.pi))[None, :, None]
            start = 0
            while start < n_samples:
                end = min(start + batch_size, n_samples)
                B = end - start
                try:
                    mu_b = mu_all[:, start:end]
                    sig_b = sigmas_all[:, start:end]
                    
                    y = mu_b[:, None, :] + np.sqrt(2.0) * sig_b[:, None, :] * z_gh[None, :, None]
                    y_exp = y[:, :, None, :]
                    mu_exp = mu_b[None, None, :, :]
                    sig_exp = sig_b[None, None, :, :]
                    
                    u = (y_exp - mu_exp) / sig_exp
                    log_comp = -0.5 * u**2 - np.log(sig_exp) - 0.5 * np.log(2.0 * np.pi)
                    log_py = logsumexp(log_comp, axis=2) - np.log(n_trees)
                    log2_py = log_py / np.log(2.0)
                    
                    h_per_tree = np.sum(norm_w * (-log2_py), axis=1)
                    batch_entropy = np.mean(h_per_tree, axis=0)
                    total_entropy[start:end] = batch_entropy
                    start += B
                except (MemoryError, Exception) as e:
                    if batch_size <= 1:
                        raise e
                    batch_size = max(1, batch_size // 2)
                    if debug_timing:
                        print(f"Memory pressure in CPU Shaker GH. Halving batch size to {batch_size}")
                        
        if debug_timing:
            print(f"   [GMM Shaker Profile] Total total_entropy calculation took: {time.time() - t0:.6f}s")
            
        return total_entropy
            
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
