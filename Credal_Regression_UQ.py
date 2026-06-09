import os
import sys
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

from scipy.special import erf as np_erf
try:
    import cupy as cp
    from cupyx.scipy.special import erf as cp_erf
except ImportError:
    cp = None
    cp_erf = None

class CredalRegressionUQ:
    def __init__(self, model, X_train, y_train):
        """
        Implementation of the Continuous Relative Likelihood (Credal UQ) framework
        for Random Forest Regression.
        """
        self.model = model
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

    def _calc_leaf_stats(self, X_test, min_var=1e-6):
        """
        Calculates leaf statistics (means, variances, and counts) for each tree and test sample.
        This runs on the CPU for robust data extraction.
        
        Returns:
            means: np.ndarray of shape (n_trees, n_samples)
            variances: np.ndarray of shape (n_trees, n_samples)
            counts: np.ndarray of shape (n_trees, n_samples)
        """
        X_test = np.atleast_2d(X_test)
        n_trees = len(self.model.estimators_)
        n_samples = X_test.shape[0]
        
        # Get leaf assignments for all test points
        all_test_leaf_ids = self.model.apply(X_test)
        
        means = np.zeros((n_trees, n_samples))
        variances = np.zeros((n_trees, n_samples))
        counts = np.zeros((n_trees, n_samples))
        
        for i, estimator in enumerate(self.model.estimators_):
            train_leaf_ids = estimator.apply(self.X_train)
            test_leaf_ids = all_test_leaf_ids[:, i]
            
            unique_test_leaves = np.unique(test_leaf_ids)
            leaf_to_stats = {}
            
            for leaf_id in unique_test_leaves:
                leaf_y = self.y_train[train_leaf_ids == leaf_id]
                k = leaf_y.shape[0]
                
                mean_val = np.mean(leaf_y)
                if k > 1:
                    s2 = np.var(leaf_y, ddof=1)
                else:
                    s2 = 0.0
                
                # Store (mean, variance, sample count)
                leaf_to_stats[leaf_id] = (float(mean_val), float(s2) + min_var, int(k))
                
            means[i, :] = [leaf_to_stats[lid][0] for lid in test_leaf_ids]
            variances[i, :] = [leaf_to_stats[lid][1] for lid in test_leaf_ids]
            counts[i, :] = [leaf_to_stats[lid][2] for lid in test_leaf_ids]
            
        return means, variances, counts

    def compute_uq(self, X_test, backend="auto", n_grid=None, batch_size=2000):
        """
        Computes the epistemic and aleatoric uncertainties using the continuous
        relative likelihood framework. Fully vectorized and GPU-accelerated when available.
        Batched to prevent out-of-memory errors on large test sets.
        
        Args:
            X_test: np.ndarray of shape (n_samples, n_features)
            backend: 'auto', 'cpu', or 'gpu'
            n_grid: Number of grid points for numerical integration of z (defaults to 100 on GPU, 32 on CPU)
            batch_size: Maximum number of test samples to process in a single batch
            
        Returns:
            epistemic_var: np.ndarray of shape (n_samples,) in variance-like units
            aleatoric_var: np.ndarray of shape (n_samples,) in variance-like units
        """
        X_test = np.atleast_2d(X_test)
        n_samples = X_test.shape[0]
        
        # Determine if GPU will be used
        is_gpu = backend == "gpu" or (backend == "auto" and cp is not None and cp.cuda.runtime.getDeviceCount() > 0)
        
        if n_grid is None:
            n_grid = 100 if is_gpu else 32
        n_iter = 15 if is_gpu else 10
        
        if n_samples <= batch_size:
            return self._compute_uq_batch(X_test, backend=backend, n_grid=n_grid, n_iter=n_iter)
            
        # Batched execution to prevent OOM
        epistemic_vars = []
        aleatoric_vars = []
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_test[i : i + batch_size]
            epistemic_batch, aleatoric_batch = self._compute_uq_batch(X_batch, backend=backend, n_grid=n_grid, n_iter=n_iter)
            epistemic_vars.append(epistemic_batch)
            aleatoric_vars.append(aleatoric_batch)
            
        return np.concatenate(epistemic_vars), np.concatenate(aleatoric_vars)

    def _compute_uq_batch(self, X_test, backend="auto", n_grid=100, n_iter=15):
        """Internal method to compute UQ for a single batch."""
        # 1. Retrieve CPU leaf statistics
        means, variances, counts = self._calc_leaf_stats(X_test)
        sigmas = np.sqrt(variances)
        
        # 2. Determine and configure backend
        xp = np
        if backend == "gpu" or (backend == "auto" and cp is not None and cp.cuda.runtime.getDeviceCount() > 0):
            xp = cp
            means_g = cp.asarray(means)
            sigmas_g = cp.asarray(sigmas)
            counts_g = cp.asarray(counts)
        else:
            means_g = means
            sigmas_g = sigmas
            counts_g = counts
            
        n_trees, n_samples = means_g.shape
        
        # 3. Setup Gauss-Legendre quadrature grid z in [-5, 5] (kink-split at 0)
        # We split the interval [-5, 5] into [-5, 0] and [0, 5] to avoid the kink at z=0,
        # ensuring C^inf smoothness and exponential convergence on each subinterval.
        n_half = max(2, n_grid // 2)
        roots_std, weights_std = np.polynomial.legendre.leggauss(n_half)
        
        # Map to left half [-5, 0]: y = 2.5 * x - 2.5
        roots_left = 2.5 * roots_std - 2.5
        weights_left = 2.5 * weights_std
        
        # Map to right half [0, 5]: y = 2.5 * x + 2.5
        roots_right = 2.5 * roots_std + 2.5
        weights_right = 2.5 * weights_std
        
        z_grid_cpu = np.concatenate([roots_left, roots_right])
        weights_cpu = np.concatenate([weights_left, weights_right])
        
        if xp is not np:
            z_grid = cp.asarray(z_grid_cpu)
            weights = cp.asarray(weights_cpu)
        else:
            z_grid = z_grid_cpu
            weights = weights_cpu
            
        # Reshape for multi-dimensional broadcasting: (n_trees, n_samples, n_grid)
        k_b = counts_g[:, :, xp.newaxis]
        z_b = z_grid[xp.newaxis, xp.newaxis, :]
        
        # Helper function for CDF of standard normal distribution
        def xp_cdf(x):
            if xp is np:
                return 0.5 * (1.0 + np_erf(x / np.sqrt(2.0)))
            else:
                return 0.5 * (1.0 + cp_erf(x / cp.sqrt(2.0)))
            
        # 4. Vectorized Bisection to find the supremum root for pi_le
        # Search interval is [y_mean - 4*sigma/sqrt(k), y_mean + 4*sigma/sqrt(k)]
        # In normalized space u, this is [-4/sqrt(k), 4/sqrt(k)]
        a_le = xp.zeros((n_trees, n_samples, 2 * n_half)) - 4.0 / xp.sqrt(k_b)
        b_le = xp.zeros((n_trees, n_samples, 2 * n_half)) + 4.0 / xp.sqrt(k_b)
        
        for _ in range(n_iter):
            mu_u = 0.5 * (a_le + b_le)
            pi_H = xp.exp(-k_b * (mu_u**2) / 2.0)
            phi_val = xp_cdf(z_b - mu_u)
            mask = pi_H < phi_val
            a_le = xp.where(mask, mu_u, a_le)
            b_le = xp.where(mask, b_le, mu_u)
            
        pi_le = xp.exp(-k_b * (a_le**2) / 2.0)
        
        # 5. Vectorized Bisection to find the supremum root for pi_ge
        a_ge = xp.zeros((n_trees, n_samples, 2 * n_half)) - 4.0 / xp.sqrt(k_b)
        b_ge = xp.zeros((n_trees, n_samples, 2 * n_half)) + 4.0 / xp.sqrt(k_b)
        
        for _ in range(n_iter):
            mu_u = 0.5 * (a_ge + b_ge)
            pi_H = xp.exp(-k_b * (mu_u**2) / 2.0)
            phi_val = xp_cdf(mu_u - z_b)
            mask = pi_H < phi_val
            a_ge = xp.where(mask, a_ge, mu_u)
            b_ge = xp.where(mask, mu_u, b_ge)
            
        pi_ge = xp.exp(-k_b * (a_ge**2) / 2.0)
        
        # 6. Epistemic (intersection) and Aleatoric (residual) integrands
        min_vals = xp.minimum(pi_le, pi_ge)
        max_vals = xp.maximum(pi_le, pi_ge)
        
        # 7. Gauss-Legendre quadrature integration (weighted sum over axis 2)
        weights_b = weights[xp.newaxis, xp.newaxis, :]
        I_ep = xp.sum(min_vals * weights_b, axis=2)
        I_al = xp.sum((1.0 - max_vals) * weights_b, axis=2)
        
        # Multiply by local sigma to scale integrals back to data units
        EU_per_tree = sigmas_g * I_ep
        AU_per_tree = sigmas_g * I_al
        
        # Average results over all trees
        EU_mean = xp.mean(EU_per_tree, axis=0)
        AU_mean = xp.mean(AU_per_tree, axis=0)
        
        # Move variables back to CPU if computed on GPU
        if xp is not np:
            EU_mean = cp.asnumpy(EU_mean)
            AU_mean = cp.asnumpy(AU_mean)
            
        # 8. Transform from standard deviation units to variance units (square the results)
        epistemic_var = EU_mean ** 2
        aleatoric_var = AU_mean ** 2
        
        return epistemic_var, aleatoric_var
