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

from scipy.special import erf as np_erf, log_ndtr as np_log_ndtr
try:
    import cupy as cp
    from cupyx.scipy.special import erf as cp_erf, log_ndtr as cp_log_ndtr
except ImportError:
    cp = None
    cp_erf = None
    cp_log_ndtr = None

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

    def compute_uq(self, X_test, backend="auto", n_grid=None, batch_size=2000, integration_method="gauss_legendre", sup_solver="bisection"):
        """
        Computes the epistemic and aleatoric uncertainties using the continuous
        relative likelihood framework. Fully vectorized and GPU-accelerated when available.
        Batched to prevent out-of-memory errors on large test sets.
        
        Args:
            X_test: np.ndarray of shape (n_samples, n_features)
            backend: 'auto', 'cpu', or 'gpu'
            n_grid: Number of grid points for numerical integration of z
            batch_size: Maximum number of test samples to process in a single batch
            integration_method: 'gauss_legendre' or 'trapezoid'
            sup_solver: 'bisection' or 'newton'
            
        Returns:
            epistemic_var: np.ndarray of shape (n_samples,) in variance-like units
            aleatoric_var: np.ndarray of shape (n_samples,) in variance-like units
        """
        X_test = np.atleast_2d(X_test)
        n_samples = X_test.shape[0]
        
        # Determine if GPU will be used
        is_gpu = backend == "gpu" or (backend == "auto" and cp is not None and cp.cuda.runtime.getDeviceCount() > 0)
        
        if n_grid is None:
            # Scale grid size dynamically based on input dimensionality
            D = X_test.shape[1]
            if integration_method == "trapezoid":
                n_grid = 128 if D >= 3 else 64
            else: # gauss_legendre
                n_grid = 64 if D >= 3 else 32
                
        n_iter = 20 if is_gpu else 15
        
        if n_samples <= batch_size:
            return self._compute_uq_batch(X_test, backend=backend, n_grid=n_grid, n_iter=n_iter, integration_method=integration_method, sup_solver=sup_solver)
            
        # Batched execution to prevent OOM
        epistemic_vars = []
        aleatoric_vars = []
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_test[i : i + batch_size]
            epistemic_batch, aleatoric_batch = self._compute_uq_batch(
                X_batch, backend=backend, n_grid=n_grid, n_iter=n_iter, integration_method=integration_method, sup_solver=sup_solver
            )
            epistemic_vars.append(epistemic_batch)
            aleatoric_vars.append(aleatoric_batch)
            
        return np.concatenate(epistemic_vars), np.concatenate(aleatoric_vars)

    def _compute_uq_batch(self, X_test, backend="auto", n_grid=100, n_iter=15, integration_method="gauss_legendre", sup_solver="bisection"):
        """Internal method to compute UQ for a single batch using Method B (ensemble-level integration)."""
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
        
        # Calculate t_min and t_max for each sample across all trees
        # We use 6 * sigma to cover the tails fully
        t_min = xp.min(means_g - 6.0 * sigmas_g, axis=0)
        t_max = xp.max(means_g + 6.0 * sigmas_g, axis=0)
        
        # 3. Setup the integration grid over t for each sample
        # t_grid shape: (n_samples, n_grid)
        if integration_method == "trapezoid":
            # Linear grid between t_min and t_max
            grid_steps = xp.linspace(0.0, 1.0, n_grid)
            t_grid = t_min[:, xp.newaxis] + grid_steps[xp.newaxis, :] * (t_max - t_min)[:, xp.newaxis]
            dt = (t_max - t_min) / (n_grid - 1)
        elif integration_method == "gauss_legendre":
            # Gauss-Legendre quadrature roots and weights on [-1, 1]
            roots_std, weights_std = np.polynomial.legendre.leggauss(n_grid)
            if xp is not np:
                roots_std = cp.asarray(roots_std)
                weights_std = cp.asarray(weights_std)
            # Map to [t_min, t_max] for each sample
            t_grid = 0.5 * (t_max - t_min)[:, xp.newaxis] * roots_std[xp.newaxis, :] + 0.5 * (t_max + t_min)[:, xp.newaxis]
            weights_g = 0.5 * (t_max - t_min)[:, xp.newaxis] * weights_std[xp.newaxis, :]
        else:
            raise ValueError(f"Unknown integration_method: {integration_method}")
            
        # Reshape for multi-dimensional broadcasting: (n_trees, n_samples, n_grid)
        means_b = means_g[:, :, xp.newaxis]
        sigmas_b = sigmas_g[:, :, xp.newaxis]
        k_b = counts_g[:, :, xp.newaxis]
        t_b = t_grid[xp.newaxis, :, :]
        
        # Normalized grid coordinates z = (t - mean) / sigma
        z_b = (t_b - means_b) / sigmas_b
        
        # Helper function for CDF of standard normal distribution
        def xp_cdf(x):
            if xp is np:
                return 0.5 * (1.0 + np_erf(x / np.sqrt(2.0)))
            else:
                return 0.5 * (1.0 + cp_erf(x / cp.sqrt(2.0)))
            
        # Helper function for log_ndtr of standard normal distribution
        def xp_log_ndtr(x):
            if xp is np:
                return np_log_ndtr(x)
            else:
                return cp_log_ndtr(x)
            
        if sup_solver == "newton":
            # Newton-Raphson iterations: 8 is more than enough for machine precision
            n_iter_newton = 8
            log_sqrt_2pi = 0.5 * xp.log(2.0 * np.pi)
            
            # --- Vectorized Newton-Raphson for pi_le ---
            # Initialize u using a negative value scaled by z
            u_le = - (xp.abs(z_b) + 1.0) / (xp.sqrt(k_b) + 1.0)
            
            for _ in range(n_iter_newton):
                w = z_b - u_le
                log_phi = xp_log_ndtr(w)
                inv_mills = xp.exp(-0.5 * w**2 - log_sqrt_2pi - log_phi)
                
                h_val = -0.5 * k_b * u_le**2 - log_phi
                h_prime = -k_b * u_le + inv_mills
                
                u_le = u_le - h_val / h_prime
                u_le = xp.minimum(u_le, -1e-15)
                
            pi_le = xp.exp(-k_b * u_le**2 / 2.0)
            
            # --- Vectorized Newton-Raphson for pi_ge ---
            # Initialize u using a negative value scaled by z
            u_ge = - (xp.abs(z_b) + 1.0) / (xp.sqrt(k_b) + 1.0)
            
            for _ in range(n_iter_newton):
                w = u_ge - z_b
                log_phi = xp_log_ndtr(w)
                inv_mills = xp.exp(-0.5 * w**2 - log_sqrt_2pi - log_phi)
                
                h_val = -0.5 * k_b * u_ge**2 - log_phi
                h_prime = -k_b * u_ge - inv_mills
                
                u_ge = u_ge - h_val / h_prime
                u_ge = xp.minimum(u_ge, -1e-15)
                
            pi_ge = xp.exp(-k_b * u_ge**2 / 2.0)
            
        else: # bisection
            # 4. Vectorized Bisection to find the supremum root for pi_le
            a_le = xp.zeros((n_trees, n_samples, n_grid)) - 10.0 / xp.sqrt(k_b)
            b_le = xp.zeros((n_trees, n_samples, n_grid)) + 10.0 / xp.sqrt(k_b)
            
            for _ in range(n_iter):
                mu_u = 0.5 * (a_le + b_le)
                pi_H = xp.exp(-k_b * (mu_u**2) / 2.0)
                phi_val = xp_cdf(z_b - mu_u)
                mask = pi_H < phi_val
                a_le = xp.where(mask, mu_u, a_le)
                b_le = xp.where(mask, b_le, mu_u)
                
            pi_le = xp.exp(-k_b * (a_le**2) / 2.0)
            
            # 5. Vectorized Bisection to find the supremum root for pi_ge
            a_ge = xp.zeros((n_trees, n_samples, n_grid)) - 10.0 / xp.sqrt(k_b)
            b_ge = xp.zeros((n_trees, n_samples, n_grid)) + 10.0 / xp.sqrt(k_b)
            
            for _ in range(n_iter):
                mu_u = 0.5 * (a_ge + b_ge)
                pi_H = xp.exp(-k_b * (mu_u**2) / 2.0)
                phi_val = xp_cdf(mu_u - z_b)
                mask = pi_H < phi_val
                a_ge = xp.where(mask, a_ge, mu_u)
                b_ge = xp.where(mask, mu_u, b_ge)
                
            pi_ge = xp.exp(-k_b * (a_ge**2) / 2.0)
        
        # 6. Ensemble-level averaging BEFORE integration (Method B)
        mean_pi_le = xp.mean(pi_le, axis=0)
        mean_pi_ge = xp.mean(pi_ge, axis=0)
        
        # 7. Epistemic (intersection) and Aleatoric (residual) integrands
        min_vals = xp.minimum(mean_pi_le, mean_pi_ge)
        max_vals = xp.maximum(mean_pi_le, mean_pi_ge)
        
        # 8. Perform numerical integration over grid (axis 1)
        if integration_method == "trapezoid":
            I_ep = xp.sum(0.5 * (min_vals[:, :-1] + min_vals[:, 1:]) * dt[:, xp.newaxis], axis=1)
            I_al = xp.sum(0.5 * ((1.0 - max_vals[:, :-1]) + (1.0 - max_vals[:, 1:])) * dt[:, xp.newaxis], axis=1)
        elif integration_method == "gauss_legendre":
            I_ep = xp.sum(min_vals * weights_g, axis=1)
            I_al = xp.sum((1.0 - max_vals) * weights_g, axis=1)
            
        # Move variables back to CPU if computed on GPU
        if xp is not np:
            I_ep = cp.asnumpy(I_ep)
            I_al = cp.asnumpy(I_al)
            
        # 9. Transform to variance units (square the results)
        epistemic_var = I_ep ** 2
        aleatoric_var = I_al ** 2
        
        return epistemic_var, aleatoric_var
