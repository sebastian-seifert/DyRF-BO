import os
import sys
import numpy as np
from scipy.special import logsumexp

try:
    import cupy as cp
    from cupyx.scipy.special import logsumexp as cp_logsumexp
except ImportError:
    cp = None
    cp_logsumexp = None

class EpistemicQuantifier:
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

    # ==========================================
    # BASE / SHARED METHODS
    # ==========================================
    def _base_calc_per_tree_variance(self, X_test, min_var=1e-6):
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

    def base_get_aleatoric_variance(self, X_test):
        """
        Returns the mean of the per-tree variances (E[sigma^2]).
        This is the baseline aleatoric uncertainty in terms of variance.
        """
        return np.mean(self._base_calc_per_tree_variance(X_test), axis=0)

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
        tree_preds = np.stack([tree.predict(X_test) for tree in self.model.estimators_])
        
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
        tree_preds = np.stack([tree.predict(X_test) for tree in self.model.estimators_])
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
    def shaker_get_epistemic_entropy(self, X_test, num_samples=1000, batch_size=1000, random_state=None, backend="auto"):
        """
        Approach 2: Shaker 2020 (Epistemic Component)
        Calculated as: Total Uncertainty (GMM Entropy) - Aleatoric Uncertainty.
        
        Total Uncertainty is the entropy of the Gaussian Mixture Model formed by the trees.
        Aleatoric is the mean entropy of the individual tree distributions.
        """
        total_unc = self._shaker_calc_total_entropy(
            X_test,
            num_samples=num_samples,
            batch_size=batch_size,
            random_state=random_state,
            backend=backend,
        )
        aleatoric_unc = self._shaker_calc_aleatoric_entropy(X_test)
        
        # Epistemic = Total - Aleatoric
        return np.maximum(total_unc - aleatoric_unc, 0.0)

    def shaker_get_epistemic_variance(self, X_test, num_samples=1000, batch_size=1000, random_state=None, backend="auto"):
        """
        Returns a Shaker-inspired epistemic proxy in variance units.

        Shaker's native decomposition is entropy-based, while Standard and Chen
        return variance-like quantities. We therefore keep the native entropy
        decomposition as mutual information in bits and map that information
        to a local variance increase relative to the aleatoric variance:

            MI = 0.5 * log2(total_var / aleatoric_var)
            epistemic_var = aleatoric_var * (2 ** (2 * MI) - 1)
        """
        mi_bits = self.shaker_get_epistemic_entropy(
            X_test,
            num_samples=num_samples,
            batch_size=batch_size,
            random_state=random_state,
            backend=backend,
        )
        aleatoric_var = self.base_get_aleatoric_variance(X_test)

        return aleatoric_var * np.maximum(2.0 ** (2.0 * mi_bits) - 1.0, 0.0)

    def shaker_get_total_variance(self, X_test, num_samples=1000, batch_size=1000, random_state=None, backend="auto"):
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
    def _shaker_calc_aleatoric_entropy(self, X_test):
        """
        Calculates the closed-form aleatoric uncertainty (mean differential entropy).
        Formula from Slide 4: (1/M) * sum( 0.5 * log2(2 * pi * e * sigma_hat^2) )
        """
        vars2 = self._base_calc_per_tree_variance(X_test) 
        individual_entropies = 0.5 * np.log2(2 * np.pi * np.e * vars2)
        return np.mean(individual_entropies, axis=0)

    def _shaker_convert_entropy_to_var(self, entropy_bits):
        """Converts differential entropy in bits to the variance of a Gaussian."""
        return (2.0 ** (2.0 * entropy_bits)) / (2.0 * np.pi * np.e)

    def _shaker_calc_total_entropy(self, X_test, num_samples=1000, batch_size=1000, random_state=None, backend="auto"):
        """
        Calculates the Total Uncertainty (Entropy of the GMM) via Monte Carlo.
        Formula from Slide 5: E[-log2(p(y|x))], with y sampled from the tree GMM.
        """
        X_test = np.atleast_2d(X_test)
        n_samples = X_test.shape[0]
        backend = self._mc_resolve_backend(backend)
        print(f"Monte Carlo backend: {backend}")
        rng = self._mc_make_gpu_rng(random_state) if backend == "gpu" else self._mc_make_cpu_rng(random_state)
        
        # 1. Get components for each tree
        mu_all = np.stack([t.predict(X_test) for t in self.model.estimators_]) # (n_trees, n_samples)
        vars_all = self._base_calc_per_tree_variance(X_test) # (n_trees, n_samples)
        sigmas_all = np.sqrt(vars_all)
        
        total_entropy = np.zeros(n_samples)

        # 2. Estimate each test point's GMM entropy in log-space.
        for i in range(n_samples):
            mu_i = mu_all[:, i]
            sigma_i = sigmas_all[:, i]
            total_entropy[i] = self._shaker_gmm_entropy_mc(
                mu_i,
                sigma_i,
                rng,
                num_samples=num_samples,
                batch_size=batch_size,
                backend=backend,
            )
        
        return total_entropy

    def _shaker_gmm_entropy_mc(self, mu, sigma, rng, num_samples=1000, batch_size=1000, backend="cpu"):
        if backend == "gpu":
            try:
                return self._shaker_gmm_entropy_mc_gpu(mu, sigma, rng, num_samples, batch_size)
            except Exception as exc:
                print(f"GPU Monte Carlo failed. Falling back to CPU for this point. ({exc})")
                cpu_rng = self._mc_make_cpu_rng(None)
                return self._shaker_gmm_entropy_mc_cpu(mu, sigma, cpu_rng, num_samples, batch_size)
        return self._shaker_gmm_entropy_mc_cpu(mu, sigma, rng, num_samples, batch_size)

    def _shaker_gmm_entropy_mc_cpu(self, mu, sigma, rng, num_samples=1000, batch_size=1000):
        """Approximates GMM entropy via Expected Value Monte Carlo (CPU)."""
        K = len(mu)
        entropy_sum = 0.0
        samples_done = 0

        while samples_done < num_samples:
            current_batch = min(batch_size, num_samples - samples_done)

            components = rng.integers(0, K, size=current_batch)
            y_samples = rng.normal(mu[components], sigma[components])

            y_expanded = y_samples[:, np.newaxis]
            mu_expanded = mu[np.newaxis, :]
            sigma_expanded = sigma[np.newaxis, :]

            log_densities = -0.5 * ((y_expanded - mu_expanded) / sigma_expanded)**2
            log_densities -= np.log(sigma_expanded * np.sqrt(2 * np.pi))

            log_p_y = logsumexp(log_densities, axis=1) - np.log(K)
            entropy_sum += -np.sum(log_p_y) / np.log(2)
            samples_done += current_batch

        return entropy_sum / num_samples

    def _shaker_gmm_entropy_mc_gpu(self, mu, sigma, rng, num_samples=1000, batch_size=1000):
        """GPU version of the GMM entropy Monte Carlo estimator."""
        mu_gpu = cp.asarray(mu)
        sigma_gpu = cp.asarray(sigma)
        K = len(mu)
        entropy_sum = cp.asarray(0.0)
        samples_done = 0

        while samples_done < num_samples:
            current_batch = min(batch_size, num_samples - samples_done)

            if hasattr(rng, "integers"):
                components = rng.integers(0, K, size=current_batch)
            else:
                components = rng.randint(0, K, size=current_batch)
                
            eps = self._mc_gpu_standard_normal(rng, current_batch)
            y_samples = mu_gpu[components] + sigma_gpu[components] * eps

            y_expanded = y_samples[:, cp.newaxis]
            mu_expanded = mu_gpu[cp.newaxis, :]
            sigma_expanded = sigma_gpu[cp.newaxis, :]

            log_densities = -0.5 * ((y_expanded - mu_expanded) / sigma_expanded)**2
            log_densities -= cp.log(sigma_expanded * cp.sqrt(2 * cp.pi))

            log_p_y = cp_logsumexp(log_densities, axis=1) - cp.log(K)
            entropy_sum += -cp.sum(log_p_y) / cp.log(2)
            samples_done += current_batch

        return float(cp.asnumpy(entropy_sum / num_samples))

    # ==========================================
    # MONTE CARLO (MC) UTILITIES
    # ==========================================
    def _mc_is_cupy_available(self):
        if cp is None:
            return False
        try:
            if cp.cuda.runtime.getDeviceCount() == 0:
                return False
            smoke_test = cp.asarray([1.0])
            smoke_test = smoke_test + 1.0
            cp.cuda.Stream.null.synchronize()
            return bool(cp.asnumpy(smoke_test)[0] == 2.0)
        except Exception as exc:
            print(f"CuPy/CUDA smoke test failed. Falling back to CPU Monte Carlo. ({exc})")
            return False

    def _mc_resolve_backend(self, backend):
        if backend not in {"auto", "cpu", "gpu"}:
            raise ValueError("backend must be one of: 'auto', 'cpu', 'gpu'")
        if backend == "auto":
            return "gpu" if self._mc_is_cupy_available() else "cpu"
        if backend == "gpu" and not self._mc_is_cupy_available():
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
