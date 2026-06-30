import os
import sys
# Reconfigure stdout and stderr to UTF-8 to prevent encoding errors on cluster environments with non-UTF-8 locales
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
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, friedmanchisquare, wilcoxon, gaussian_kde
from scipy.special import logsumexp, xlogy, entr
from scipy.spatial.distance import jensenshannon
from Credal_Regression_UQ import CredalRegressionUQ
from Proximity_Regression_UQ import ProximityRegressionUQ

# Helps on clusters where NVRTC does not directly support the GPU's native arch.
os.environ.setdefault("CUPY_COMPILE_WITH_PTX", "1")

try:
    import cupy as cp
    from cupyx.scipy.special import logsumexp as cp_logsumexp
except ImportError:
    cp = None
    cp_logsumexp = None

"""
Bachelor Thesis: Epistemic Uncertainty Quantification
Primary Focus: Quantifying uncertainty due to lack of data/exploration (Epistemic).
Evaluation Metric: Correlation with Error in Out-of-Distribution (OOD) regions.
"""

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
        Pre-calculates variances for unique leaves to ensure low complexity.
        
        Returns: np.array of shape (n_trees, n_samples_test)
        """
        X_test = np.atleast_2d(X_test)
        n_trees = len(self.model.estimators_)
        n_samples = X_test.shape[0]
        
        # Get all leaf assignments for test data: shape (n_samples, n_trees)
        all_test_leaf_ids = self.model.apply(X_test)
        
        variances = np.zeros((n_trees, n_samples))
        
        for i, estimator in enumerate(self.model.estimators_):
            # 1. Get training leaf IDs and test leaf IDs for this specific tree
            train_leaf_ids = estimator.apply(self.X_train)
            test_leaf_ids = all_test_leaf_ids[:, i]
            
            # 2. Map unique test leaves to their training variance
            unique_test_leaves = np.unique(test_leaf_ids)
            leaf_to_var = {}
            
            for leaf_id in unique_test_leaves:
                # Optimized mask: only look at training data in this leaf
                leaf_y = self.y_train[train_leaf_ids == leaf_id]
                
                if leaf_y.shape[0] > 1:
                    s2 = np.var(leaf_y, ddof=1)
                else:
                    s2 = 0.0
                
                leaf_to_var[leaf_id] = float(s2) + min_var
                
            # 3. Fast mapping of pre-calculated variances back to the test samples
            variances[i, :] = [leaf_to_var[lid] for lid in test_leaf_ids]
            
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
        return np.maximum(total_unc - aleatoric_unc, 0.0) # Ensure non-negative epistemic uncertainty

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


def plot_uncertainty(name, X_test, y_test, y_pred, var_pred, X_train, y_train):
    """
    Visualizes the prediction mean and the uncertainty bands.
    Shaded area represents 2 standard deviations (approx 95% confidence).
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train, y_train, color='black', s=10, alpha=0.3, label='Training Data')
    plt.plot(X_test, y_test, color='green', alpha=0.5, label='True Function')
    plt.plot(X_test, y_pred, color='blue', label='RF Prediction')
    
    # Calculate 2-sigma bands
    std = np.sqrt(var_pred)
    plt.fill_between(X_test.ravel(), y_pred - 2*std, y_pred + 2*std, color='blue', alpha=0.2, label='2-sigma (Epistemic)')
    
    # Highlight the Gap
    plt.axvspan(4, 6, color='red', alpha=0.1, label='Exploration Gap')
    
    plt.title(f"Epistemic Uncertainty: {name}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    
    # Save the plot
    os.makedirs("figures", exist_ok=True)
    filename = f"uncertainty_{name.lower().replace(' ', '_')}.png"
    filename = os.path.join("figures", filename)
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()

# ==========================================
# TEST FUNCTION GENERATORS (15 functions)
# ==========================================

def get_1d_functions():
    """Returns 5 diverse 1D functions with training gaps."""
    functions = {
        "sin": {
            "func": lambda x: np.sin(x),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "cos_trend": {
            "func": lambda x: np.cos(x) + x / 10,
            "gap": (4, 6),
            "range": (0, 10),
        },
        "poly": {
            "func": lambda x: x**2 / 50,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "damped_osc": {
            "func": lambda x: np.exp(-x / 5) * np.sin(2 * x),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "log_mod": {
            "func": lambda x: np.log(x + 1) * np.sin(x),
            "gap": (3.5, 6.5),
            "range": (0.1, 10),
        },
    }
    return functions

def get_2d_functions():
    """Returns 5 diverse 2D functions with training gaps."""
    functions = {
        "sin_cos": {
            "func": lambda x1, x2: np.sin(x1) * np.cos(x2),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic": {
            "func": lambda x1, x2: (x1**2 + x2**2) / 100,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "sin_sum_mod": {
            "func": lambda x1, x2: np.sin(x1 + x2) + 0.1 * x1 * x2,
            "gap": (4, 6),
            "range": (0, 10),
        },
        "gaussian": {
            "func": lambda x1, x2: np.exp(-(x1**2 + x2**2) / 10),
            "gap": (3.5, 6.5),
            "range": (-5, 5),
        },
        "abs_sin": {
            "func": lambda x1, x2: np.abs(x1 - x2) + np.sin(x1 * x2),
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_3d_functions():
    """Returns 5 diverse 3D functions with training gaps."""
    functions = {
        "sin_cos_sin": {
            "func": lambda x1, x2, x3: np.sin(x1) * np.cos(x2) * np.sin(x3),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_3d": {
            "func": lambda x1, x2, x3: (x1**2 + x2**2 + x3**2) / 150,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "sin_sum_3d": {
            "func": lambda x1, x2, x3: np.sin(x1 + x2 + x3) + 0.1 * x1 * x2 * x3,
            "gap": (4, 6),
            "range": (0, 10),
        },
        "gaussian_3d": {
            "func": lambda x1, x2, x3: np.exp(-(x1**2 + x2**2 + x3**2) / 15),
            "gap": (3.5, 6.5),
            "range": (-5, 5),
        },
        "sin_exp_cos": {
            "func": lambda x1, x2, x3: np.sin(x1) * np.exp(-x2 / 5) * np.cos(x3),
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_4d_functions():
    """Returns 3 diverse 4D functions with training gaps."""
    functions = {
        "sin_cos_4d": {
            "func": lambda x1, x2, x3, x4: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_4d": {
            "func": lambda x1, x2, x3, x4: (x1**2 + x2**2 + x3**2 + x4**2) / 200,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "sin_sum_4d": {
            "func": lambda x1, x2, x3, x4: np.sin(x1 + x2 + x3 + x4) + 0.05 * x1 * x2 * x3 * x4,
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_5d_functions():
    """Returns 3 diverse 5D functions with training gaps."""
    functions = {
        "sin_cos_5d": {
            "func": lambda x1, x2, x3, x4, x5: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_5d": {
            "func": lambda x1, x2, x3, x4, x5: (x1**2 + x2**2 + x3**2 + x4**2 + x5**2) / 250,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "gaussian_5d": {
            "func": lambda x1, x2, x3, x4, x5: np.exp(-(x1**2 + x2**2 + x3**2 + x4**2 + x5**2) / 20),
            "gap": (3.5, 6.5),
            "range": (-5, 5),
        },
    }
    return functions

def get_6d_functions():
    """Returns 3 diverse 6D functions with training gaps."""
    functions = {
        "sin_cos_6d": {
            "func": lambda x1, x2, x3, x4, x5, x6: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_6d": {
            "func": lambda x1, x2, x3, x4, x5, x6: (x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2) / 300,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "friedman_6d": {
            "func": lambda x1, x2, x3, x4, x5, x6: 10 * np.sin(np.pi * x1 * x2 / 100) + 20 * (x3 / 10 - 0.5)**2 + 10 * x4 / 10 + 5 * x5 / 10 + x6 / 10,
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_7d_functions():
    """Returns 3 diverse 7D functions with training gaps."""
    functions = {
        "sin_cos_7d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6) * np.sin(x7),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_7d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7: (x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 + x7**2) / 350,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "interaction_7d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7: (x1*x2 + x2*x3 + x3*x4 + x4*x5 + x5*x6 + x6*x7) / 50,
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_8d_functions():
    """Returns 3 diverse 8D functions with training gaps."""
    functions = {
        "sin_cos_8d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6) * np.sin(x7) * np.cos(x8),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_8d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8: (x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 + x7**2 + x8**2) / 400,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "exp_sum_8d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8: np.exp(-( (x1-5)**2 + (x2-5)**2 + (x3-5)**2 + (x4-5)**2 + (x5-5)**2 + (x6-5)**2 + (x7-5)**2 + (x8-5)**2 ) / 80),
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_9d_functions():
    """Returns 3 diverse 9D functions with training gaps."""
    functions = {
        "sin_cos_9d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6) * np.sin(x7) * np.cos(x8) * np.sin(x9),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_9d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9: (x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 + x7**2 + x8**2 + x9**2) / 450,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "multi_modal_9d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9: (np.cos(2*x1) + np.cos(2*x2) + np.cos(2*x3) + np.cos(2*x4) + np.cos(2*x5) + np.cos(2*x6) + np.cos(2*x7) + np.cos(2*x8) + np.cos(2*x9)) + (x1+x2+x3+x4+x5+x6+x7+x8+x9)/10,
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def get_10d_functions():
    """Returns 3 diverse 10D functions with training gaps."""
    functions = {
        "sin_cos_10d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: np.sin(x1) * np.cos(x2) * np.sin(x3) * np.cos(x4) * np.sin(x5) * np.cos(x6) * np.sin(x7) * np.cos(x8) * np.sin(x9) * np.cos(x10),
            "gap": (4, 6),
            "range": (0, 10),
        },
        "quadratic_10d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: (x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 + x7**2 + x8**2 + x9**2 + x10**2) / 500,
            "gap": (3.5, 6.5),
            "range": (0, 10),
        },
        "friedman_10d": {
            "func": lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10: 10 * np.sin(np.pi * x1 * x2 / 100) + 20 * (x3 / 10 - 0.5)**2 + 10 * x4 / 10 + 5 * x5 / 10 + (x6+x7+x8+x9+x10)/10,
            "gap": (4, 6),
            "range": (0, 10),
        },
    }
    return functions

def generate_data(func_dict, func_name, seed, points_per_dim=None, gap_type='empty', sparse_multiplier=12, scaling_law='linear', min_samples_leaf=5):
    """
    Generates training and test data. Uses grid-based meshes for 1D-5D and
    fallback random uniform sampling for >=6D to avoid exponential complexity.
    """
    rng = np.random.default_rng(seed)
    func = func_dict[func_name]
    func_obj = func["func"]
    gap = func["gap"]
    x_range = func["range"]

    # Determine dimensionality dynamically from lambda argument count
    ndim = func_obj.__code__.co_argcount

    # If ndim >= 3, use random uniform sampling instead of dense grids to prevent OOM
    if ndim >= 3:
        if ndim == 3:
            n_samples = 3000
        elif ndim == 4:
            n_samples = 3000
        elif ndim == 5:
            n_samples = 3000
        elif ndim == 6:
            n_samples = 3000
        elif ndim == 7:
            n_samples = 3000
        elif ndim == 8:
            n_samples = 4000
        elif ndim == 9:
            n_samples = 4000
        else: # 10D
            n_samples = 5000
            
        X = rng.uniform(x_range[0], x_range[1], size=(n_samples, ndim))
    else:
        # Dynamic default points_per_dim depending on dimension to control exponential explosion
        if points_per_dim is None:
            if ndim == 1:
                points_per_dim = 1200
            elif ndim == 2:
                points_per_dim = 50
            else:
                points_per_dim = 30

        # Generate the coordinate grids for each axis
        grids = [np.linspace(x_range[0], x_range[1], points_per_dim) for _ in range(ndim)]
        meshes = np.meshgrid(*grids, indexing='ij')
        X = np.stack([m.ravel() for m in meshes], axis=1)
        n_samples = len(X)
    
    # Create the multidimensional OOD gap mask (Hypercube) for training set generation
    gap_mask_train = np.ones(len(X), dtype=bool)
    for d in range(ndim):
        gap_mask_train &= (X[:, d] >= gap[0]) & (X[:, d] <= gap[1])

    if gap_type == 'sparse':
        # Apply the chosen OOD gap sparsity scaling law
        if scaling_law == 'linear':
            n_keep = sparse_multiplier * ndim
        elif scaling_law == 'fractional':
            n_keep = int(sparse_multiplier * (1.3 ** ndim))
        elif scaling_law == 'leaf':
            n_keep = int(sparse_multiplier * min_samples_leaf)
        else:
            n_keep = sparse_multiplier * ndim
        
        # Train mask excludes all points inside the gap
        train_mask = ~gap_mask_train
        
        # Generate n_keep points uniformly sampled across the entire hypercube gap
        gap_rng = np.random.default_rng(seed + 100000)
        X_sparse_gap = gap_rng.uniform(gap[0], gap[1], size=(n_keep, ndim))
        
        X_train = np.concatenate([X[train_mask], X_sparse_gap], axis=0)
    else:
        # gap_type == 'empty'
        train_mask = ~gap_mask_train
        X_train = X[train_mask]
    y_train = func_obj(*[X_train[:, d] for d in range(ndim)]).ravel()
    y_train += rng.normal(0, 0.1, len(y_train))

    # Construct test set by explicitly sampling ID and OOD points (preserving hypercube structure)
    n_id = int(n_samples * 0.7)
    n_ood = n_samples - n_id
    
    X_ood = rng.uniform(gap[0], gap[1], size=(n_ood, ndim))
    
    X_id = []
    while len(X_id) < n_id:
        batch = rng.uniform(x_range[0], x_range[1], size=(n_id - len(X_id), ndim))
        batch_in_gap = np.ones(len(batch), dtype=bool)
        for d in range(ndim):
            batch_in_gap &= (batch[:, d] >= gap[0]) & (batch[:, d] <= gap[1])
        X_id.extend(batch[~batch_in_gap])
    X_id = np.array(X_id)
    
    X_test = np.concatenate([X_id, X_ood], axis=0)
    y_test = func_obj(*[X_test[:, d] for d in range(ndim)]).ravel()
    y_test += rng.normal(0, 0.1, len(y_test))
    
    y_true_binary = np.zeros(len(X_test), dtype=int)
    y_true_binary[n_id:] = 1

    return X_train, y_train, X_test, y_test, y_true_binary

def calculate_jensen_shannon_divergence(uncertainty, y_true_binary, n_bins=50):
    """FIXED: Now squares the distance value to return true JSD in [0, 1]"""
    u_id = uncertainty[y_true_binary == 0]
    u_ood = uncertainty[y_true_binary == 1]

    if len(u_id) < 2 or len(u_ood) < 2: return np.nan

    u_min = min(np.min(u_id), np.min(u_ood))
    u_max = max(np.max(u_id), np.max(u_ood))
    if u_max - u_min < 1e-10:
        return 0.0
    bin_edges = np.linspace(u_min, u_max, n_bins + 1)

    p_id, _ = np.histogram(u_id, bins=bin_edges)
    p_ood, _ = np.histogram(u_ood, bins=bin_edges)

    p_id = p_id / np.sum(p_id) + 1e-10
    p_ood = p_ood / np.sum(p_ood) + 1e-10
    p_id = p_id / np.sum(p_id)
    p_ood = p_ood / np.sum(p_ood)

    js_distance = jensenshannon(p_id, p_ood, base=2.0)
    return float(js_distance ** 2)

def calculate_mutual_information(uncertainty, y_true_binary, n_bins=50):
    """
    Computes Normalized Mutual Information (NMI) using discrete binning.
    Guarantees output is in [0, 1] and eliminates resubstitution bias.
    """
    n_total = len(uncertainty)
    if n_total < 3 or np.min(y_true_binary) == np.max(y_true_binary):
        return np.nan

    # 1. Discretize the continuous uncertainty into bins
    u_min, u_max = np.min(uncertainty), np.max(uncertainty)
    if u_max - u_min < 1e-10:
        return 0.0 # Constant uncertainty carries 0 information
        
    bin_edges = np.linspace(u_min, u_max, n_bins + 1)
    # Map each uncertainty value to its bin index (1 to n_bins)
    u_discrete = np.digitize(uncertainty, bin_edges) - 1
    # Clip boundaries
    u_discrete = np.clip(u_discrete, 0, n_bins - 1)

    # 2. Compute joint and marginal distributions
    joint_counts, _, _ = np.histogram2d(u_discrete, y_true_binary, 
                                        bins=[n_bins, 2], 
                                        range=[[0, n_bins], [0, 2]])
    
    P_joint = joint_counts / n_total
    P_u = np.sum(P_joint, axis=1)
    P_y = np.sum(P_joint, axis=0)

    # 3. Calculate Shannon Entropies in bits
    # H(Y)
    P_y_nonzero = P_y[P_y > 0]
    h_y = -np.sum(P_y_nonzero * np.log2(P_y_nonzero))
    if h_y < 1e-10:
        return np.nan

    # H(U)
    P_u_nonzero = P_u[P_u > 0]
    h_u = -np.sum(P_u_nonzero * np.log2(P_u_nonzero))

    # H(U, Y)
    P_joint_nonzero = P_joint[P_joint > 0]
    h_uy = -np.sum(P_joint_nonzero * np.log2(P_joint_nonzero))

    # 4. MI = H(U) + H(Y) - H(U, Y)
    mi = h_u + h_y - h_uy
    
    # 5. Return Uncertainty Coefficient (fraction of H(Y) explained by U) bounded in [0, 1]
    return float(np.clip(mi / h_y, 0.0, 1.0))

def save_results_to_file(results_all, results_by_dim, approaches, n_runs, alpha=0.05, suffix=""):
    """Save comprehensive summary to a .txt file."""
    import io
    from contextlib import redirect_stdout

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_suffix = f"_{suffix}" if suffix else ""
    filename = f"results/uncertainty_quantification_results{file_suffix}_{timestamp}.txt"

    string_buffer = io.StringIO()
    with redirect_stdout(string_buffer):
        print_comprehensive_summary(results_all, results_by_dim, approaches, n_runs, alpha=alpha)

    summary_text = string_buffer.getvalue()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"EPISTEMIC UNCERTAINTY QUANTIFICATION - RESULTS SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(summary_text)
        f.write(f"\n{'='*80}\n")
        f.write(f"End of Report\n")
        f.write(f"{'='*80}\n")

    print(f"\n[Report] Results saved to: {filename}")
    return filename

def print_comprehensive_summary(results_all, results_by_dim, approaches, n_runs, alpha=0.05):
    print(f"\n\n{'='*80}")
    print(f"COMPREHENSIVE STATISTICAL SUMMARY")
    print(f"{'='*80}\n")

    metrics = ["auroc", "spearman", "brier", "mi", "jsd"]
    dimensions = [("All Functions", results_all),
                  ("1D Functions", results_by_dim["1D"]),
                  ("2D Functions", results_by_dim["2D"]),
                  ("3D Functions", results_by_dim["3D"]),
                  ("4D Functions", results_by_dim["4D"]),
                  ("5D Functions", results_by_dim["5D"])]

    for metric in metrics:
        print(f"\n{'-'*80}")
        print(f"METRIC: {metric.upper()}")
        print(f"{'-'*80}\n")

        print(f"{'DESCRIPTIVE STATISTICS':^80}")
        header = f"{'':<20} " + " ".join(f"{app:>18}" for app in approaches)
        print(header)
        print(f"{'-'*80}")

        for dim_name, results_dict in dimensions:
            print(f"{dim_name:<20}", end="")
            for app in approaches:
                values = np.array([v for v in results_dict[app][metric] if not np.isnan(v)])
                if len(values) > 0:
                    print(f" {np.mean(values):.4f}+/-{np.std(values):.4f}  ", end="")
                else:
                    print(f" {'N/A':>18}", end="")
            print()

        print(f"\n{'STATISTICAL TESTS (Friedman + Bonferroni-corrected Wilcoxon)':^80}\n")

        for dim_name, results_dict in dimensions:
            print(f"> {dim_name}")
            
            # FIXED: Reshape and average across seeds to eliminate Pseudo-Replication
            total_items = len(results_dict[approaches[0]][metric])
            n_functions = total_items // n_runs
            
            processed_data = []
            for app in approaches:
                flat_vals = np.array(results_dict[app][metric], dtype=float)
                matrix = flat_vals.reshape(n_runs, n_functions)
                processed_data.append(np.nanmean(matrix, axis=0))
                
            valid_mask = ~np.isnan(processed_data).any(axis=0)
            data = [d[valid_mask] for d in processed_data]

            if all(len(d) > 2 for d in data):
                stat, p_f = friedmanchisquare(*data)
                sig_symbol = "***" if p_f < 0.001 else "**" if p_f < 0.01 else "*" if p_f < alpha else "ns"
                print(f"  Friedman: chi2 = {stat:8.4f}, p = {p_f:.4e} {sig_symbol}")

                if p_f < alpha:
                    import itertools
                    pairs = list(itertools.combinations(approaches, 2))
                    alpha_bonf = alpha / len(pairs)
                    print(f"  Bonferroni alpha = {alpha_bonf:.4e}")
                    print(f"  {'Pairwise Comparisons':<30} {'p-value':<15} {'Significant?':<15}")
                    print(f"  {'-'*65}")

                    for app1, app2 in pairs:
                        idx1 = approaches.index(app1)
                        idx2 = approaches.index(app2)
                        try:
                            _, p_w = wilcoxon(data[idx1], data[idx2])
                        except ValueError:
                            p_w = 1.0  # Safe fallback if differences are all zero
                        sig = "[SIG]" if p_w < alpha_bonf else "[NS]"
                        pair_str = f"{app1} vs {app2}"
                        print(f"  {pair_str:<30} {p_w:>14.4e} {sig:>15}")
                else:
                    print(f"  -> No significant difference across methods (Friedman p >= {alpha})")
            else:
                print("  -> Not enough valid independent functions (blocks) to perform paired testing (Requires >= 3)")
            print()

    print(f"{'='*80}")
    print(f"Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print(f"{'='*80}\n")

def run_single_test(func_dict, func_name, seed, approaches, rf_config=1, k_neighbors='auto', gap_type='empty', sparse_multiplier=12, scaling_law='linear'):
    # Determine standard Random Forest hyperparameters based on config selection to pass min_leaf to generate_data
    if rf_config == 1:
        n_est, min_leaf = 100, 5
    elif rf_config == 2:
        n_est, min_leaf = 100, 25
    elif rf_config == 3:
        n_est, min_leaf = 100, 50
    elif rf_config == 4:
        n_est, min_leaf = 300, 10
    else: # Config 5
        n_est, min_leaf = 300, 30

    X_train, y_train, X_test, y_test, y_true_binary = generate_data(
        func_dict, func_name, seed, gap_type=gap_type, sparse_multiplier=sparse_multiplier,
        scaling_law=scaling_law, min_samples_leaf=min_leaf
    )

    rf = RandomForestRegressor(n_estimators=n_est, min_samples_leaf=min_leaf, random_state=seed)
    rf.fit(X_train, y_train)
    quantifier = EpistemicQuantifier(rf, X_train, y_train)

    y_pred = rf.predict(X_test)
    sq_error = (y_test - y_pred)**2
    gap_mask = y_true_binary == 1

    results = {}
    u_a = quantifier.base_get_aleatoric_variance(X_test)

    uncertainties = {}
    u_a_credal_dict = {}
    for app in approaches:
        if app == "Standard": uncertainties[app] = quantifier.standard_get_epistemic_variance(X_test)
        elif app == "Shaker": uncertainties[app] = quantifier.shaker_get_epistemic_variance(X_test, random_state=seed)
        elif app == "Chen": uncertainties[app] = quantifier.chen_get_epistemic_variance(X_test)
        elif app == "Credal_GL_Bisect":
            credal_q = CredalRegressionUQ(rf, X_train, y_train)
            u_e_credal, u_a_credal = credal_q.compute_uq(X_test, backend="auto", integration_method="gauss_legendre", sup_solver="bisection")
            uncertainties[app] = u_e_credal
            u_a_credal_dict[app] = u_a_credal
        elif app == "Credal_GL_Newton":
            credal_q = CredalRegressionUQ(rf, X_train, y_train)
            u_e_credal, u_a_credal = credal_q.compute_uq(X_test, backend="auto", integration_method="gauss_legendre", sup_solver="newton")
            uncertainties[app] = u_e_credal
            u_a_credal_dict[app] = u_a_credal
        elif app == "Credal_Trapz_Bisect":
            credal_q = CredalRegressionUQ(rf, X_train, y_train)
            u_e_credal, u_a_credal = credal_q.compute_uq(X_test, backend="auto", integration_method="trapezoid", sup_solver="bisection")
            uncertainties[app] = u_e_credal
            u_a_credal_dict[app] = u_a_credal
        elif app == "Credal_Trapz_Newton":
            credal_q = CredalRegressionUQ(rf, X_train, y_train)
            u_e_credal, u_a_credal = credal_q.compute_uq(X_test, backend="auto", integration_method="trapezoid", sup_solver="newton")
            uncertainties[app] = u_e_credal
            u_a_credal_dict[app] = u_a_credal
        elif app == "Proximity":
            prox_q = ProximityRegressionUQ(rf, X_train, y_train)
            uncertainties[app] = prox_q.compute_uq(X_test, n_neighbors=k_neighbors, level=0.95)


    for app in approaches:
        u_e = uncertainties[app]
        results[app] = {"auroc": None, "spearman": None, "brier": None, "mi": None, "jsd": None}

        results[app]["auroc"] = roc_auc_score(y_true_binary, u_e)
        if np.any(gap_mask):
            if app in u_a_credal_dict and u_a_credal_dict[app] is not None:
                spear_corr, _ = spearmanr(sq_error[gap_mask], (u_e + u_a_credal_dict[app])[gap_mask])
            else:
                spear_corr, _ = spearmanr(sq_error[gap_mask], (u_e + u_a)[gap_mask])
            results[app]["spearman"] = spear_corr
        else:
            results[app]["spearman"] = np.nan

        # Sigmoid Calibration (Platt Scaling) to map epistemic uncertainty to OOD probability
        try:
            # Revert to standard L2 regularization (C=1.0)
            lr = LogisticRegression(C=1.0)
            lr.fit(u_e.reshape(-1, 1), y_true_binary)
            p_calibrated = lr.predict_proba(u_e.reshape(-1, 1))[:, 1]
        except Exception:
            # Fallback to min-max normalization if logistic regression fails
            u_min = np.min(u_e)
            u_max = np.max(u_e)
            u_range = u_max - u_min + 1e-10
            p_calibrated = (u_e - u_min) / u_range
        # Balanced Brier Score to prevent dominance by class imbalance in higher dimensions
        brier_id = np.mean((p_calibrated[y_true_binary == 0] - 0) ** 2)
        brier_ood = np.mean((p_calibrated[y_true_binary == 1] - 1) ** 2)
        results[app]["brier"] = 0.5 * (brier_id + brier_ood)

        results[app]["mi"] = calculate_mutual_information(u_e, y_true_binary)
        results[app]["jsd"] = calculate_jensen_shannon_divergence(u_e, y_true_binary)

    return results

def print_results(results_dict, test_name):
    print(f"\n{'='*70}")
    print(f"{test_name}")
    print(f"{'='*70}")

    for metric in ["auroc", "spearman", "brier", "mi", "jsd"]:
        print(f"\n--- {metric.upper()} ---")
        for app in results_dict:
            values = np.array([v for v in results_dict[app][metric] if not np.isnan(v)])
            if len(values) > 0:
                print(f"{app:12s}: Mean = {np.mean(values):.4f}, Std = {np.std(values):.4f}")

def run_statistical_tests(results_dict, approaches, n_runs, alpha=0.05):
    print(f"\n--- Statistical Validation (alpha = {alpha}) ---")

    for metric in ["auroc", "spearman", "brier", "mi", "jsd"]:
        print(f"\n{metric.upper()}:")
        
        # FIXED: Reshape and average across seeds to eliminate Pseudo-Replication bias
        total_items = len(results_dict[approaches[0]][metric])
        n_functions = total_items // n_runs
        
        processed_data = []
        for app in approaches:
            flat_vals = np.array(results_dict[app][metric], dtype=float)
            matrix = flat_vals.reshape(n_runs, n_functions)
            processed_data.append(np.nanmean(matrix, axis=0))
            
        valid_mask = ~np.isnan(processed_data).any(axis=0)
        data = [d[valid_mask] for d in processed_data]

        if not all(len(d) > 2 for d in data):
            print("  Result: NOT ENOUGH VALID DATA FOR Paired Testing (Requires >= 3 independent functions)")
            continue

        if len(approaches) == 2:
            print("  Bypassing Friedman Test (requires >= 3 approaches). Running paired Wilcoxon Signed-Rank Test directly:")
            app1, app2 = approaches[0], approaches[1]
            try:
                _, p_w = wilcoxon(data[0], data[1])
            except ValueError:
                p_w = 1.0  # Safe fallback if differences are all zero
            sig = "[SIG]" if p_w < alpha else "[NS]"
            print(f"    {app1} vs {app2:<25}: p = {p_w:.4e} ({sig})")
        else:
            # len(approaches) >= 3
            stat, p_f = friedmanchisquare(*data)
            print(f"  Friedman Test: chi2 = {stat:.4f}, p = {p_f:.4e}")

            if p_f < alpha:
                print(f"  Result: SIGNIFICANT (p < {alpha})")
                import itertools
                pairs = list(itertools.combinations(approaches, 2))
                alpha_bonf = alpha / len(pairs)
                print(f"  Bonferroni-corrected alpha = {alpha_bonf:.4e}")

                for app1, app2 in pairs:
                    idx1 = approaches.index(app1)
                    idx2 = approaches.index(app2)
                    try:
                        _, p_w = wilcoxon(data[idx1], data[idx2])
                    except ValueError:
                        p_w = 1.0  # Safe fallback if differences are all zero
                    sig = "[SIG]" if p_w < alpha_bonf else "[NS]"
                    pair_str = f"{app1} vs {app2}"
                    print(f"    {pair_str:<25}: p = {p_w:.4e} ({sig})")
            else:
                print(f"  Result: NOT SIGNIFICANT (p >= {alpha})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Epistemic UQ Benchmarks")
    parser.add_argument("--rf_config", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Random Forest Config ID")
    parser.add_argument("--k_neighbors", type=str, default="20", help="Neighborhood size for Proximity UQ (int or 'auto' or 'all')")
    parser.add_argument("--gap_type", type=str, default="empty", choices=["empty", "sparse"], help="OOD gap type")
    parser.add_argument("--sparse_multiplier", type=int, default=12, help="Multiplier for sparse gap points (n_keep = multiplier * ndim)")
    parser.add_argument("--scaling_law", type=str, default="linear", choices=["linear", "fractional", "leaf"], help="Scaling law for sparse gap points")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of seeds/runs")
    args = parser.parse_args()

    rf_config_arg = args.rf_config
    try:
        k_neighbors_arg = int(args.k_neighbors)
    except ValueError:
        k_neighbors_arg = args.k_neighbors  # 'auto' or 'all'
    gap_type_arg = args.gap_type
    sparse_multiplier_arg = args.sparse_multiplier
    scaling_law_arg = args.scaling_law
    n_runs = args.n_runs

    start_time = time.time()
    print(f"\n{'='*70}")
    print(f"EPISTEMIC UNCERTAINTY QUANTIFICATION - COMPREHENSIVE TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: RF Config={rf_config_arg}, K Neighbors={k_neighbors_arg}, Gap Type={gap_type_arg}, Multiplier={sparse_multiplier_arg}, Scaling Law={scaling_law_arg}, Runs={n_runs}")
    print(f"{'='*70}")

    approaches = ["Standard", "Proximity"]
    alpha = 0.05

    functions_1d = get_1d_functions()
    functions_2d = get_2d_functions()
    functions_3d = get_3d_functions()
    functions_4d = get_4d_functions()
    functions_5d = get_5d_functions()
    functions_6d = get_6d_functions()
    functions_7d = get_7d_functions()
    functions_8d = get_8d_functions()
    functions_9d = get_9d_functions()
    functions_10d = get_10d_functions()
    all_functions = {
        **functions_1d, **functions_2d, **functions_3d, **functions_4d, **functions_5d,
        **functions_6d, **functions_7d, **functions_8d, **functions_9d, **functions_10d
    }

    print(f"\n[SETUP SUMMARY]")
    print(f"  * Functions: {len(all_functions)} total (5 1D, 5 2D, 5 3D, 3 4D, 3 5D, 3 6D, 3 7D, 3 8D, 3 9D, 3 10D)")
    print(f"  * Runs: {n_runs} (total evaluations: {len(all_functions) * n_runs})")
    print(f"  * Approaches: {', '.join(approaches)}")
    print(f"  * Metrics: AUROC, Spearman, Brier, MI, JSD")
    print(f"  * Statistical tests: Friedman + Bonferroni-corrected Wilcoxon (alpha={alpha})")
    sys.stdout.flush()

    # ====================
    # UNIFIED TEST: Single Pass - Aggregate by Dimension
    # ====================
    print(f"\n\n{'#'*70}")
    print("# UNIFIED TEST: All Functions (aggregated by dimension)")
    print(f"{'#'*70}")
    print(f"Running {len(all_functions)} functions x {n_runs} seeds = {len(all_functions) * n_runs} evaluations (single pass)\n")

    results_all = {app: {"auroc": [], "spearman": [], "brier": [], "mi": [], "jsd": []} for app in approaches}
    results_by_dim = {
        "1D": {app: {"auroc": [], "spearman": [], "brier": [], "mi": [], "jsd": []} for app in approaches},
        "2D": {app: {"auroc": [], "spearman": [], "brier": [], "mi": [], "jsd": []} for app in approaches},
        "3D": {app: {"auroc": [], "spearman": [], "brier": [], "mi": [], "jsd": []} for app in approaches},
        "4D": {app: {"auroc": [], "spearman": [], "brier": [], "mi": [], "jsd": []} for app in approaches},
        "5D": {app: {"auroc": [], "spearman": [], "brier": [], "mi": [], "jsd": []} for app in approaches},
        "6D": {app: {"auroc": [], "spearman": [], "brier": [], "mi": [], "jsd": []} for app in approaches},
        "7D": {app: {"auroc": [], "spearman": [], "brier": [], "mi": [], "jsd": []} for app in approaches},
        "8D": {app: {"auroc": [], "spearman": [], "brier": [], "mi": [], "jsd": []} for app in approaches},
        "9D": {app: {"auroc": [], "spearman": [], "brier": [], "mi": [], "jsd": []} for app in approaches},
        "10D": {app: {"auroc": [], "spearman": [], "brier": [], "mi": [], "jsd": []} for app in approaches},
    }

    test_start = time.time()
    for seed in range(n_runs):
        seed_start = time.time()
        print(f"[RUN {seed+1}/{n_runs}] ", end="", flush=True)

        for func_name in all_functions:
            try:
                test_results = run_single_test(
                    all_functions, func_name, seed, approaches,
                    rf_config=rf_config_arg, k_neighbors=k_neighbors_arg,
                    gap_type=gap_type_arg, sparse_multiplier=sparse_multiplier_arg,
                    scaling_law=scaling_law_arg
                )

                if func_name in functions_1d:
                    dim_key = "1D"
                elif func_name in functions_2d:
                    dim_key = "2D"
                elif func_name in functions_3d:
                    dim_key = "3D"
                elif func_name in functions_4d:
                    dim_key = "4D"
                elif func_name in functions_5d:
                    dim_key = "5D"
                elif func_name in functions_6d:
                    dim_key = "6D"
                elif func_name in functions_7d:
                    dim_key = "7D"
                elif func_name in functions_8d:
                    dim_key = "8D"
                elif func_name in functions_9d:
                    dim_key = "9D"
                else:
                    dim_key = "10D"

                for app in approaches:
                    results_all[app]["auroc"].append(test_results[app]["auroc"])
                    results_all[app]["spearman"].append(test_results[app]["spearman"])
                    results_all[app]["brier"].append(test_results[app]["brier"])
                    results_all[app]["mi"].append(test_results[app]["mi"])
                    results_all[app]["jsd"].append(test_results[app]["jsd"])

                    results_by_dim[dim_key][app]["auroc"].append(test_results[app]["auroc"])
                    results_by_dim[dim_key][app]["spearman"].append(test_results[app]["spearman"])
                    results_by_dim[dim_key][app]["brier"].append(test_results[app]["brier"])
                    results_by_dim[dim_key][app]["mi"].append(test_results[app]["mi"])
                    results_by_dim[dim_key][app]["jsd"].append(test_results[app]["jsd"])

            except Exception as e:
                print(f"\n[ERROR] in seed={seed}, func={func_name}: {str(e)}")
                sys.stdout.flush()
                raise

        seed_time = time.time() - seed_start
        remaining_seeds = n_runs - (seed + 1)
        eta_total_sec = seed_time * remaining_seeds
        eta_min = int(eta_total_sec / 60)

        print(f" [OK] ({seed_time:.1f}s, ETA: {eta_min}m remaining)")
        sys.stdout.flush()

    test_time = time.time() - test_start
    total_time = time.time() - start_time
    print(f"\n[SUCCESS] Unified test completed in {test_time/60:.1f} minutes")
    sys.stdout.flush()

    # ====================
    # TEST 1: All Functions Together
    # ====================
    print(f"\n\n{'#'*70}")
    print("# TEST 1: ALL FUNCTIONS TOGETHER")
    print(f"{'#'*70}")
    print_results(results_all, f"ALL FUNCTIONS ({len(all_functions)} x {n_runs} = {len(all_functions) * n_runs} tests)")
    run_statistical_tests(results_all, approaches, n_runs, alpha=alpha)
    sys.stdout.flush()

    # ====================
    # TEST 2: By Dimension
    # ====================
    print(f"\n\n{'#'*70}")
    print("# TEST 2: BY DIMENSION")
    print(f"{'#'*70}\n")

    for dim_name, dim_key in [
        ("1D Functions", "1D"), ("2D Functions", "2D"), ("3D Functions", "3D"),
        ("4D Functions", "4D"), ("5D Functions", "5D"), ("6D Functions", "6D"),
        ("7D Functions", "7D"), ("8D Functions", "8D"), ("9D Functions", "9D"),
        ("10D Functions", "10D")
    ]:
        print(f"\n[DIMENSION] {dim_name}")
        print(f"{'-'*70}")
        print_results(results_by_dim[dim_key], f"{dim_name} (runs = {n_runs})")
        run_statistical_tests(results_by_dim[dim_key], approaches, n_runs, alpha=alpha)
        sys.stdout.flush()

    # ====================
    # Final Summary & Auto-Save
    # ====================
    print(f"\n\n{'='*70}")
    print(f"[SUCCESS] ALL TESTS COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Total Runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Evaluations: {len(all_functions) * n_runs} tests (single-pass optimization)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    # Print comprehensive report summary to terminal
    print_comprehensive_summary(results_all, results_by_dim, approaches, n_runs, alpha=alpha)
    
    # Executing file generator to dump everything into a clean timestamped report txt file
    save_results_to_file(
        results_all, results_by_dim, approaches, n_runs, alpha=alpha,
        suffix=f"rf{rf_config_arg}_k{k_neighbors_arg}_{gap_type_arg}_m{sparse_multiplier_arg}_{scaling_law_arg}"
    )
    sys.stdout.flush()


