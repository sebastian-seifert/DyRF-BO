# Todo: Change the MSE to an NLPD / Log-Likelihood based evaluation for a more principled approach.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from scipy.integrate import simpson, quad

"""
Bachelor Thesis: Epistemic Uncertainty Quantification
Student: Sebastian Seifert
Primary Focus: Quantifying uncertainty due to lack of data/exploration (Epistemic).
Evaluation Metric: Negative Log Predictive Density (NLPD/NLL) - The Gold Standard.
"""

class EpistemicQuantifier:
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

    def calculate_per_tree_var(self, X_test, min_var=1e-6):
        """
        Calculates the per-tree unbiased variances (sigma^2) for each sample in X_test.
        Pre-calculates variances for unique leaves to ensure O(N_test + N_train) complexity.
        """
        X_test = np.atleast_2d(X_test)
        n_trees = len(self.model.estimators_)
        n_samples = X_test.shape[0]
        all_test_leaf_ids = self.model.apply(X_test)
        variances = np.zeros((n_trees, n_samples))
        
        for i, estimator in enumerate(self.model.estimators_):
            train_leaf_ids = estimator.apply(self.X_train)
            test_leaf_ids = all_test_leaf_ids[:, i]
            unique_test_leaves = np.unique(test_leaf_ids)
            leaf_to_var = {}
            for leaf_id in unique_test_leaves:
                leaf_y = self.y_train[train_leaf_ids == leaf_id]
                s2 = np.var(leaf_y, ddof=1) if leaf_y.shape[0] > 1 else 0.0
                leaf_to_var[leaf_id] = float(s2) + min_var
            variances[i, :] = [leaf_to_var[lid] for lid in test_leaf_ids]
        return variances

    def get_aleatoric_variance(self, X_test):
        return np.mean(self.calculate_per_tree_var(X_test), axis=0)

    def get_aleatoric_uncertainty(self, X_test):
        vars2 = self.calculate_per_tree_var(X_test) 
        return np.mean(0.5 * np.log2(2 * np.pi * np.e * vars2), axis=0)

    def get_total_uncertainty(self, X_test):
        """
        Calculates GMM Entropy via Adaptive Quadrature (High Precision).
        """
        X_test = np.atleast_2d(X_test)
        n_samples = X_test.shape[0]
        mu_all = np.stack([t.predict(X_test) for t in self.model.estimators_])
        vars_all = self.calculate_per_tree_var(X_test)
        sigmas_all = np.sqrt(vars_all)
        total_entropy = np.zeros(n_samples)

        def gmm_entropy_integrand(y, mu, sigma):
            exponent = -0.5 * ((y - mu) / sigma)**2
            densities = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(exponent)
            p_y = np.mean(densities)
            return -p_y * np.log2(p_y) if p_y > 1e-15 else 0.0

        for i in range(n_samples):
            val, _ = quad(gmm_entropy_integrand, -np.inf, np.inf, args=(mu_all[:, i], sigmas_all[:, i]), limit=100)
            total_entropy[i] = val
        return total_entropy

    def get_standard_disagreement(self, X_test):
        tree_preds = np.stack([tree.predict(X_test) for tree in self.model.estimators_])
        return np.maximum(np.var(tree_preds, axis=0), 0.0)

    def get_shaker_epistemic(self, X_test):
        return np.maximum(self.get_total_uncertainty(X_test) - self.get_aleatoric_uncertainty(X_test), 0.0)

    def get_chen_stability_epistemic(self, X_test):
        tree_preds = np.stack([tree.predict(X_test) for tree in self.model.estimators_])
        M = tree_preds.shape[0]
        if M % 2 != 0: tree_preds = tree_preds[:-1]; M -= 1
        return np.sum((tree_preds[0::2] - tree_preds[1::2])**2, axis=0) / M

def calculate_nll_gaussian(y_true, y_pred, var_pred):
    return np.mean(0.5 * np.log(2 * np.pi * var_pred) + (y_true - y_pred)**2 / (2 * var_pred))

def calculate_nll_gmm(y_true, mus, vars2):
    sigmas = np.sqrt(vars2)
    exponent = -0.5 * ((y_true[:, np.newaxis] - mus.T) / sigmas.T)**2
    densities = (1.0 / (sigmas.T * np.sqrt(2 * np.pi))) * np.exp(exponent)
    return -np.mean(np.log(np.mean(densities, axis=1) + 1e-12))

def evaluate_epistemic(name, X_test, y_true, quantifier, epistemic_var=None, is_shaker=False):
    """
    Evaluates Epistemic Uncertainty using NLL as the Gold Standard.
    - For Standard/Chen: Total Var = Aleatoric + Epistemic
    - For Shaker: Direct GMM NLL
    """
    y_pred = quantifier.model.predict(X_test)
    aleatoric_var = quantifier.get_aleatoric_variance(X_test)
    
    if is_shaker:
        # Get raw components for GMM NLL
        mus = np.stack([t.predict(X_test) for t in quantifier.model.estimators_])
        vars2 = quantifier.calculate_per_tree_var(X_test)
        nll = calculate_nll_gmm(y_true, mus, vars2)
        total_var = quantifier.get_standard_disagreement(X_test) + aleatoric_var # Proxy for plotting
    else:
        # Single Gaussian assumption
        total_var = aleatoric_var + epistemic_var
        nll = calculate_nll_gaussian(y_true, y_pred, total_var)
    
    # Also check correlation (legacy metric)
    squared_errors = (y_true - y_pred)**2
    corr, _ = pearsonr(squared_errors, total_var)
    
    print(f"--- {name} ---")
    print(f"NLL (Gold Standard): {nll:.4f}")
    print(f"Error-Uncertainty Correlation: {corr:.4f}")
    
    # Check Gap behavior
    gap_mask = (X_test.ravel() > 4) & (X_test.ravel() < 6)
    if np.any(gap_mask):
        avg_gap_unc = np.mean(total_var[gap_mask])
        avg_data_unc = np.mean(total_var[~gap_mask])
        print(f"Uncertainty Ratio (Gap/Data): {avg_gap_unc/avg_data_unc:.2f}x")
    print("")
    return nll

def plot_uncertainty(name, X_test, y_test, y_pred, var_pred, X_train, y_train):
    # Plotting is currently disabled for batch runs
    pass
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train, y_train, color='black', s=10, alpha=0.3, label='Training Data')
    plt.plot(X_test, y_test, color='green', alpha=0.5, label='True Function')
    plt.plot(X_test, y_pred, color='blue', label='RF Prediction')
    std = np.sqrt(var_pred)
    plt.fill_between(X_test.ravel(), y_pred - 2*std, y_pred + 2*std, color='blue', alpha=0.2, label='2-sigma')
    plt.axvspan(4, 6, color='red', alpha=0.1, label='Gap')
    plt.title(f"UQ: {name}"); plt.legend()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(os.path.join("figures", f"uncertainty_{name.lower().replace(' ', '_')}.png"))
    plt.show()
    """

def run_experiment(n_runs=50):
    results = {
        "Standard Disagreement": [],
        "Shaker GMM": [],
        "Chen Stability": []
    }

    print(f"Running {n_runs} experiments for statistical robustness...")

    for seed in range(n_runs):
        # 1. Generate synthetic data with an "Exploration Gap"
        np.random.seed(seed)
        X = np.linspace(0, 10, 1000).reshape(-1, 1)
        y = np.sin(X).ravel() + np.random.normal(0, 0.1, 1000)
        train_mask = (X.ravel() < 4) | (X.ravel() > 6)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X, y

        # 2. Train baseline model
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=seed)
        rf.fit(X_train, y_train)
        quantifier = EpistemicQuantifier(rf, X_train, y_train)

        # 3. Evaluate
        # Standard
        results["Standard Disagreement"].append(
            evaluate_epistemic("Standard", X_test, y_test, quantifier, quantifier.get_standard_disagreement(X_test))
        )
        # Shaker
        results["Shaker GMM"].append(
            evaluate_epistemic("Shaker", X_test, y_test, quantifier, is_shaker=True)
        )
        # Chen
        results["Chen Stability"].append(
            evaluate_epistemic("Chen", X_test, y_test, quantifier, quantifier.get_chen_stability_epistemic(X_test))
        )

        if (seed + 1) % 10 == 0:
            print(f"Progress: {seed + 1}/{n_runs} runs complete.")

    # 4. Aggregate Results
    print(f"\n{'Approach':<25} | {'Mean NLL':<10} | {'Std Dev':<10}")
    print("-" * 50)
    for name, nlls in results.items():
        print(f"{name:<25} | {np.mean(nlls):<10.4f} | {np.std(nlls):<10.4f}")

if __name__ == "__main__":
    run_experiment(n_runs=500)
