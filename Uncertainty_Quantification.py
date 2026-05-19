# Todo: Change the MSE to an NLPD / Log-Likelihood based evaluation for a more principled approach.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, friedmanchisquare, wilcoxon
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

def calculate_cohens_d(x1, x2):
    """Calculates Cohen's d effect size."""
    n1, n2 = len(x1), len(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / pooled_std

def evaluate_epistemic(name, X_test, y_true, quantifier, epistemic_var=None, is_shaker=False):
    """
    Evaluates Epistemic Uncertainty. Returns NLL for the 'Gap' region (x in [4, 6]).
    """
    # Identify the Gap region
    gap_mask = (X_test.ravel() >= 4) & (X_test.ravel() <= 6)
    X_gap, y_gap = X_test[gap_mask], y_true[gap_mask]
    
    if is_shaker:
        mus = np.stack([t.predict(X_gap) for t in quantifier.model.estimators_])
        vars2 = quantifier.calculate_per_tree_var(X_gap)
        gap_nll = calculate_nll_gmm(y_gap, mus, vars2)
    else:
        # Standard/Chen
        aleat_gap = quantifier.get_aleatoric_variance(X_gap)
        epi_gap = epistemic_var[gap_mask]
        total_var_gap = aleat_gap + epi_gap
        y_pred_gap = quantifier.model.predict(X_gap)
        gap_nll = calculate_nll_gaussian(y_gap, y_pred_gap, total_var_gap)
    
    return gap_nll

def plot_uncertainty(name, X_test, y_test, y_pred, var_pred, X_train, y_train):
    # Plotting is currently disabled for batch runs
    pass

def run_experiment(n_runs=50):
    results_gap = {
        "Standard Disagreement": [],
        "Shaker GMM": [],
        "Chen Stability": []
    }

    print(f"Running {n_runs} experiments (Gap-only evaluation)...")

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

        # 3. Evaluate (Gap-only NLL)
        results_gap["Standard Disagreement"].append(
            evaluate_epistemic("Standard", X_test, y_test, quantifier, quantifier.get_standard_disagreement(X_test))
        )
        results_gap["Shaker GMM"].append(
            evaluate_epistemic("Shaker", X_test, y_test, quantifier, is_shaker=True)
        )
        results_gap["Chen Stability"].append(
            evaluate_epistemic("Chen", X_test, y_test, quantifier, quantifier.get_chen_stability_epistemic(X_test))
        )

        if (seed + 1) % 10 == 0:
            print(f"Progress: {seed + 1}/{n_runs} runs complete.")

    # 4. Aggregate Results
    print(f"\n{'Approach':<25} | {'Mean Gap NLL':<12} | {'Std Dev':<10}")
    print("-" * 55)
    for name, nlls in results_gap.items():
        print(f"{name:<25} | {np.mean(nlls):<12.4f} | {np.std(nlls):<10.4f}")

    # 5. Friedman Test (on Gap NLL)
    stat, p_value = friedmanchisquare(
        results_gap["Standard Disagreement"],
        results_gap["Shaker GMM"],
        results_gap["Chen Stability"]
    )

    print("\n--- Statistical Significance (Friedman Test on Gap) ---")
    print(f"Statistic: {stat:.4f}")
    print(f"P-value:   {p_value:.4e}")

    if p_value < 0.05:
        print("Result: Significant difference found in the gap (p < 0.05).")
        
        # 6. Post-hoc Wilcoxon Signed-Rank Tests with Bonferroni Correction
        pairs = [
            ("Shaker GMM", "Standard Disagreement"),
            ("Shaker GMM", "Chen Stability"),
            ("Standard Disagreement", "Chen Stability")
        ]
        alpha = 0.05
        bonferroni_alpha = alpha / len(pairs)
        
        print(f"\n--- Post-hoc Analysis (Wilcoxon Signed-Rank + Bonferroni Correction) ---")
        print(f"{'Comparison':<45} | {'p-value':<10} | {'Sig.':<5} | {'Cohen\'s d':<10}")
        print("-" * 80)
        
        for g1_name, g2_name in pairs:
            g1 = results_gap[g1_name]
            g2 = results_gap[g2_name]
            
            w_stat, p_val = wilcoxon(g1, g2)
            d = calculate_cohens_d(g1, g2)
            is_sig = p_val < bonferroni_alpha
            
            sig_str = "YES" if is_sig else "NO"
            print(f"{g1_name + ' vs ' + g2_name:<45} | {p_val:<10.2e} | {sig_str:<5} | {d:<10.4f}")
            
        print(f"\nBonferroni-corrected alpha: {bonferroni_alpha:.4f}")
        print("Note: Cohen's d > 0.8 is considered a large effect size.")
    else:
        print("Result: No significant difference found in the gap (p >= 0.05).")

if __name__ == "__main__":
    run_experiment(n_runs=50)
