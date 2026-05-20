
# Todo: Change the MSE to an NLPD / Log-Likelihood based evaluation for a more principled approach.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, friedmanchisquare, wilcoxon
from scipy.integrate import quad

"""
Bachelor Thesis: Epistemic Uncertainty Quantification
Student: James
Primary Focus: Quantifying uncertainty due to lack of data/exploration (Epistemic).
Evaluation Metric: Correlation with Error in Out-of-Distribution (OOD) regions.
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
        
        Returns: np.array of shape (n_trees, n_samples_test)
        """
        X_test = np.atleast_2d(X_test)
        n_trees = len(self.model.estimators_)
        n_samples = X_test.shape[0]
        
        # Get all leaf assignments for test data: shape (n_samples, n_trees)
        all_test_leaf_ids = self.model.apply(X_test)
        
        variances = np.zeros((n_trees, n_samples))
        
        for i, estimator in enumerate(self.model.estimators_) :
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

    def get_aleatoric_variance(self, X_test):
        """
        Returns the mean of the per-tree variances (E[sigma^2]).
        This is the baseline aleatoric uncertainty in terms of variance.
        """
        return np.mean(self.calculate_per_tree_var(X_test), axis=0)

    def get_aleatoric_uncertainty(self, X_test):
        """
        Calculates the closed-form aleatoric uncertainty (mean differential entropy).
        Formula from Slide 4: (1/M) * sum( 0.5 * log2(2 * pi * e * sigma_hat^2) )
        """
        vars2 = self.calculate_per_tree_var(X_test) 
        individual_entropies = 0.5 * np.log2(2 * np.pi * np.e * vars2)
        return np.mean(individual_entropies, axis=0)

    def get_total_uncertainty(self, X_test):
        """
        Calculates the Total Uncertainty (Entropy of the GMM) via Adaptive Quadrature.
        This is the most precise numerical method available for 1D integration.
        Formula from Slide 5: -Integral[ p(y|x) * log2(p(y|x)) ] dy over (-inf, inf)
        """
        X_test = np.atleast_2d(X_test)
        n_samples = X_test.shape[0]
        
        # 1. Get components for each tree
        mu_all = np.stack([t.predict(X_test) for t in self.model.estimators_]) # (n_trees, n_samples)
        vars_all = self.calculate_per_tree_var(X_test) # (n_trees, n_samples)
        sigmas_all = np.sqrt(vars_all)
        
        total_entropy = np.zeros(n_samples)

        # 2. Define the GMM entropy integrand for a single sample
        def gmm_entropy_integrand(y, mu, sigma):
            # Evaluate GMM density at point y
            exponent = -0.5 * ((y - mu) / sigma)**2
            densities = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(exponent)
            p_y = np.mean(densities)
            # Differential entropy in bits (log2)
            if p_y < 1e-15: return 0.0
            return -p_y * np.log2(p_y)

        # 3. Integrate for each test point using adaptive quadrature
        for i in range(n_samples):
            mu_i = mu_all[:, i]
            sigma_i = sigmas_all[:, i]
            
            # IMPROVEMENT: To prevent the integrator from missing sharp, distant peaks:
            # 1. Define a range that covers all means +/- 5 standard deviations
            lower_bound = np.min(mu_i - 5 * sigma_i)
            upper_bound = np.max(mu_i + 5 * sigma_i)
            
            # 2. Use 'points' to tell quad where the peaks are located
            # 3. Use a wide but finite range for better stability than -inf, inf
            val, err = quad(gmm_entropy_integrand, lower_bound, upper_bound, 
                            args=(mu_i, sigma_i), points=mu_i, limit=200)
            total_entropy[i] = val
        
        return total_entropy

    def get_standard_disagreement(self, X_test):
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
        # We use the expanded form for theoretical consistency with lectures, 
        # though it can be slightly less numerically stable for very large numbers.
        variance = mean_of_squares - square_of_mean
        
        # Ensure no negative variances due to floating point precision errors
        return np.maximum(variance, 0.0)

    def get_shaker_epistemic(self, X_test):
        """
        Approach 2: Shaker 2020 (Epistemic Component)
        Calculated as: Total Uncertainty (GMM Entropy) - Aleatoric Uncertainty.
        
        Total Uncertainty is the entropy of the Gaussian Mixture Model formed by the trees.
        Aleatoric is the mean entropy of the individual tree distributions.
        """
        total_unc = self.get_total_uncertainty(X_test)
        aleatoric_unc = self.get_aleatoric_uncertainty(X_test)
        
        # Epistemic = Total - Aleatoric
        return np.maximum(total_unc - aleatoric_unc, 0.0)

    def get_chen_stability_epistemic(self, X_test):
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
        # This is a statistically consistent estimator for the epistemic variance.
        return np.sum(squared_diffs, axis=0) / M

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
    filename = f"uncertainty_{name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()

if __name__ == "__main__":
    n_runs = 100
    approaches = ["Standard", "Shaker", "Chen"]
    
    # Storage for metrics
    results = {app: {"auroc": [], "spearman": []} for app in approaches}

    print(f"Starting {n_runs} runs for statistical validation...")

    for seed in range(n_runs):
        print(f"\n--- Run {seed+1}/{n_runs} ---")
        # 1. Generate fresh synthetic data
        np.random.seed(seed)
        X = np.linspace(0, 10, 1000).reshape(-1, 1)
        y = np.sin(X).ravel() + np.random.normal(0, 0.1, 1000)

        # Gap between x=4 and x=6
        train_mask = (X.ravel() < 4) | (X.ravel() > 6)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X, y

        # 2. Train RF
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=seed)
        rf.fit(X_train, y_train)
        quantifier = EpistemicQuantifier(rf, X_train, y_train)
        
        y_pred = rf.predict(X_test)
        sq_error = (y_test - y_pred)**2
        
        # OOD Labels: 1 for gap, 0 for outside
        y_true_binary = ((X_test.ravel() >= 4) & (X_test.ravel() <= 6)).astype(int)
        gap_mask = y_true_binary == 1

        # 3. Calculate metrics for each approach
        u_a = quantifier.get_aleatoric_variance(X_test)

        # --- Standard ---
        u_e_std = quantifier.get_standard_disagreement(X_test)
        results["Standard"]["auroc"].append(roc_auc_score(y_true_binary, u_e_std))
        results["Standard"]["spearman"].append(spearmanr(sq_error[gap_mask], (u_e_std + u_a)[gap_mask])[0])

        # --- Shaker ---
        u_e_shaker = quantifier.get_shaker_epistemic(X_test)
        results["Shaker"]["auroc"].append(roc_auc_score(y_true_binary, u_e_shaker))
        results["Shaker"]["spearman"].append(spearmanr(sq_error[gap_mask], (u_e_shaker + u_a)[gap_mask])[0])

        # --- Chen ---
        u_e_chen = quantifier.get_chen_stability_epistemic(X_test)
        results["Chen"]["auroc"].append(roc_auc_score(y_true_binary, u_e_chen))
        results["Chen"]["spearman"].append(spearmanr(sq_error[gap_mask], (u_e_chen + u_a)[gap_mask])[0])

    # 4. Summary Statistics
    for metric in ["auroc", "spearman"]:
        print(f"\n--- Summary Statistics for {metric.upper()} ---")
        for app in approaches:
            mean_val = np.mean(results[app][metric])
            std_val = np.std(results[app][metric])
            print(f"{app}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")

    # 5. Statistical Analysis
    for metric in ["auroc", "spearman"]:
        print(f"\n--- Statistical Validation for {metric.upper()} ---")
        data = [results[app][metric] for app in approaches]
        
        # Friedman Test
        stat, p_f = friedmanchisquare(*data)
        print(f"Friedman Test: p = {p_f:.4e}")
        
        if p_f < 0.05:
            # Pairwise Wilcoxon with Bonferroni
            pairs = [("Shaker", "Standard"), ("Shaker", "Chen"), ("Standard", "Chen")]
            alpha_adj = 0.05 / 3
            for app1, app2 in pairs:
                _, p_w = wilcoxon(results[app1][metric], results[app2][metric])
                sig = "YES" if p_w < alpha_adj else "NO"
                print(f"Wilcoxon {app1} vs {app2}: p = {p_w:.4e} (Sig: {sig})")
