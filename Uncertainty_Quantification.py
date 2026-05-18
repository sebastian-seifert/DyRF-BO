
# Todo: Change the MSE to an NLPD / Log-Likelihood based evaluation for a more principled approach.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

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

    def calculate_per_tree_sigma(self, X_test, min_sigma=1e-6):
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
        
        sigmas = np.zeros((n_trees, n_samples))
        
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
                
                leaf_to_var[leaf_id] = float(s2) + min_sigma
                
            # 3. Fast mapping of pre-calculated variances back to the test samples
            sigmas[i, :] = [leaf_to_var[lid] for lid in test_leaf_ids]
            
        return sigmas

    def aleatoric_uncertainty(self, sigmas):
        return np.sum(np.log2(2*np.pi*np.e*sigmas), axis=0) / 2*len(self.model.estimators_)


    
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

    def get_aleatoric_uncertainty(self, X_test):
        """
        Calculates the closed-form aleatoric uncertainty (mean differential entropy).
        Formula from Slide 4: (1/M) * sum( 0.5 * log2(2 * pi * e * sigma_hat^2) )
        """
        sigmas2 = self.calculate_per_tree_sigma(X_test) # (n_trees, n_samples)
        
        # Differential entropy of a Gaussian: 0.5 * log2(2 * pi * e * sigma^2)
        individual_entropies = 0.5 * np.log2(2 * np.pi * np.e * sigmas2)
        
        return np.mean(individual_entropies, axis=0)

    def get_total_uncertainty(self, X_test, n_grid=200):
        """
        Calculates the Total Uncertainty (Entropy of the GMM) via numerical integration.
        Formula from Slide 5: -Integral[ p(y|x) * log2(p(y|x)) ] dy
        """
        X_test = np.atleast_2d(X_test)
        n_samples = X_test.shape[0]
        
        # 1. Get components: means (mu) and variances (sigma2)
        # mu shape: (n_trees, n_samples), sigmas2 shape: (n_trees, n_samples)
        mu = np.stack([t.predict(X_test) for t in self.model.estimators_])
        sigmas2 = self.calculate_per_tree_sigma(X_test)
        sigmas = np.sqrt(sigmas2)
        n_trees = mu.shape[0]

        # 2. Define integration grid for each test point (dynamic y-range)
        mu_min, mu_max = np.min(mu, axis=0), np.max(mu, axis=0)
        sigma_max = np.max(sigmas, axis=0)
        
        y_start = mu_min - 5 * sigma_max
        y_end = mu_max + 5 * sigma_max
        
        # y_grid shape: (n_samples, n_grid)
        y_grid = np.linspace(y_start, y_end, n_grid).T
        dy = (y_end - y_start) / (n_grid - 1) # (n_samples,)

        # 3. Evaluate GMM Density p(y|x) for each grid point
        # We use broadcasting to evaluate (n_samples, n_grid, n_trees)
        # mu: (n_trees, n_samples) -> (n_samples, 1, n_trees)
        mu_b = mu.T[:, np.newaxis, :]
        sigmas_b = sigmas.T[:, np.newaxis, :]
        # y_grid: (n_samples, n_grid) -> (n_samples, n_grid, 1)
        y_b = y_grid[:, :, np.newaxis]

        # Component densities: (n_samples, n_grid, n_trees)
        exponent = -0.5 * ((y_b - mu_b) / sigmas_b)**2
        probs = (1.0 / (sigmas_b * np.sqrt(2 * np.pi))) * np.exp(exponent)
        
        # GMM Density: Mean across trees
        p_y = np.mean(probs, axis=2) # (n_samples, n_grid)
        
        # 4. Integrate -p(y) * log2(p_y) using Trapezoidal rule
        # Add epsilon to prevent log(0)
        entropy_integrand = -p_y * np.log2(p_y + 1e-12)
        
        # Use np.trapz over the grid axis (axis 1)
        total_entropy = np.trapz(entropy_integrand, x=y_grid, axis=1)
        
        return total_entropy

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

def evaluate_epistemic(name, X_test, y_true, y_pred, var_pred):
    """
    Evaluates Epistemic Uncertainty.
    In BO, we care if uncertainty is high in regions where we have NO data.
    We check the correlation between predicted uncertainty and actual error.
    """
    squared_errors = (y_true - y_pred)**2
    corr, _ = pearsonr(squared_errors, var_pred)
    
    print(f"--- {name} ---")
    print(f"Error-Uncertainty Correlation: {corr:.4f}")
    
    # Check if uncertainty is higher in the 'Gap' (OOD) region
    gap_mask = (X_test.ravel() > 4) & (X_test.ravel() < 6)
    if np.any(gap_mask):
        avg_gap_unc = np.mean(var_pred[gap_mask])
        avg_data_unc = np.mean(var_pred[~gap_mask])
        print(f"Uncertainty Ratio (Gap/Data): {avg_gap_unc/avg_data_unc:.2f}x")
    print("")

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
    save_path = os.path.join("figures", filename)
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")
    plt.show()

if __name__ == "__main__":
    # 1. Generate synthetic data with an "Exploration Gap" (OOD region)
    X = np.linspace(0, 10, 1000).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, 1000)

    # Create a gap in the training data between x=4 and x=6
    train_mask = (X.ravel() < 4) | (X.ravel() > 6)
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    # Test on the full range (including the gap)
    X_test = X
    y_test = y

    # 2. Train baseline model
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)

    # 3. Initialize Quantifier
    quantifier = EpistemicQuantifier(rf, X_train, y_train)
    y_pred = rf.predict(X_test)

    # 4. Compare Approaches
    # Standard
    var_std = quantifier.get_standard_disagreement(X_test)
    evaluate_epistemic("Standard Disagreement", X_test, y_test, y_pred, var_std)
    plot_uncertainty("Standard Disagreement", X_test, y_test, y_pred, var_std, X_train, y_train)

    # Shaker
    var_shaker = quantifier.get_shaker_epistemic(X_test)
    evaluate_epistemic("Shaker Epistemic", X_test, y_test, y_pred, var_shaker)
    plot_uncertainty("Shaker Epistemic", X_test, y_test, y_pred, var_shaker, X_train, y_train)

    # Chen
    var_chen = quantifier.get_chen_stability_epistemic(X_test)
    evaluate_epistemic("Chen Stability", X_test, y_test, y_pred, var_chen)
    plot_uncertainty("Chen Stability", X_test, y_test, y_pred, var_chen, X_train, y_train)
