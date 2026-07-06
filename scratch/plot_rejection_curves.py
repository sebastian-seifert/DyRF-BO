import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator import generate_data
from Epistemic_Quantifier import EpistemicQuantifier
from Credal_Regression_UQ import CredalRegressionUQ
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
from metrics import (
    calculate_rejection_curve,
    calculate_aurc,
    calculate_oracle_rejection_curve,
    calculate_random_rejection_curve
)

def main():
    print("Generating data and training Random Forest...")
    # Define a 1D function with an empty gap
    func_dict = {
        "sin": {
            "func": lambda x: np.sin(x),
            "gap": [4.0, 6.0],
            "range": [0.0, 10.0]
        }
    }
    
    # 1. Generate train and test data
    X_train, y_train, X_test, y_test, y_true_binary = generate_data(
        func_dict, "sin", seed=42, points_per_dim=150, gap_type="empty", min_samples_leaf=5
    )
    
    # 2. Fit Random Forest
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    rf.oob_prediction_ = rf.predict(X_train)
    
    y_pred = rf.predict(X_test)
    rejection_rates = np.linspace(0.0, 0.95, 96)
    
    # 3. Instantiate UQ Quantifiers
    quantifier = EpistemicQuantifier(rf, X_train, y_train)
    credal = CredalRegressionUQ(rf, X_train, y_train)
    
    # 4. Compute Epistemic Uncertainties
    print("Computing uncertainties for standard and shaker methods...")
    u_standard = quantifier.standard_get_epistemic_variance(X_test)
    u_chen = quantifier.chen_get_epistemic_variance(X_test)
    u_gmm = quantifier.shaker_get_epistemic_variance(X_test, random_state=42)
    
    print("Computing uncertainties for Credal UQ...")
    u_credal, _ = credal.compute_uq(X_test, backend="cpu", integration_method="gauss_legendre", sup_solver="bisection")
    
    print("Computing uncertainties for Proximity methods...")
    # Standard Proximity (Baseline)
    prox_std = GPUProximityRegressionUQ(rf, X_train, y_train, device="cpu", topological_decay_lambda=None)
    u_prox_std = prox_std.compute_uq(X_test, n_neighbors=20, level=0.95)
    
    # Method B (TWQ - Topological Weighted Quantiles)
    prox_twq = GPUProximityRegressionUQ(rf, X_train, y_train, device="cpu", topological_decay_lambda=1.0)
    u_prox_twq = prox_twq.compute_uq(X_test, n_neighbors="auto", level=0.95)
    
    # 5. Calculate Rejection Curves
    print("Calculating error-rejection curves...")
    curve_standard = calculate_rejection_curve(u_standard, y_pred, y_test, rejection_rates)
    curve_chen = calculate_rejection_curve(u_chen, y_pred, y_test, rejection_rates)
    curve_gmm = calculate_rejection_curve(u_gmm, y_pred, y_test, rejection_rates)
    curve_credal = calculate_rejection_curve(u_credal, y_pred, y_test, rejection_rates)
    curve_prox_std = calculate_rejection_curve(u_prox_std, y_pred, y_test, rejection_rates)
    curve_prox_twq = calculate_rejection_curve(u_prox_twq, y_pred, y_test, rejection_rates)
    
    # Reference Bounds
    curve_oracle = calculate_oracle_rejection_curve(y_pred, y_test, rejection_rates)
    curve_random = calculate_random_rejection_curve(y_pred, y_test, rejection_rates, n_shuffles=50, random_state=42)
    
    # Calculate Areas
    aurc_std = calculate_aurc(rejection_rates, curve_standard)
    aurc_chen = calculate_aurc(rejection_rates, curve_chen)
    aurc_gmm = calculate_aurc(rejection_rates, curve_gmm)
    aurc_credal = calculate_aurc(rejection_rates, curve_credal)
    aurc_prox_std = calculate_aurc(rejection_rates, curve_prox_std)
    aurc_prox_twq = calculate_aurc(rejection_rates, curve_prox_twq)
    aurc_oracle = calculate_aurc(rejection_rates, curve_oracle)
    aurc_random = calculate_aurc(rejection_rates, curve_random)
    
    # 6. Plotting
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    
    # Plot standard bounds in gray/dashed styles
    ax.plot(rejection_rates * 100, curve_oracle, color='black', linestyle='--', linewidth=2.5, label=f"Oracle (AURC = {aurc_oracle:.4f})")
    ax.plot(rejection_rates * 100, curve_random, color='gray', linestyle=':', linewidth=2.0, label=f"Random Rejection (AURC = {aurc_random:.4f})")
    
    # Plot UQ approaches
    ax.plot(rejection_rates * 100, curve_standard, color='#d62728', linewidth=1.8, label=f"Standard Disagreement (AURC = {aurc_std:.4f})")
    ax.plot(rejection_rates * 100, curve_chen, color='#ff7f0e', linewidth=1.8, label=f"Chen Paired Variance (AURC = {aurc_chen:.4f})")
    ax.plot(rejection_rates * 100, curve_gmm, color='#2ca02c', linewidth=1.8, label=f"Shaker GMM Entropy (AURC = {aurc_gmm:.4f})")
    
    # Proximity approaches
    ax.plot(rejection_rates * 100, curve_prox_std, color='#9467bd', linewidth=1.8, label=f"Standard Proximity (AURC = {aurc_prox_std:.4f})")
    ax.plot(rejection_rates * 100, curve_prox_twq, color='#17becf', linewidth=1.8, label=f"Method B: Proximity TWQ (AURC = {aurc_prox_twq:.4f})")
    
    # Highlight Credal Likelihood
    ax.plot(rejection_rates * 100, curve_credal, color='#1f77b4', linewidth=3.0, label=f"Shaker Likelihood [Credal] (AURC = {aurc_credal:.4f})")
    
    ax.set_title("Error-Rejection Curve Comparison (1D Sin with Empty OOD Gap)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Rejection Rate (%)", fontsize=12)
    ax.set_ylabel("Remaining Prediction Error (MSE)", fontsize=12)
    ax.set_xlim(0, 95)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Adjust layout
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/rejection_curves_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"Rejection curves plot successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
