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
from synthetic_functions import get_1d_functions, get_2d_functions

def run_evaluation(gap_type, ndim):
    print(f"\n==================================================")
    print(f"RUNNING EVALUATION FOR {ndim}D - GAP TYPE: {gap_type.upper()}")
    print(f"==================================================")
    
    if ndim == 1:
        func_dict = get_1d_functions()
        func_name = "sin"
        points_per_dim = 150
    else:
        func_dict = get_2d_functions()
        func_name = "sin_cos"
        points_per_dim = 50  # 50 * 50 = 2500 samples
        
    # 1. Generate train and test data
    X_train, y_train, X_test, y_test, y_true_binary = generate_data(
        func_dict, func_name, seed=42, points_per_dim=points_per_dim, gap_type=gap_type, min_samples_leaf=5
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
    
    print("Computing uncertainties for Proximity UQ...")
    # Standard Proximity (Baseline, K=20)
    prox_std = GPUProximityRegressionUQ(rf, X_train, y_train, device="cpu", use_density_scaling=False, topological_decay_lambda=None)
    u_prox_std = prox_std.compute_uq(X_test, n_neighbors=20, level=0.95)
    
    # Method A (TNS, K=20)
    prox_tns = GPUProximityRegressionUQ(rf, X_train, y_train, device="cpu", use_density_scaling=False, topological_decay_lambda=1.0)
    u_prox_tns = prox_tns.compute_uq(X_test, n_neighbors=20, level=0.95)
    
    # Method B (TWQ, K=auto)
    prox_twq = GPUProximityRegressionUQ(rf, X_train, y_train, device="cpu", use_density_scaling=False, topological_decay_lambda=1.0)
    u_prox_twq = prox_twq.compute_uq(X_test, n_neighbors="auto", level=0.95)
    
    # Method C (TDS, K=20, alpha=1.0)
    prox_tds = GPUProximityRegressionUQ(rf, X_train, y_train, device="cpu", use_density_scaling=True, density_scaling_alpha=1.0, topological_decay_lambda=5.0)
    u_prox_tds = prox_tds.compute_uq(X_test, n_neighbors=20, level=0.95)
    
    # Method B+C (TWQ+TDS, K=auto, alpha=1.0)
    prox_twq_tds = GPUProximityRegressionUQ(rf, X_train, y_train, device="cpu", use_density_scaling=True, density_scaling_alpha=1.0, topological_decay_lambda=5.0)
    u_prox_twq_tds = prox_twq_tds.compute_uq(X_test, n_neighbors="auto", level=0.95)
    
    # 5. Calculate Rejection Curves
    print("Calculating error-rejection curves...")
    curve_standard = calculate_rejection_curve(u_standard, y_pred, y_test, rejection_rates)
    curve_chen = calculate_rejection_curve(u_chen, y_pred, y_test, rejection_rates)
    curve_gmm = calculate_rejection_curve(u_gmm, y_pred, y_test, rejection_rates)
    curve_credal = calculate_rejection_curve(u_credal, y_pred, y_test, rejection_rates)
    curve_prox_std = calculate_rejection_curve(u_prox_std, y_pred, y_test, rejection_rates)
    curve_prox_tns = calculate_rejection_curve(u_prox_tns, y_pred, y_test, rejection_rates)
    curve_prox_twq = calculate_rejection_curve(u_prox_twq, y_pred, y_test, rejection_rates)
    curve_prox_tds = calculate_rejection_curve(u_prox_tds, y_pred, y_test, rejection_rates)
    curve_prox_twq_tds = calculate_rejection_curve(u_prox_twq_tds, y_pred, y_test, rejection_rates)
    
    # Reference Bounds
    curve_oracle = calculate_oracle_rejection_curve(y_pred, y_test, rejection_rates)
    curve_random = calculate_random_rejection_curve(y_pred, y_test, rejection_rates, n_shuffles=50, random_state=42)
    
    # Calculate Areas
    aurc_std = calculate_aurc(rejection_rates, curve_standard)
    aurc_chen = calculate_aurc(rejection_rates, curve_chen)
    aurc_gmm = calculate_aurc(rejection_rates, curve_gmm)
    aurc_credal = calculate_aurc(rejection_rates, curve_credal)
    aurc_prox_std = calculate_aurc(rejection_rates, curve_prox_std)
    aurc_prox_tns = calculate_aurc(rejection_rates, curve_prox_tns)
    aurc_prox_twq = calculate_aurc(rejection_rates, curve_prox_twq)
    aurc_prox_tds = calculate_aurc(rejection_rates, curve_prox_tds)
    aurc_prox_twq_tds = calculate_aurc(rejection_rates, curve_prox_twq_tds)
    aurc_oracle = calculate_aurc(rejection_rates, curve_oracle)
    aurc_random = calculate_aurc(rejection_rates, curve_random)
    
    print(f"\n--- AURC Results ({ndim}D {gap_type}) ---")
    print(f"Oracle AURC: {aurc_oracle:.6f}")
    print(f"Random AURC: {aurc_random:.6f}")
    print(f"Standard AURC: {aurc_std:.6f}")
    print(f"Chen AURC: {aurc_chen:.6f}")
    print(f"GMM AURC: {aurc_gmm:.6f}")
    print(f"Credal AURC: {aurc_credal:.6f}")
    print(f"Standard Proximity AURC: {aurc_prox_std:.6f}")
    print(f"Method A (TNS) AURC: {aurc_prox_tns:.6f}")
    print(f"Method B (TWQ) AURC: {aurc_prox_twq:.6f}")
    print(f"Method C (TDS) AURC: {aurc_prox_tds:.6f}")
    print(f"Method B+C (TWQ+TDS) AURC: {aurc_prox_twq_tds:.6f}")
    print("--------------------\n")
    
    # 6. Plotting
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    
    # Plot standard bounds
    ax.plot(rejection_rates * 100, curve_oracle, color='black', linestyle='--', linewidth=2.5, label=f"Oracle (AURC = {aurc_oracle:.4f})")
    ax.plot(rejection_rates * 100, curve_random, color='gray', linestyle=':', linewidth=2.0, label=f"Random Rejection (AURC = {aurc_random:.4f})")
    
    # Plot baseline methods
    ax.plot(rejection_rates * 100, curve_standard, color='#d62728', linewidth=1.5, alpha=0.8, label=f"Standard Disagreement (AURC = {aurc_std:.4f})")
    ax.plot(rejection_rates * 100, curve_chen, color='#ff7f0e', linewidth=1.5, alpha=0.8, label=f"Chen Paired Variance (AURC = {aurc_chen:.4f})")
    ax.plot(rejection_rates * 100, curve_gmm, color='#2ca02c', linewidth=1.5, alpha=0.8, label=f"Shaker GMM Entropy (AURC = {aurc_gmm:.4f})")
    ax.plot(rejection_rates * 100, curve_credal, color='#1f77b4', linewidth=2.5, label=f"Shaker Likelihood [Credal] (AURC = {aurc_credal:.4f})")
    
    # Plot proximity methods
    ax.plot(rejection_rates * 100, curve_prox_std, color='brown', linestyle='-.', linewidth=1.5, label=f"Standard Proximity (AURC = {aurc_prox_std:.4f})")
    ax.plot(rejection_rates * 100, curve_prox_tns, color='magenta', linestyle='-.', linewidth=1.5, label=f"Method A: TNS (AURC = {aurc_prox_tns:.4f})")
    ax.plot(rejection_rates * 100, curve_prox_twq, color='#17becf', linewidth=2.0, label=f"Method B: TWQ (AURC = {aurc_prox_twq:.4f})")
    ax.plot(rejection_rates * 100, curve_prox_tds, color='olive', linewidth=2.0, label=f"Method C: TDS (AURC = {aurc_prox_tds:.4f})")
    ax.plot(rejection_rates * 100, curve_prox_twq_tds, color='purple', linewidth=2.5, label=f"Method B+C: TWQ+TDS (AURC = {aurc_prox_twq_tds:.4f})")
    
    ax.set_title(f"Error-Rejection Curve Comparison ({ndim}D {func_name.replace('_', '-').upper()} with {gap_type.capitalize()} Gap)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Rejection Rate (%)", fontsize=12)
    ax.set_ylabel("Remaining Prediction Error (MSE)", fontsize=12)
    ax.set_xlim(0, 95)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out_path = f"figures/rejection_curves_comparison_{ndim}d_{gap_type}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"Rejection curves plot saved to: {out_path}")

def main():
    # 1D evaluations
    run_evaluation("empty", 1)
    run_evaluation("sparse", 1)
    # 2D evaluations
    run_evaluation("empty", 2)
    run_evaluation("sparse", 2)

if __name__ == "__main__":
    main()
