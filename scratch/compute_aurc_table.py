import os
import sys
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator import generate_data
from Epistemic_Quantifier import EpistemicQuantifier
from Credal_Regression_UQ import CredalRegressionUQ
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
from metrics import calculate_rejection_curve, calculate_aurc
from sklearn.ensemble import RandomForestRegressor
from synthetic_functions import (
    get_1d_functions,
    get_2d_functions,
    get_3d_functions,
    get_5d_functions
)

def main():
    funcs_1d = get_1d_functions()
    funcs_2d = get_2d_functions()
    funcs_3d = get_3d_functions()
    funcs_5d = get_5d_functions()
    
    # Selected functions
    selected = [
        ("1D", "sin", funcs_1d),
        ("2D", "rastrigin", funcs_2d),
        ("3D", "ackley", funcs_3d),
        ("5D", "rosenbrock", funcs_5d)
    ]
    
    rejection_rates = np.linspace(0.0, 0.95, 96)
    
    print("\n| Dimension | Function | Standard | Chen | GMM | Credal | Std Prox | TWQ (Method B) |")
    print("|-----------|----------|----------|------|-----|--------|----------|----------------|")
    
    for dim, name, func_dict in selected:
        # Generate data
        X_train, y_train, X_test, y_test, y_true_binary = generate_data(
            func_dict, name, seed=42, points_per_dim=150, gap_type="empty", min_samples_leaf=5
        )
        
        # Fit Random Forest
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, oob_score=True, random_state=42)
        rf.fit(X_train, y_train)
        rf.oob_prediction_ = rf.predict(X_train)
        
        y_pred = rf.predict(X_test)
        
        # Instantiate quantifiers
        quantifier = EpistemicQuantifier(rf, X_train, y_train)
        credal = CredalRegressionUQ(rf, X_train, y_train)
        prox_std = GPUProximityRegressionUQ(rf, X_train, y_train, device="cpu", topological_decay_lambda=None)
        prox_twq = GPUProximityRegressionUQ(rf, X_train, y_train, device="cpu", topological_decay_lambda=1.0)
        
        # Compute uncertainties
        u_std = quantifier.standard_get_epistemic_variance(X_test)
        u_chen = quantifier.chen_get_epistemic_variance(X_test)
        u_gmm = quantifier.shaker_get_epistemic_variance(X_test, random_state=42)
        u_credal, _ = credal.compute_uq(X_test, backend="cpu", integration_method="gauss_legendre", sup_solver="bisection")
        u_prox_std = prox_std.compute_uq(X_test, n_neighbors=20, level=0.95)
        u_prox_twq = prox_twq.compute_uq(X_test, n_neighbors="auto", level=0.95)
        
        # Compute AURCs
        aurc_std = calculate_aurc(rejection_rates, calculate_rejection_curve(u_std, y_pred, y_test, rejection_rates))
        aurc_chen = calculate_aurc(rejection_rates, calculate_rejection_curve(u_chen, y_pred, y_test, rejection_rates))
        aurc_gmm = calculate_aurc(rejection_rates, calculate_rejection_curve(u_gmm, y_pred, y_test, rejection_rates))
        aurc_credal = calculate_aurc(rejection_rates, calculate_rejection_curve(u_credal, y_pred, y_test, rejection_rates))
        aurc_prox_std = calculate_aurc(rejection_rates, calculate_rejection_curve(u_prox_std, y_pred, y_test, rejection_rates))
        aurc_prox_twq = calculate_aurc(rejection_rates, calculate_rejection_curve(u_prox_twq, y_pred, y_test, rejection_rates))
        
        print(f"| {dim} | {name} | {aurc_std:.5f} | {aurc_chen:.5f} | {aurc_gmm:.5f} | {aurc_credal:.5f} | {aurc_prox_std:.5f} | {aurc_prox_twq:.5f} |")

if __name__ == "__main__":
    main()
