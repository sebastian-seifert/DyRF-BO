import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Uncertainty_Quantification import get_1d_functions, generate_data
from Proximity_Regression_UQ import ProximityRegressionUQ

def main():
    print("Starting Smoke Test for Proximity Regression UQ...")
    
    # 1. Setup 1D "sin" function
    funcs_1d = get_1d_functions()
    func_name = "sin"
    seed = 42
    
    X_train, y_train, X_test, y_test, y_true_binary = generate_data(funcs_1d, func_name, seed)
    print(f"Generated data: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
    
    # 2. Fit standard random forest model
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=seed)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # 3. Apply ProximityRegressionUQ (RF-FIRE)
    uq = ProximityRegressionUQ(rf, X_train, y_train)
    uq.fit()
    
    # Compute prediction intervals directly to plot bounds
    print("Computing proximity prediction intervals (RF-FIRE)...")
    y_pred_lwr, _, y_pred_upr = uq.rfgap_model.predict_with_intervals(
        X_test=X_test,
        n_neighbors='auto',
        level=0.95,
        verbose=False
    )
    
    # 4. Generate visual representation (Prediction intervals & training gap)
    plt.figure(figsize=(10, 6))
    
    # Plot training points
    plt.scatter(X_train, y_train, color='black', s=15, alpha=0.4, label='Training Data')
    
    # Plot true function & RF predictions
    plt.plot(X_test, y_test, color='green', linestyle='--', alpha=0.7, label='True Function')
    plt.plot(X_test, y_pred, color='blue', linewidth=2, label='RF Prediction')
    
    # Shade prediction intervals (RF-FIRE)
    plt.fill_between(
        X_test.ravel(),
        y_pred_lwr,
        y_pred_upr,
        color='purple',
        alpha=0.2,
        label='95% RF-FIRE Interval (Proximity)'
    )
    
    # Highlight exploration gap (4 to 6)
    plt.axvspan(4, 6, color='red', alpha=0.08, label='OOD Gap (No Data)')
    
    plt.title("Localized Uncertainty via Proximities (RF-FIRE) on 1D Sin")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save figure with a new descriptive name
    os.makedirs("figures", exist_ok=True)
    fig_path = "figures/uncertainty_proximity_sin_1d.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    print(f"Smoke test complete! Figure successfully saved to: {fig_path}")

if __name__ == "__main__":
    main()
