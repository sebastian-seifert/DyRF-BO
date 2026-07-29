#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
from Epistemic_Quantifier import EpistemicQuantifier
from data_generator import generate_data

def generate_spatial_comparison_plot():
    # Set clean, publication-ready style parameters
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['grid.color'] = '#e5e5e5'
    plt.rcParams['grid.linewidth'] = 0.6
    
    # 1. Generate 1D sine wave data with an empty gap in [4.0, 6.0]
    func_dict = {
        "sin": {
            "func": lambda x: np.sin(x),
            "gap": [4.0, 6.0],
            "range": [0.0, 10.0]
        }
    }
    
    X_train, y_train, X_test, y_test, y_true_binary = generate_data(
        func_dict, "sin", seed=42, points_per_dim=300, gap_type="empty"
    )
    
    # Sort test set for clean line plotting
    sort_idx = np.argsort(X_test[:, 0])
    X_test_sorted = X_test[sort_idx]
    
    # Fit a standard random forest model
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)
    rf.oob_prediction_ = rf.predict(X_train)
    rf.oob_score = True
    
    # 2. Compute uncertainties under the three approaches
    quantifier = EpistemicQuantifier(rf, X_train, y_train)
    
    # Baseline A: Standard Disagreement (Tree Variance)
    u_std = quantifier.standard_get_epistemic_variance(X_test_sorted)
    
    # Baseline B: Leaf Variance (Aleatoric Variance inside assigned leaves)
    u_leaf = quantifier.base_get_aleatoric_variance(X_test_sorted)
    
    # Our Method: Topological Proximity UQ + DS (Method C)
    uq_model = GPUProximityRegressionUQ(
        rf, X_train, y_train, device="cpu",
        topological_decay_lambda=5.0,
        use_density_scaling=True,
        density_scaling_alpha=1.0
    )
    u_prox = uq_model.compute_uq(X_test_sorted, n_neighbors="auto", level=0.95)
    
    # Normalize for plotting on the same scale [0, 1] for visual clarity
    def normalize(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-10)
        
    u_std_norm = normalize(u_std)
    u_leaf_norm = normalize(u_leaf)
    u_prox_norm = normalize(u_prox)
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot training points
    plt.scatter(X_train[:, 0], y_train, color='#7f8c8d', alpha=0.3, s=15, label="Training Data")
    
    # Highlight the OOD Gap region
    plt.axvspan(4.0, 6.0, color='#e74c3c', alpha=0.1, label="OOD Gap Region [4.0, 6.0]")
    
    # Plot predicted uncertainty curves
    plt.plot(X_test_sorted[:, 0], u_std_norm, color='#3498db', linewidth=2, label="Standard RFGAP (Ensemble Disagreement)")
    plt.plot(X_test_sorted[:, 0], u_leaf_norm, color='#e67e22', linewidth=2, linestyle=':', label="Leaf Variance Baseline (Aleatoric leaf MSE)")
    plt.plot(X_test_sorted[:, 0], u_prox_norm, color='#2ecc71', linewidth=2.5, label="Topological Proximity + DS (Method C)")
    
    plt.title("Spatial Uncertainty Comparison: Behavior Inside the OOD Gap", pad=15, fontsize=12)
    plt.xlabel("Input Space (X)")
    plt.ylabel("Normalized Uncertainty Score [0, 1]")
    plt.xlim(0.0, 10.0)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5', loc='upper right')
    
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, "uncertainty_behavior_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved spatial uncertainty comparison plot to: {save_path}")

if __name__ == "__main__":
    generate_spatial_comparison_plot()
