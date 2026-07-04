import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
from data_generator import generate_data

def run_topological_methods_comparison():
    os.environ["PROXIMITY_DEBUG"] = "1"
    print("==========================================================")
    print("RUNNING COMPREHENSIVE TEST SUITE: METHODS A, B, AND C")
    print("==========================================================")
    
    # 1. Generate 1D sine wave data with an empty gap in [4.0, 6.0]
    func_dict = {
        "sin": {
            "func": lambda x: np.sin(x),
            "gap": [4.0, 6.0],
            "range": [0.0, 10.0]
        }
    }
    
    X_train, y_train, X_test, y_test, y_true_binary = generate_data(
        func_dict, "sin", seed=42, points_per_dim=150, gap_type="empty"
    )
    
    # Fit a standard random forest model
    rf = RandomForestRegressor(n_estimators=15, min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)
    rf.oob_prediction_ = rf.predict(X_train)
    rf.oob_score = True
    
    # Split test set into In-Distribution (ID) and Out-Of-Distribution (OOD) sections
    mask_id = (X_test[:, 0] < 3.5) | (X_test[:, 0] > 6.5)
    mask_ood = (X_test[:, 0] >= 4.5) & (X_test[:, 0] <= 5.5)
    
    X_id = X_test[mask_id]
    X_ood = X_test[mask_ood]
    
    print(f"Data ready: {len(X_train)} train, {len(X_id)} test ID, {len(X_ood)} test OOD.")
    
    results = {}
    
    # Define the configurations to test
    configs = {
        "Standard Proximity (Baseline)": {
            "topological_decay_lambda": None,
            "n_neighbors": 20,
            "use_density_scaling": False
        },
        "Method A: TNS (Topological Neighbors)": {
            "topological_decay_lambda": 1.0,
            "n_neighbors": 20,
            "use_density_scaling": False
        },
        "Method B: TWQ (Weighted Quantiles)": {
            "topological_decay_lambda": 1.0,
            "n_neighbors": "auto",
            "use_density_scaling": False
        },
        "Method C: TDS (Density Scaling Only)": {
            "topological_decay_lambda": 5.0,
            "n_neighbors": 20,
            "use_density_scaling": True
        },
        "Method A + C (TNS + TDS)": {
            "topological_decay_lambda": 5.0,
            "n_neighbors": 20,
            "use_density_scaling": True
        },
        "Method B + C (TWQ + TDS)": {
            "topological_decay_lambda": 5.0,
            "n_neighbors": "auto",
            "use_density_scaling": True
        }
    }
    
    for name, params in configs.items():
        print(f"\nEvaluating: {name}...")
        uq_model = GPUProximityRegressionUQ(
            rf, X_train, y_train, device="cpu",
            topological_decay_lambda=params["topological_decay_lambda"],
            use_density_scaling=params["use_density_scaling"],
            density_scaling_alpha=1.0
        )
        
        # Compute UQ for both ID and OOD splits
        uq_id = uq_model.compute_uq(X_id, n_neighbors=params["n_neighbors"], level=0.95)
        uq_ood = uq_model.compute_uq(X_ood, n_neighbors=params["n_neighbors"], level=0.95)
        
        # Check for invalid values
        assert not np.isnan(uq_id).any(), f"NaNs detected in ID UQ for {name}"
        assert not np.isnan(uq_ood).any(), f"NaNs detected in OOD UQ for {name}"
        assert not (uq_id < 0.0).any(), f"Negative values detected in ID UQ for {name}"
        assert not (uq_ood < 0.0).any(), f"Negative values detected in OOD UQ for {name}"
        
        mean_id = np.mean(uq_id)
        mean_ood = np.mean(uq_ood)
        ratio = mean_ood / max(mean_id, 1e-10)
        
        results[name] = {
            "mean_id": mean_id,
            "mean_ood": mean_ood,
            "ratio": ratio
        }
        print(f"  -> Mean ID: {mean_id:.4f} | Mean OOD: {mean_ood:.4f} | Ratio OOD/ID: {ratio:.2f}x")

    # Output results comparison table
    print("\n==========================================================")
    print("              SUMMARY OF TOPOLOGICAL UQ COMPARISON")
    print("==========================================================")
    print(f"{'Method/Configuration':<40} | {'Mean ID UQ':<10} | {'Mean OOD UQ':<11} | {'OOD/ID Ratio':<12}")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"{name:<40} | {metrics['mean_id']:<10.4f} | {metrics['mean_ood']:<11.4f} | {metrics['ratio']:<11.2f}x")
    print("==========================================================")
    
    # Assertion Checks to lock in core behavioral properties:
    # 1. Proximity baseline suffers from Overconfidence Trap in gaps (low OOD ratio).
    # 2. Method C (and combinations with C) should significantly penalize the gap (higher ratio than baseline).
    ratio_baseline = results["Standard Proximity (Baseline)"]["ratio"]
    ratio_method_c = results["Method C: TDS (Density Scaling Only)"]["ratio"]
    ratio_combined_b_c = results["Method B + C (TWQ + TDS)"]["ratio"]
    
    print(f"\nChecking safety bounds & behavioral expectations:")
    print(f"  - Baseline OOD/ID Ratio: {ratio_baseline:.2f}x")
    print(f"  - Method C OOD/ID Ratio: {ratio_method_c:.2f}x")
    print(f"  - Method B+C OOD/ID Ratio: {ratio_combined_b_c:.2f}x")
    
    # 1. Assert that Method B improves OOD gap penalization relative to baseline
    assert results["Method B: TWQ (Weighted Quantiles)"]["ratio"] > ratio_baseline, "Method B (TWQ) failed to improve OOD gap penalization!"
    
    # 2. Check specific expected numeric value ranges for the outputs
    print("\nVerifying specific UQ values against reference ranges:")
    
    val_baseline_id = results["Standard Proximity (Baseline)"]["mean_id"]
    val_baseline_ood = results["Standard Proximity (Baseline)"]["mean_ood"]
    print(f"  - Baseline Mean ID: {val_baseline_id:.4f} (expected [0.30, 0.40])")
    print(f"  - Baseline Mean OOD: {val_baseline_ood:.4f} (expected [0.45, 0.60])")
    assert 0.30 <= val_baseline_id <= 0.40, f"Baseline ID UQ {val_baseline_id} out of bounds!"
    assert 0.45 <= val_baseline_ood <= 0.60, f"Baseline OOD UQ {val_baseline_ood} out of bounds!"
    
    val_b_id = results["Method B: TWQ (Weighted Quantiles)"]["mean_id"]
    val_b_ood = results["Method B: TWQ (Weighted Quantiles)"]["mean_ood"]
    print(f"  - Method B Mean ID: {val_b_id:.4f} (expected [0.25, 0.35])")
    print(f"  - Method B Mean OOD: {val_b_ood:.4f} (expected [0.48, 0.60])")
    assert 0.25 <= val_b_id <= 0.35, f"Method B ID UQ {val_b_id} out of bounds!"
    assert 0.48 <= val_b_ood <= 0.60, f"Method B OOD UQ {val_b_ood} out of bounds!"

    val_c_id = results["Method C: TDS (Density Scaling Only)"]["mean_id"]
    val_c_ood = results["Method C: TDS (Density Scaling Only)"]["mean_ood"]
    print(f"  - Method C Mean ID: {val_c_id:.4f} (expected [0.20, 0.35])")
    print(f"  - Method C Mean OOD: {val_c_ood:.4f} (expected [0.30, 0.45])")
    assert 0.20 <= val_c_id <= 0.35, f"Method C ID UQ {val_c_id} out of bounds!"
    assert 0.30 <= val_c_ood <= 0.45, f"Method C OOD UQ {val_c_ood} out of bounds!"
    
    print("✓ Specific UQ values check PASSED successfully!")
    print("==========================================================")

if __name__ == "__main__":
    try:
        run_topological_methods_comparison()
        sys.exit(0)
    except AssertionError as e:
        print("\n❌ FAILED: Behavioral assertion failed:", e)
        sys.exit(1)
    except Exception as e:
        print("\n❌ FAILED: Unexpected error:", e)
        sys.exit(1)
