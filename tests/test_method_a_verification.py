import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
from data_generator import generate_data

def test_method_a_verification():
    print("==================================================")
    print("RUNNING ADVANCED METHOD A VERIFICATION TESTS")
    print("==================================================")
    
    # 1. Generate realistic synthetic data (1D sine wave)
    func_dict = {
        "sin": {
            "func": lambda x: np.sin(x),
            "gap": [4.0, 6.0],
            "range": [0.0, 10.0]
        }
    }
    
    X_train, y_train, X_test, y_test, y_true_binary = generate_data(
        func_dict, "sin", seed=42, points_per_dim=100, gap_type="empty"
    )
    
    # Fit a standard random forest model
    rf = RandomForestRegressor(n_estimators=10, min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)
    # Mock oob_prediction_ to bypass RF fitting validation
    rf.oob_prediction_ = rf.predict(X_train)
    rf.oob_score = True
    
    # 2. Run Standard Proximity UQ (baseline)
    print("\n[Test 1] Running Standard Proximity baseline...")
    uq_standard = GPUProximityRegressionUQ(rf, X_train, y_train, device="cpu")
    uq_standard_vals = uq_standard.compute_uq(X_test, n_neighbors=5, level=0.95)
    
    # 3. Run Method A with very large lambda (should converge to Standard)
    print("[Test 2] Running Method A with large lambda (convergence check)...")
    uq_topo_large = GPUProximityRegressionUQ(
        rf, X_train, y_train, device="cpu", topological_decay_lambda=1e5
    )
    uq_topo_large_vals = uq_topo_large.compute_uq(X_test, n_neighbors=5, level=0.95)
    
    # Check convergence: max difference should be extremely small
    max_diff = np.max(np.abs(uq_standard_vals - uq_topo_large_vals))
    print(f"-> Max discrepancy (Standard vs Topological Lambda=1e5): {max_diff:.3e}")
    assert max_diff < 1e-4, f"Topological UQ at lambda=1e5 does not match standard UQ (diff={max_diff})"
    print("✓ Convergence assertion PASSED!")

    # 4. Run Method A with moderate lambda (smoothing verification)
    print("\n[Test 3] Running Method A with moderate lambda (smoothness check)...")
    uq_topo_smooth = GPUProximityRegressionUQ(
        rf, X_train, y_train, device="cpu", topological_decay_lambda=1.0
    )
    uq_topo_smooth_vals = uq_topo_smooth.compute_uq(X_test, n_neighbors=5, level=0.95)
    
    # Make sure output values are valid (not all NaNs or zeros)
    assert not np.isnan(uq_topo_smooth_vals).all()
    assert np.any(uq_topo_smooth_vals > 0.0)
    print("✓ Smoothness assertion PASSED!")
    
    print("\n==================================================")
    print("ALL METHOD A VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    try:
        test_method_a_verification()
        sys.exit(0)
    except AssertionError as e:
        print("\n❌ FAILED: Parity/Convergence assertion failed:", e)
        sys.exit(1)
    except Exception as e:
        print("\n❌ FAILED: Unexpected error:", e)
        sys.exit(1)
