import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
from data_generator import generate_data

def test_topological_density_scaling():
    print("==================================================")
    print("RUNNING METHOD C (TOPOLOGICAL DENSITY SCALING) TESTS")
    print("==================================================")
    
    # 1. Generate synthetic data
    func_dict = {
        "sin": {
            "func": lambda x: np.sin(x),
            "gap": [4.0, 6.0],
            "range": [0.0, 10.0]
        }
    }
    
    X_train, y_train, X_test, y_test, y_true_binary = generate_data(
        func_dict, "sin", seed=42, points_per_dim=50, gap_type="empty"
    )
    
    rf = RandomForestRegressor(n_estimators=5, min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)
    rf.oob_prediction_ = rf.predict(X_train)
    rf.oob_score = True
    
    # 2. Run Standard Density Scaling UQ
    print("\n[Test 1] Running Standard Density Scaling UQ...")
    uq_standard = GPUProximityRegressionUQ(
        rf, X_train, y_train, device="cpu", use_density_scaling=True, density_scaling_alpha=1.0
    )
    uq_standard_vals = uq_standard.compute_uq(X_test, n_neighbors=5, level=0.95)
    
    # 3. Run Topological Density Scaling UQ with large lambda
    print("[Test 2] Running Method C with large lambda (convergence check)...")
    uq_topo_large = GPUProximityRegressionUQ(
        rf, X_train, y_train, device="cpu", use_density_scaling=True, 
        density_scaling_alpha=1.0, topological_decay_lambda=1e5
    )
    uq_topo_large_vals = uq_topo_large.compute_uq(X_test, n_neighbors=5, level=0.95)
    
    # Assert they converge (discrepancy should be tiny)
    max_diff = np.max(np.abs(uq_standard_vals - uq_topo_large_vals))
    print(f"-> Max discrepancy (Standard Density vs Method C Lambda=1e5): {max_diff:.3e}")
    assert max_diff < 1e-4, f"Topological density scaling at lambda=1e5 does not match standard (diff={max_diff})"
    print("✓ Convergence assertion PASSED!")
    
    # 4. Run Method C with moderate lambda
    print("\n[Test 3] Running Method C with moderate lambda (smoothing check)...")
    uq_topo_smooth = GPUProximityRegressionUQ(
        rf, X_train, y_train, device="cpu", use_density_scaling=True, 
        density_scaling_alpha=1.0, topological_decay_lambda=1.0
    )
    uq_topo_smooth_vals = uq_topo_smooth.compute_uq(X_test, n_neighbors=5, level=0.95)
    
    assert not np.isnan(uq_topo_smooth_vals).all()
    assert np.any(uq_topo_smooth_vals > 0.0)
    
    diff_smooth = np.max(np.abs(uq_standard_vals - uq_topo_smooth_vals))
    print(f"-> Difference (Standard vs Method C Lambda=1.0): {diff_smooth:.3e}")
    assert diff_smooth > 1e-3, f"Topological density UQ at lambda=1.0 is identical to standard density scaling!"
    print("✓ Smoothness assertion PASSED!")
    
    print("\n==================================================")
    print("ALL METHOD C TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    try:
        test_topological_density_scaling()
        sys.exit(0)
    except AssertionError as e:
        print("\n❌ FAILED: Method C assertion failed:", e)
        sys.exit(1)
    except Exception as e:
        print("\n❌ FAILED: Unexpected error:", e)
        sys.exit(1)
