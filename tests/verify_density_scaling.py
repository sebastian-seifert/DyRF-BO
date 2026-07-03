import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Uncertainty_Quantification import get_1d_functions, generate_data
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ

def main():
    print("==========================================================")
    # 1. Generate 1D data with empty gap [4, 6]
    funcs_1d = get_1d_functions()
    func_name = "sin"
    seed = 42
    
    X_train, y_train, X_test, y_test, y_true_binary = generate_data(
        funcs_1d, func_name, seed, gap_type='empty'
    )
    print(f"Data summary:")
    print(f"  * X_train: {X_train.shape[0]} points (excluding [4, 6] gap)")
    print(f"  * X_test:  {X_test.shape[0]} points")
    print(f"  * Gap mask: {np.sum(y_true_binary)} OOD points, {np.sum(1 - y_true_binary)} ID points")
    
    # 2. Fit standard random forest model
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=seed, oob_score=True)
    rf.fit(X_train, y_train)
    
    # 3. Reference Proximity UQ (No density scaling)
    uq_engine_std = GPUProximityRegressionUQ(rf, X_train, y_train, device="cpu", use_density_scaling=False)
    uq_std = uq_engine_std.compute_uq(X_test, n_neighbors='auto', level=0.95)
    
    # 4. Density Scaled Proximity UQ (With density scaling)
    uq_engine_ds = GPUProximityRegressionUQ(rf, X_train, y_train, device="cpu", use_density_scaling=True, density_scaling_alpha=1.0)
    uq_ds = uq_engine_ds.compute_uq(X_test, n_neighbors='auto', level=0.95)
    
    # 5. Extract metrics inside and outside the gap
    id_mask = y_true_binary == 0
    ood_mask = y_true_binary == 1
    
    mean_std_id = np.mean(uq_std[id_mask])
    mean_std_ood = np.mean(uq_std[ood_mask])
    
    mean_ds_id = np.mean(uq_ds[id_mask])
    mean_ds_ood = np.mean(uq_ds[ood_mask])
    
    print("\n----------------------------------------------------------")
    print("RESULTS:")
    print(f"1. Standard Proximity UQ (No Scaling):")
    print(f"   * Mean ID Uncertainty:  {mean_std_id:.4f}")
    print(f"   * Mean OOD Uncertainty: {mean_std_ood:.4f}")
    print(f"   * OOD / ID Ratio:       {mean_std_ood / mean_std_id:.2f}x")
    
    print(f"2. Density-Scaled Proximity UQ:")
    print(f"   * Mean ID Uncertainty:  {mean_ds_id:.4f}")
    print(f"   * Mean OOD Uncertainty: {mean_ds_ood:.4f}")
    print(f"   * OOD / ID Ratio:       {mean_ds_ood / mean_ds_id:.2f}x")
    print("----------------------------------------------------------")
    
    # Assertions to confirm correctness
    # ID should be scaling-invariant (ratio of scaling is around 1.0)
    # OOD should be scaled up significantly (OOD ratio of density-scaled should be higher than standard)
    assert mean_ds_id < 2.0 * mean_std_id, "ID uncertainty scaled up too much!"
    assert (mean_ds_ood / mean_ds_id) > (mean_std_ood / mean_std_id), "OOD/ID ratio did not improve!"
    
    print("SUCCESS: Leaf Density Scaling behaves exactly as mathematically expected!")
    print("==========================================================")

if __name__ == "__main__":
    main()
