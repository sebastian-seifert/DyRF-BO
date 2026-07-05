import os
import sys
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Epistemic_Quantifier import EpistemicQuantifier
from data_generator import generate_data

def run_shaker_test_suite():
    print("==========================================================")
    print("RUNNING EXTENSIVE SHAKER VECTORIZED UQ TEST SUITE")
    print("==========================================================")
    
    # 1. Generate synthetic 2D data
    func_dict = {
        "sin_cos": {
            "func": lambda x, y: np.sin(x) + np.cos(y),
            "gap": [3.0, 7.0],
            "range": [0.0, 10.0]
        }
    }
    
    X_train, y_train, X_test, y_test, y_true_binary = generate_data(
        func_dict, "sin_cos", seed=42, points_per_dim=25, gap_type="empty"
    )
    
    print(f"Generated 2D test data: {len(X_train)} training points, {len(X_test)} query points.")
    
    # Fit a standard random forest model
    rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)
    
    quantifier = EpistemicQuantifier(rf, X_train, y_train)
    
    # 2. Check if GPU backend is available
    gpu_available = quantifier._mc_is_cupy_available()
    print(f"GPU Backend Available (CuPy): {gpu_available}")
    
    # 3. Test Vectorized CPU Execution
    print("\n[CPU Test] Running vectorized CPU Shaker...")
    t0 = time.perf_counter()
    u_cpu = quantifier.shaker_get_epistemic_variance(X_test, num_samples=1000, batch_size="auto", random_state=42, backend="cpu")
    t1 = time.perf_counter()
    cpu_time = t1 - t0
    print(f"✓ CPU Shaker completed in {cpu_time:.4f} seconds.")
    print(f"  Uncertainty summary: mean={np.mean(u_cpu):.4f}, min={np.min(u_cpu):.4f}, max={np.max(u_cpu):.4f}")
    
    # Check that outputs are sound
    assert len(u_cpu) == len(X_test), "CPU output length mismatch"
    assert not np.isnan(u_cpu).any(), "NaNs detected in CPU output"
    assert not (u_cpu < 0.0).any(), "Negative values detected in CPU output"
    
    # 4. Test Vectorized GPU Execution if available
    if gpu_available:
        print("\n[GPU Test] Running vectorized GPU Shaker...")
        t0 = time.perf_counter()
        u_gpu = quantifier.shaker_get_epistemic_variance(X_test, num_samples=1000, batch_size="auto", random_state=42, backend="gpu")
        t1 = time.perf_counter()
        gpu_time = t1 - t0
        print(f"✓ GPU Shaker completed in {gpu_time:.4f} seconds.")
        print(f"  Uncertainty summary: mean={np.mean(u_gpu):.4f}, min={np.min(u_gpu):.4f}, max={np.max(u_gpu):.4f}")
        
        # Parity Check
        # Since both use Monte Carlo sampling with different RNG implementations (NumPy vs CuPy),
        # they will not be identical down to the last float, but their statistics should be very close.
        mean_diff = np.abs(np.mean(u_cpu) - np.mean(u_gpu))
        print(f"\n[Parity Check] Mean difference between CPU and GPU: {mean_diff:.5f}")
        
        # Verify correlation of predictions
        corr = np.corrcoef(u_cpu, u_gpu)[0, 1]
        print(f"[Parity Check] Correlation coefficient between CPU and GPU: {corr:.4f}")
        
        assert corr > 0.90, f"Correlation between CPU and GPU is too low: {corr:.4f}"
        print("✓ CPU/GPU statistical parity test passed!")
        
        # Speedup comparison
        speedup = cpu_time / gpu_time
        print(f"[Performance] GPU Speedup: {speedup:.2f}x")
    else:
        print("\n[GPU Test] Skipped (CuPy not available).")
        
    print("\n==========================================================")
    print("ALL SHAKER VECTORIZED UQ TESTS PASSED SUCCESSFULLY!")
    print("==========================================================")

if __name__ == "__main__":
    run_shaker_test_suite()
