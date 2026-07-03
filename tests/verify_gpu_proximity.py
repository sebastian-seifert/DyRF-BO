import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Proximity_Regression_UQ import ProximityRegressionUQ
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ

def main():
    print("Running verification and comparison test...")
    
    # 1. Generate synthetic data (100 train points, 20 test points)
    np.random.seed(42)
    X_train = np.random.uniform(0, 10, size=(100, 2))
    y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) + np.random.normal(0, 0.1, size=100)
    X_test = np.random.uniform(0, 10, size=(20, 2))
    
    # 2. Fit standard random forest model with oob_score=True
    rf = RandomForestRegressor(n_estimators=10, min_samples_leaf=3, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    
    # 3. Compute Reference Proximity UQ (CPU-based RFGAP)
    print("Computing Reference Proximity UQ (CPU RFGAP)...")
    ref_uq_engine = ProximityRegressionUQ(rf, X_train, y_train)
    ref_uq_engine.fit()
    ref_uq_auto = ref_uq_engine.compute_uq(X_test, n_neighbors='auto', level=0.95)
    ref_uq_k = ref_uq_engine.compute_uq(X_test, n_neighbors=5, level=0.95)
    
    # 4. Compute New Vectorized Proximity UQ (device='cpu')
    print("Computing New Vectorized Proximity UQ (CPU backend)...")
    gpu_uq_engine_cpu = GPUProximityRegressionUQ(rf, X_train, y_train, device="cpu", batch_size=5)
    gpu_uq_engine_cpu.fit()
    gpu_uq_auto_cpu = gpu_uq_engine_cpu.compute_uq(X_test, n_neighbors='auto', level=0.95)
    gpu_uq_k_cpu = gpu_uq_engine_cpu.compute_uq(X_test, n_neighbors=5, level=0.95)
    
    # Calculate differences
    diff_auto_cpu = np.max(np.abs(ref_uq_auto - gpu_uq_auto_cpu))
    diff_k_cpu = np.max(np.abs(ref_uq_k - gpu_uq_k_cpu))
    print(f"Max diff CPU vs Ref (n_neighbors='auto'): {diff_auto_cpu:.6e}")
    print(f"Max diff CPU vs Ref (n_neighbors=5): {diff_k_cpu:.6e}")
    
    # Assert correctness
    assert diff_auto_cpu < 1e-6, f"Auto UQ differs by {diff_auto_cpu}"
    assert diff_k_cpu < 1e-6, f"k UQ differs by {diff_k_cpu}"
    print("SUCCESS: CPU backend matches Ref UQ exactly!")
    
    # 5. Compute New Vectorized Proximity UQ (device='gpu' if available)
    try:
        import cupy as cp
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count > 0:
            print("GPU detected! Computing GPU Proximity UQ on GPU backend...")
            gpu_uq_engine_gpu = GPUProximityRegressionUQ(rf, X_train, y_train, device="gpu", batch_size=5)
            gpu_uq_engine_gpu.fit()
            gpu_uq_auto_gpu = gpu_uq_engine_gpu.compute_uq(X_test, n_neighbors='auto', level=0.95)
            gpu_uq_k_gpu = gpu_uq_engine_gpu.compute_uq(X_test, n_neighbors=5, level=0.95)
            
            diff_auto_gpu = np.max(np.abs(ref_uq_auto - gpu_uq_auto_gpu))
            diff_k_gpu = np.max(np.abs(ref_uq_k - gpu_uq_k_gpu))
            print(f"Max diff GPU vs Ref (n_neighbors='auto'): {diff_auto_gpu:.6e}")
            print(f"Max diff GPU vs Ref (n_neighbors=5): {diff_k_gpu:.6e}")
            
            assert diff_auto_gpu < 1e-6, f"Auto UQ differs on GPU by {diff_auto_gpu}"
            assert diff_k_gpu < 1e-6, f"k UQ differs on GPU by {diff_k_gpu}"
            print("SUCCESS: GPU backend matches Ref UQ exactly!")
        else:
            print("No GPUs detected. Skipping GPU backend test.")
    except ImportError:
        print("CuPy not installed. Skipping GPU backend test.")

if __name__ == "__main__":
    main()
