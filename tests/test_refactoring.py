import os
import sys
import json
import time
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Uncertainty_Quantification import run_single_test
from synthetic_functions import get_1d_functions, get_2d_functions

BASELINE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refactoring_baseline.json")

def gather_test_results():
    funcs_1d = get_1d_functions()
    funcs_2d = get_2d_functions()
    
    approaches = ["Standard", "Proximity"]
    
    # Run simple 1D and 2D test configurations
    res_1d, _ = run_single_test(
        func_dict=funcs_1d,
        func_name="sin",
        seed=42,
        approaches=approaches,
        rf_config=1,
        k_neighbors=20,
        gap_type="empty",
        use_density_scaling=True,
        density_scaling_alpha=1.0
    )
    
    res_2d, _ = run_single_test(
        func_dict=funcs_2d,
        func_name="sin_cos",
        seed=42,
        approaches=approaches,
        rf_config=1,
        k_neighbors=20,
        gap_type="sparse",
        sparse_multiplier=5,
        scaling_law="linear",
        use_density_scaling=False
    )
    
    def extract_stats(res):
        out = {}
        for app in approaches:
            out[app] = {
                "auroc": float(np.mean(res[app]["auroc"])),
                "fpr95": float(np.mean(res[app]["fpr95"])),
                "aupr": float(np.mean(res[app]["aupr"])),
                "brier": float(np.mean(res[app]["brier"])),
                "spearman": float(np.mean(res[app]["spearman"])),
                "mi": float(np.mean(res[app]["mi"])),
                "jsd": float(np.mean(res[app]["jsd"])),
                "naurc": float(np.mean(res[app]["naurc"]))
            }
        return out

    return {
        "1D_sin_empty_ds": extract_stats(res_1d),
        "2D_sincos_sparse": extract_stats(res_2d)
    }

def main():
    generate_mode = "--generate" in sys.argv
    
    print("Running regression UQ runs...")
    t_start = time.perf_counter()
    current_results = gather_test_results()
    t_end = time.perf_counter()
    print(f"Regression UQ runs completed in {t_end - t_start:.4f} seconds.")
    
    if generate_mode:
        with open(BASELINE_FILE, "w", encoding="utf-8") as f:
            json.dump(current_results, f, indent=2)
        print(f"✓ Baseline successfully generated and saved to: {BASELINE_FILE}")
        sys.exit(0)
        
    if not os.path.exists(BASELINE_FILE):
        print(f"Error: Baseline file '{BASELINE_FILE}' not found. Run with --generate first.")
        sys.exit(1)
        
    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        baseline_results = json.load(f)
        
    # Verify results match exactly
    mismatches = 0
    tolerance = 1e-6
    
    for run_key in baseline_results:
        for app in baseline_results[run_key]:
            for metric in baseline_results[run_key][app]:
                base_val = baseline_results[run_key][app][metric]
                curr_val = current_results[run_key][app][metric]
                
                diff = abs(base_val - curr_val)
                if diff > tolerance:
                    print(f"❌ MISMATCH [{run_key}][{app}][{metric}]: baseline={base_val:.6f}, current={curr_val:.6f} (diff={diff:.6e})")
                    mismatches += 1
                else:
                    print(f"✓ MATCH [{run_key}][{app}][{metric}]: {curr_val:.6f}")
                    
    # Micro-benchmark for Credal UQ and Epistemic Quantifier under load
    print("\n==================================================")
    print("RUNNING PERFORMANCE BENCHMARKS UNDER LOAD")
    print("==================================================")
    from sklearn.ensemble import RandomForestRegressor
    from Credal_Regression_UQ import CredalRegressionUQ
    from Epistemic_Quantifier import EpistemicQuantifier
    
    np.random.seed(42)
    X_tr = np.random.uniform(0, 10, size=(1000, 5))
    y_tr = np.sin(X_tr[:, 0]) + np.cos(X_tr[:, 1]) + np.random.normal(0, 0.1, size=1000)
    X_te = np.random.uniform(0, 10, size=(500, 5))
    
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    rf.fit(X_tr, y_tr)
    rf.oob_prediction_ = rf.predict(X_tr)
    rf.oob_score = True
    
    # Benchmark Credal GL Bisect
    credal = CredalRegressionUQ(rf, X_tr, y_tr)
    t0 = time.perf_counter()
    credal.compute_uq(X_te, backend="cpu", integration_method="gauss_legendre", sup_solver="bisection")
    t1 = time.perf_counter()
    credal_bisect_time = t1 - t0
    print(f"Credal GL Bisect UQ (CPU) completed in: {credal_bisect_time:.4f} seconds")
    
    # Benchmark Epistemic Quantifier Shaker CPU
    eq = EpistemicQuantifier(rf, X_tr, y_tr)
    t0 = time.perf_counter()
    eq.shaker_get_epistemic_variance(X_te, num_samples=1000, backend="cpu", random_state=42)
    t1 = time.perf_counter()
    shaker_cpu_time = t1 - t0
    print(f"Epistemic GMM Shaker (CPU) completed in: {shaker_cpu_time:.4f} seconds")
    
    if mismatches > 0:
        print(f"\n❌ Refactoring regression failed with {mismatches} mismatch(es)!")
        sys.exit(1)
    else:
        print("\n🎉 SUCCESS: All metric calculations align perfectly with the baseline!")
        sys.exit(0)

if __name__ == "__main__":
    main()
