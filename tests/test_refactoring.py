import os
import sys
import json
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
                "brier": float(np.mean(res[app]["brier"])),
                "spearman": float(np.mean(res[app]["spearman"])),
                "mi": float(np.mean(res[app]["mi"])),
                "jsd": float(np.mean(res[app]["jsd"]))
            }
        return out

    return {
        "1D_sin_empty_ds": extract_stats(res_1d),
        "2D_sincos_sparse": extract_stats(res_2d)
    }

def main():
    generate_mode = "--generate" in sys.argv
    
    print("Running regression UQ runs...")
    current_results = gather_test_results()
    
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
                    
    if mismatches > 0:
        print(f"\n❌ Refactoring regression failed with {mismatches} mismatch(es)!")
        sys.exit(1)
    else:
        print("\n🎉 SUCCESS: All metric calculations align perfectly with the baseline!")
        sys.exit(0)

if __name__ == "__main__":
    main()
