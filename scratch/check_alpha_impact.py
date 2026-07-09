import json
import os
import numpy as np

def main():
    # Let's inspect the files for rf1_k500_empty_m12_linear
    files = [
        "uncertainty_quantification_results_rf1_k500_empty_m12_linear_20260707_223759.json",
        "uncertainty_quantification_results_rf1_k500_empty_m12_linear_20260707_224820.json",
        "uncertainty_quantification_results_rf1_k500_empty_m12_linear_20260707_230603.json"
    ]
    
    print("Comparing Proximity Method C & B+C NAURC across the three parallel runs:")
    for f in files:
        filepath = os.path.join("results", f)
        with open(filepath, "r") as file:
            d = json.load(file)
        
        c_all = d["results_all"]["Proximity_Method_C"]["naurc"]
        bc_all = d["results_all"]["Proximity_Method_B_C"]["naurc"]
        
        print(f"\nFile: {f}")
        print(f"  Method C NAURC: {np.mean(c_all):.6f} (std={np.std(c_all):.6f})")
        print(f"  Method B+C NAURC: {np.mean(bc_all):.6f} (std={np.std(bc_all):.6f})")

if __name__ == "__main__":
    main()
