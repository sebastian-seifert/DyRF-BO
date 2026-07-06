import json
import numpy as np

def print_auroc_table():
    # Let's read the RF3 empty gap results file (primary representative configuration)
    fpath = "results/uncertainty_quantification_results_rf3_k20_empty_m12_linear_20260705_215428.json"
    
    with open(fpath, "r") as f:
        data = json.load(f)
        
    results_by_dim = data["results_by_dim"]
    approaches = ["Standard", "Chen", "Shaker_GMM_Entropy", "Shaker_Likelihood_GL_Bisect", "Shaker_Likelihood_GL_Newton", "Shaker_Likelihood_Trapz_Bisect"]
    
    dimensions = ["1D", "2D", "3D", "4D", "5D", "6D", "7D", "8D", "9D", "10D"]
    
    print("| Dimension | Standard | Chen | GMM Entropy | Likelihood GL Bisect | Likelihood GL Newton | Likelihood Trapz Bisect |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for dim in dimensions:
        row = f"| **{dim}** |"
        for app in approaches:
            vals = results_by_dim[dim][app]["auroc"]
            mean_val = np.nanmean(vals)
            std_val = np.nanstd(vals)
            row += f" {mean_val:.4f} |"
        print(row)

if __name__ == "__main__":
    print_auroc_table()
