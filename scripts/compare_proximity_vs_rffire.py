import json
import numpy as np

def compare():
    # File paths for the two corresponding configs:
    # 1. RF-FIRE: RF=1, K=20, gap=sparse (m=5, linear), lambda=0.0, ds=False
    fire_file = "results/uncertainty_quantification_results_rf1_k20_sparse_m5_linear_20260704_235036.json"
    # 2. Method C (Topological + DS): RF=1, K=auto, gap=sparse (m=5, linear), lambda=5.0, ds=True
    method_c_file = "results/density_scaling/uncertainty_quantification_results_rf1_kauto_sparse_m5_linear_lambda5.0_ds_20260705_000346.json"

    with open(fire_file, "r") as f:
        data_fire = json.load(f)
    with open(method_c_file, "r") as f:
        data_c = json.load(f)

    dims = ["1D", "2D", "3D", "4D", "5D", "6D", "7D", "8D", "9D", "10D"]

    print("==========================================================================================")
    # Print dimensional comparison header
    print(f"{'Dimension':<10} | {'RF-FIRE AUROC':<15} | {'Method C AUROC':<15} | {'Diff AUROC':<12} | {'RF-FIRE Brier':<15} | {'Method C Brier':<15} | {'Diff Brier':<12}")
    print("------------------------------------------------------------------------------------------")

    win_auroc = 0
    win_brier = 0

    for dim in dims:
        # Extract AUROC
        fire_auroc = np.nanmean(data_fire["results_by_dim"][dim]["Proximity"]["auroc"])
        c_auroc = np.nanmean(data_c["results_by_dim"][dim]["Proximity"]["auroc"])
        diff_auroc = c_auroc - fire_auroc
        
        # Extract Brier
        fire_brier = np.nanmean(data_fire["results_by_dim"][dim]["Proximity"]["brier"])
        c_brier = np.nanmean(data_c["results_by_dim"][dim]["Proximity"]["brier"])
        diff_brier = c_brier - fire_brier

        # Determine wins
        sign_auroc = "+" if diff_auroc >= 0 else ""
        sign_brier = "+" if diff_brier >= 0 else ""
        
        if c_auroc > fire_auroc:
            win_auroc += 1
        if c_brier < fire_brier:
            win_brier += 1

        print(f"{dim:<10} | {fire_auroc:>15.4f} | {c_auroc:>15.4f} | {sign_auroc}{diff_auroc:>11.4f} | {fire_brier:>15.4f} | {c_brier:>15.4f} | {sign_brier}{diff_brier:>11.4f}")

    print("==========================================================================================")
    print(f"SUMMARY: Method C wins in {win_auroc}/10 dimensions for AUROC.")
    print(f"SUMMARY: Method C wins (lower error) in {win_brier}/10 dimensions for Brier Score.")
    print("==========================================================================================")

if __name__ == "__main__":
    compare()
