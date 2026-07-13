import sys
from synthetic_functions import (
    get_1d_functions, get_2d_functions, get_3d_functions, get_4d_functions,
    get_5d_functions, get_6d_functions, get_7d_functions, get_8d_functions,
    get_9d_functions, get_10d_functions, get_11d_functions, get_12d_functions,
    get_13d_functions, get_14d_functions, get_15d_functions
)

def main():
    funcs = {}
    for g in [
        get_1d_functions, get_2d_functions, get_3d_functions, get_4d_functions,
        get_5d_functions, get_6d_functions, get_7d_functions, get_8d_functions,
        get_9d_functions, get_10d_functions, get_11d_functions, get_12d_functions,
        get_13d_functions, get_14d_functions, get_15d_functions
    ]:
        funcs.update(g())

    BASELINES = "Standard,Chen,Shaker_GMM_Entropy,Shaker_Likelihood_GL_Bisect,Shaker_Likelihood_GL_Newton,Shaker_Likelihood_Trapz_Bisect,Shaker_Likelihood_Trapz_Newton"
    PROXIMITY_METHODS = "Proximity_Baseline,Proximity_Method_B,Proximity_Method_C,Proximity_Method_B_C"

    # Write out parameter strings for SLURM Job Array tasks (locking scaling law to "linear" only)
    with open("unified_sweep_params.txt", "w") as f:
        for func_name in sorted(funcs.keys()):
            for config in [1, 5]:
                # --- empty gap_type ---
                # 1. Baselines
                f.write(f"--function {func_name} --rf_config {config} --gap_type empty --approaches {BASELINES}\n")
                # 2. Proximity runs
                for alpha in [1.0, 5.0]:
                    f.write(f"--function {func_name} --rf_config {config} --gap_type empty --k_neighbors 20 --density_scaling_alpha {alpha} --approaches {PROXIMITY_METHODS}\n")
                
                # --- sparse gap_type ---
                for law in ["linear"]:
                    for mult in [5, 50]:
                        # 1. Baselines
                        f.write(f"--function {func_name} --rf_config {config} --gap_type sparse --scaling_law {law} --sparse_multiplier {mult} --approaches {BASELINES}\n")
                        # 2. Proximity runs
                        for alpha in [1.0, 5.0]:
                            f.write(f"--function {func_name} --rf_config {config} --gap_type sparse --scaling_law {law} --sparse_multiplier {mult} --k_neighbors 20 --density_scaling_alpha {alpha} --approaches {PROXIMITY_METHODS}\n")

    # 41 functions * 2 configs * (1 + 2 + 2*(1 + 2)) = 82 * (3 + 6) = 738 runs
    print(f"Generated unified_sweep_params.txt with {len(funcs) * 2 * 9} execution configurations (linear scaling only).")

if __name__ == "__main__":
    main()
