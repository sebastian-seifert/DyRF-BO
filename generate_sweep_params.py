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

    # Write out parameter strings for SLURM Job Array tasks
    with open("sweep_params.txt", "w") as f:
        for func_name in sorted(funcs.keys()):
            for config in [1, 5]:
                # Empty gap type (no sparse multiplier or scaling law required)
                f.write(f"--function {func_name} --rf_config {config} --gap_type empty\n")
                # Sparse gap type
                for law in ["linear", "leaf"]:
                    for mult in [5, 50]:
                        f.write(f"--function {func_name} --rf_config {config} --gap_type sparse --scaling_law {law} --sparse_multiplier {mult}\n")

    print(f"Generated sweep_params.txt with {len(funcs) * 2 * 5} execution configurations.")

if __name__ == "__main__":
    main()
