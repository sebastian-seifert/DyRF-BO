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

    # Write out parameter strings for SLURM Job Array tasks for hybrid sweep
    with open("hybrid_sweep_params.txt", "w") as f:
        for func_name in sorted(funcs.keys()):
            for config in [1, 5]:
                # Empty gap type
                f.write(f"--function {func_name} --rf_config {config} --gap_type empty\n")
                # Sparse gap type (linear only, mult=12)
                f.write(f"--function {func_name} --rf_config {config} --gap_type sparse --scaling_law linear --sparse_multiplier 12\n")

    # 41 functions * 2 configs * 2 gap scenarios = 164 runs
    print(f"Generated hybrid_sweep_params.txt with {len(funcs) * 2 * 2} execution configurations.")

if __name__ == "__main__":
    main()
