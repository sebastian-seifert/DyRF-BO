#!/usr/bin/env python3
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_functions import (
    get_1d_functions, get_2d_functions, get_3d_functions, get_4d_functions,
    get_5d_functions, get_6d_functions, get_7d_functions, get_8d_functions,
    get_9d_functions, get_10d_functions, get_11d_functions, get_12d_functions,
    get_13d_functions, get_14d_functions, get_15d_functions, get_branin_hartmann_functions
)

def main():
    funcs = {}
    for g in [
        get_1d_functions, get_2d_functions, get_3d_functions, get_4d_functions,
        get_5d_functions, get_6d_functions, get_7d_functions, get_8d_functions,
        get_9d_functions, get_10d_functions, get_11d_functions, get_12d_functions,
        get_13d_functions, get_14d_functions, get_15d_functions, get_branin_hartmann_functions
    ]:
        funcs.update(g())

    output_dir = "results/sweep_5_manifold"
    os.makedirs(output_dir, exist_ok=True)
    tasks_file = os.path.join(output_dir, "tasks.txt")
    metadata_file = os.path.join(output_dir, "metadata.json")

    BASELINES = "Standard,Chen,Shaker_GMM_Entropy,Shaker_Likelihood_GL_Bisect,Shaker_Likelihood_GL_Newton,Shaker_Likelihood_Trapz_Bisect,Shaker_Likelihood_Trapz_Newton"

    seeds = list(range(1, 8))
    rf_configs = ["A", "B", "C"]

    lambdas = [0.5, 1.0, 5.0]
    ks = [10, 20, 50]
    alphas = [1.0, 5.0]

    lines = []

    for func_name in sorted(funcs.keys()):
        for cfg in rf_configs:
            for seed in seeds:
                # 1. Baselines
                line_base = (
                    f"--function {func_name} --rf_config {cfg} --seed {seed} --gap_type empty "
                    f"--ood_type manifold --approaches {BASELINES} --output_dir {output_dir}/raw"
                )
                lines.append(line_base)

                # 2. Proximity Baseline & Proximity Method B
                for lmbda in lambdas:
                    for k in ks:
                        line_prox_base = (
                            f"--function {func_name} --rf_config {cfg} --seed {seed} --gap_type empty "
                            f"--ood_type manifold --topological_decay_lambda {lmbda} --k_neighbors {k} "
                            f"--approaches Proximity_Baseline --output_dir {output_dir}/raw"
                        )
                        lines.append(line_prox_base)
                    
                    line_prox_b = (
                        f"--function {func_name} --rf_config {cfg} --seed {seed} --gap_type empty "
                        f"--ood_type manifold --topological_decay_lambda {lmbda} --k_neighbors auto "
                        f"--approaches Proximity_Method_B --output_dir {output_dir}/raw"
                    )
                    lines.append(line_prox_b)

                # 3. Proximity Method C & Proximity Method B_C
                for lmbda in lambdas:
                    for alpha in alphas:
                        for k in ks:
                            line_prox_c = (
                                f"--function {func_name} --rf_config {cfg} --seed {seed} --gap_type empty "
                                f"--ood_type manifold --topological_decay_lambda {lmbda} --k_neighbors {k} "
                                f"--use_density_scaling --density_scaling_alpha {alpha} "
                                f"--approaches Proximity_Method_C --output_dir {output_dir}/raw"
                            )
                            lines.append(line_prox_c)
                        
                        line_prox_bc = (
                            f"--function {func_name} --rf_config {cfg} --seed {seed} --gap_type empty "
                            f"--ood_type manifold --topological_decay_lambda {lmbda} --k_neighbors auto "
                            f"--use_density_scaling --density_scaling_alpha {alpha} "
                            f"--approaches Proximity_Method_B_C --output_dir {output_dir}/raw"
                        )
                        lines.append(line_prox_bc)

    with open(tasks_file, "w") as f:
        for line in lines:
            f.write(f"{line}\n")

    metadata = {
        "sweep_name": "sweep_5_manifold",
        "description": "Manifold OOD Data Generation UQ Evaluation across 44 functions, 7 seeds, 3 RF configs (A, B, C)",
        "functions_count": len(funcs),
        "seeds": seeds,
        "rf_configs": rf_configs,
        "total_tasks": len(lines)
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {len(lines)} array task lines in {tasks_file}")

if __name__ == "__main__":
    main()
