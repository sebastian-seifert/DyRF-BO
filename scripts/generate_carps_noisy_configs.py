"""Generates CARP-S task YAML configuration files for BBOB-Noisy and hetGP suites."""

import os
import sys
import yaml
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ConfigSpace.read_and_write import json as cs_json
from noisy_benchmarks.registry import NoisyBenchmarkRegistry
from noisy_benchmarks.base import NoisyBenchmarkProblem


TASKS_TO_GENERATE = [
    # hetGP Suite
    ("hetgp_yuan_wahba_1d", "hetgp", "yuan_wahba_1d", 1),
    ("hetgp_branin_2d", "hetgp", "branin_2d", 2),
    ("hetgp_goldstein_price_2d", "hetgp", "goldstein_price_2d", 2),
    ("hetgp_sinusoid_2d", "hetgp", "sinusoid_2d", 2),
    ("hetgp_sinusoid_4d", "hetgp", "sinusoid_4d", 4),
    # BBOB-Noisy Suite
    ("bbob_noisy_sphere_2d_gaussian", "bbob", "sphere_2d_gaussian", 2),
    ("bbob_noisy_sphere_2d_cauchy", "bbob", "sphere_2d_cauchy", 2),
    ("bbob_noisy_sphere_2d_uniform", "bbob", "sphere_2d_uniform", 2),
    ("bbob_noisy_rosenbrock_2d_gaussian", "bbob", "rosenbrock_2d_gaussian", 2),
    ("bbob_noisy_rosenbrock_2d_cauchy", "bbob", "rosenbrock_2d_cauchy", 2),
    ("bbob_noisy_rastrigin_2d_gaussian", "bbob", "rastrigin_2d_gaussian", 2),
    ("bbob_noisy_bent_cigar_2d_gaussian", "bbob", "bent_cigar_2d_gaussian", 2),
    ("bbob_noisy_attractive_sector_2d_gaussian", "bbob", "attractive_sector_2d_gaussian", 2),
    ("bbob_noisy_schwefel_2d_gaussian", "bbob", "schwefel_2d_gaussian", 2),
    ("bbob_noisy_sphere_4d_gaussian", "bbob", "sphere_4d_gaussian", 4),
    ("bbob_noisy_rosenbrock_4d_gaussian", "bbob", "rosenbrock_4d_gaussian", 4),
]

def generate_task_yaml(problem_key: str, suite_folder: str, task_name: str, dim: int, base_dir: str):
    problem: NoisyBenchmarkProblem = NoisyBenchmarkRegistry.get_problem(problem_key, seed=0)
    cs = problem.configspace
    cs_dict = yaml.safe_load(cs_json.write(cs))

    task_yaml_content = {
        "task_type": "blackbox",
        "benchmark_id": f"Noisy_{suite_folder.upper()}",
        "task_id": f"Noisy/{suite_folder}/{task_name}",
        "task": {
            "_target_": "carps.utils.task.Task",
            "name": f"Noisy/{suite_folder}/{task_name}",
            "seed": "${seed}",
            "objective_function": {
                "_target_": "carps_integration.noisy_objective.CARPSNoisyObjectiveFunction",
                "problem_name": problem_key,
                "seed": "${seed}",
            },
            "input_space": {
                "_target_": "carps.utils.task.InputSpace",
                "configuration_space": {
                    "_target_": "ConfigSpace.configuration_space.ConfigurationSpace.from_serialized_dict",
                    "_convert_": "object",
                    "d": cs_dict,
                },
                "fidelity_space": {
                    "_target_": "carps.utils.task.FidelitySpace",
                    "is_multifidelity": False,
                    "fidelity_type": None,
                    "min_fidelity": None,
                    "max_fidelity": None,
                },
                "instance_space": None,
            },
            "output_space": {
                "_target_": "carps.utils.task.OutputSpace",
                "n_objectives": 1,
                "objectives": ["quality"],
            },
            "optimization_resources": {
                "_target_": "carps.utils.task.OptimizationResources",
                "n_trials": 50,
                "time_budget": None,
                "n_workers": 1,
            },
            "metadata": {
                "_target_": "carps.utils.task.TaskMetadata",
                "has_constraints": False,
                "domain": "synthetic",
                "objective_function_approximation": "real",
                "has_virtual_time": False,
                "deterministic": False,
                "dimensions": dim,
                "search_space_n_categoricals": 0,
                "search_space_n_ordinals": 0,
                "search_space_n_integers": 0,
                "search_space_n_floats": dim,
                "search_space_has_conditionals": False,
                "search_space_has_forbiddens": False,
                "search_space_has_priors": False,
            }
        }
    }

    out_folder = os.path.join(base_dir, suite_folder)
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"cfg_{task_name}.yaml")

    with open(out_path, "w") as f:
        f.write("# @package _global_\n")
        yaml.dump(task_yaml_content, f, sort_keys=False)

    print(f"Generated: {out_path}")

if __name__ == "__main__":
    base_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "carps_integration", "configs", "task", "Noisy")
    for prob_key, suite_dir, t_name, d in TASKS_TO_GENERATE:
        generate_task_yaml(prob_key, suite_dir, t_name, d, base_out)
    print(f"Successfully generated all {len(TASKS_TO_GENERATE)} CARP-S noisy task configs in {base_out}!")
