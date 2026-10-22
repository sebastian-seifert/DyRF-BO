#!/usr/bin/env python3
"""
Generator for BBOB 128-Dimensional CARP-S Task Configurations.

Generates 72 YAML configuration files (24 functions x 3 instances) for dimension 128
and stores them in `carps_integration/configs/task/BBOB/cfg_128_{fid}_{inst}.yaml`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "carps_integration" / "configs" / "task" / "BBOB"


def generate_single_dim128_yaml(fid: int, inst: int, dimension: int = 128, n_trials: int = 100) -> str:
    """Constructs the YAML string for a BBOB task with given dimension, fid, and instance."""
    hp_lines: List[str] = []
    for i in range(dimension):
        hp_lines.append(f"""        - type: uniform_float
          name: x{i}
          lower: -5.0
          upper: 5.0
          default_value: 0.0
          log: false
          meta: null""")
    hyperparameters_str = "\n".join(hp_lines)

    content = f"""# @package _global_
benchmark_id: BBOB
task_id: ${{task.name}}
task:
  _target_: carps.utils.task.Task
  name: bbob/{dimension}/{fid}/{inst}
  seed: ${{seed}}
  objective_function:
    _target_: carps.objective_functions.bbob.BBOBObjectiveFunction
    dimension: {dimension}
    fid: {fid}
    instance: {inst}
    seed: ${{seed}}
  input_space:
    _target_: carps.utils.task.InputSpace
    configuration_space:
      _target_: ConfigSpace.configuration_space.ConfigurationSpace.from_serialized_dict
      _convert_: object
      d:
        name: null
        hyperparameters:
{hyperparameters_str}
        conditions: []
        forbiddens: []
        python_module_version: 1.2.0
        format_version: 0.4
    fidelity_space:
      _target_: carps.utils.task.FidelitySpace
      is_multifidelity: false
      fidelity_type: null
      min_fidelity: null
      max_fidelity: null
    instance_space: null
  output_space:
    _target_: carps.utils.task.OutputSpace
    n_objectives: 1
    objectives:
    - quality
  optimization_resources:
    _target_: carps.utils.task.OptimizationResources
    n_trials: {n_trials}
    time_budget: null
    n_workers: 1
  metadata:
    _target_: carps.utils.task.TaskMetadata
    dimensions: {dimension}
    has_constraints: false
    domain: synthetic
    objective_function_approximation: real
    has_virtual_time: false
    deterministic: true
    search_space_n_categoricals: 0
    search_space_n_ordinals: 0
    search_space_n_integers: 0
    search_space_n_floats: {dimension}
    search_space_has_conditionals: false
    search_space_has_forbiddens: false
    search_space_has_priors: false
"""
    return content


def generate_all_dim128_configs(target_dir: Path = TARGET_DIR, dimension: int = 128) -> List[Path]:
    """Generates all 72 YAML configs for the given dimension."""
    target_dir.mkdir(parents=True, exist_ok=True)
    created_files: List[Path] = []

    for fid in range(1, 25):
        for inst in (0, 1, 2):
            yaml_content = generate_single_dim128_yaml(fid=fid, inst=inst, dimension=dimension)
            cfg_path = target_dir / f"cfg_{dimension}_{fid}_{inst}.yaml"
            cfg_path.write_text(yaml_content, encoding="utf-8")
            created_files.append(cfg_path)

    print(f"Generated {len(created_files)} configs in {target_dir}")
    return created_files


def main():
    parser = argparse.ArgumentParser(description="Generate 128D BBOB task configs for CARP-S.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=TARGET_DIR,
        help="Directory where cfg_128_*.yaml files will be written.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=128,
        help="Dimension to generate (default: 128).",
    )
    args = parser.parse_args()
    generate_all_dim128_configs(target_dir=args.target_dir, dimension=args.dimension)


if __name__ == "__main__":
    main()
