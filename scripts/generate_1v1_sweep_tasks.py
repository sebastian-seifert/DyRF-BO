#!/usr/bin/env python3
"""Task Generator for 3 Standalone 1v1 Sweeps with 30 Seeds.

Generates exact, isolated CARP-S Hydra task lists for:
1. Sweep 1 (Disagreement 1v1): SMAC20_CustomUncertainty_ei_standard_disagreement vs SMAC3_HPOFacade_ei
2. Sweep 2 (Proximity 1v1):    SMAC20_CustomUncertainty_ei_standard_proximity vs SMAC3_HPOFacade_ei
3. Sweep 3 (Credal 1v1):       CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal vs SMAC3_HPOFacade_ei
"""

import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry

SWEEP_CONFIGS = {
    "disagreement": {
        "title": "Direct Epistemic Disagreement vs. Standard SMAC3 Baseline",
        "approach": "standard_disagreement",
        "paradigm": "direct",
        "optimizer_id": "SMAC20_CustomUncertainty_ei_standard_disagreement",
        "baseline_id": "SMAC3_HPOFacade_ei",
        "dir_name": "sweep_1v1_disagreement"
    },
    "proximity": {
        "title": "Direct Epistemic Standard Proximity vs. Standard SMAC3 Baseline",
        "approach": "standard_proximity",
        "paradigm": "direct",
        "optimizer_id": "SMAC20_CustomUncertainty_ei_standard_proximity",
        "baseline_id": "SMAC3_HPOFacade_ei",
        "dir_name": "sweep_1v1_proximity"
    },
    "credal": {
        "title": "Decoupled Additive Likelihood Credal vs. Standard SMAC3 Baseline",
        "approach": "likelihood_credal",
        "paradigm": "additive",
        "optimizer_id": "CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal",
        "baseline_id": "SMAC3_HPOFacade_ei",
        "dir_name": "sweep_1v1_credal"
    }
}

def generate_single_1v1_sweep(
    sweep_name: str,
    output_file: str,
    runs_dir: str,
    n_seeds: int = 30,
    trials: int = 50,
    beta_max: float = 1.0,
    warmup_ratio: float = 0.20
) -> list[str]:
    """Generates the Hydra CLI task lines for a single 1v1 sweep."""
    if sweep_name not in SWEEP_CONFIGS:
        raise ValueError(f"Unknown sweep name '{sweep_name}'. Options: {list(SWEEP_CONFIGS.keys())}")
        
    cfg = SWEEP_CONFIGS[sweep_name]
    tasks = CarpsBBSubsetRegistry.get_dev_tasks()
    seeds = list(range(1, n_seeds + 1))
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    os.makedirs(os.path.abspath(runs_dir), exist_ok=True)
    
    lines = []
    
    # 1. Approach Tasks (20 tasks * 30 seeds = 600 tasks)
    for task in tasks:
        task_name = task.split("/")[-1]
        task_arg = f"+{task}" if not task.startswith("+") else task
        for seed in seeds:
            telemetry = f"{runs_dir}/telemetry_{cfg['optimizer_id']}_{task_name}_seed{seed}.json"
            
            if cfg["paradigm"] == "direct":
                line = (
                    f"--config-dir carps_integration/configs "
                    f"+optimizer=smac20_custom_uncertainty "
                    f"++optimizer.acq_func_name=ei "
                    f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={cfg['approach']} "
                    f"{task_arg} task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"optimizer_id={cfg['optimizer_id']} optimizer_container_id=SMAC20_CustomUncertainty "
                    f"log_dir={runs_dir}"
                )
            else: # additive
                line = (
                    f"--config-dir carps_integration/configs "
                    f"+optimizer=dyrf_additive_epistemic_ei "
                    f"++optimizer.extractor_name={cfg['approach']} "
                    f"++optimizer.beta_max={beta_max} "
                    f"++optimizer.warmup_ratio={warmup_ratio} "
                    f"{task_arg} task.optimization_resources.n_trials={trials} "
                    f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                    f"optimizer_id={cfg['optimizer_id']} optimizer_container_id=CARPSDynamicRF "
                    f"log_dir={runs_dir}"
                )
            lines.append(line)
            
    # 2. SMAC3 Baseline Tasks (20 tasks * 30 seeds = 600 tasks)
    for task in tasks:
        task_name = task.split("/")[-1]
        task_arg = f"+{task}" if not task.startswith("+") else task
        for seed in seeds:
            telemetry = f"{runs_dir}/telemetry_smac3_ei_{task_name}_seed{seed}.json"
            line = (
                f"--config-dir carps_integration/configs "
                f"+optimizer/smac20=hpo "
                f"++optimizer.acq_func_name=ei "
                f"{task_arg} task.optimization_resources.n_trials={trials} "
                f"seed={seed} ++optimizer.telemetry_path={telemetry} "
                f"optimizer_id=SMAC3_HPOFacade_ei optimizer_container_id=SMAC3_HPOFacade "
                f"log_dir={runs_dir}"
            )
            lines.append(line)
            
    with open(output_file, "w") as f:
        for line in lines:
            f.write(line + "\n")
            
    print(f"[✓] Generated '{sweep_name}' ({cfg['title']}):")
    print(f"    - Approach: {cfg['optimizer_id']}")
    print(f"    - Baseline: {cfg['baseline_id']}")
    print(f"    - Seeds: 1 to {n_seeds} ({n_seeds} total)")
    print(f"    - Total Tasks: {len(lines)} runs written to {output_file}")
    print(f"    - Runs Directory: {runs_dir}")
    
    return lines

def generate_all_1v1_sweeps(
    base_dir: str = "results",
    n_seeds: int = 30
) -> dict[str, str]:
    """Generates all 3 1v1 sweep task files in their respective results directories."""
    file_map = {}
    for name, cfg in SWEEP_CONFIGS.items():
        sweep_folder = os.path.join(base_dir, cfg["dir_name"])
        out_file = os.path.join(sweep_folder, "tasks.txt")
        runs_dir = os.path.join(sweep_folder, "runs")
        generate_single_1v1_sweep(name, out_file, runs_dir, n_seeds=n_seeds)
        file_map[name] = out_file
    return file_map

if __name__ == "__main__":
    base_results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    seeds_count = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    print("\n==================================================")
    print(f"GENERATING 3 STANDALONE 1v1 SWEEPS ({seeds_count} SEEDS EACH)")
    print("==================================================")
    generate_all_1v1_sweeps(base_results_dir, seeds_count)
    print("\n==================================================")
    print("ALL 3 SWEEP TASK FILES SUCCESSFULLY CREATED!")
    print("==================================================")
