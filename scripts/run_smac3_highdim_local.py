#!/usr/bin/env python3
"""
Local parallel executor for SMAC3 High-Dim Baseline Tasks.
Executes tasks from results/smac3_highdim_array_tasks.txt in parallel,
logs temporary outputs to results/array_smac3_*.log,
and extracts telemetry JSON results to results/epistemic_ei_highdim/baseline/.
"""

import os
import sys
import re
import json
import glob
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_telemetry_from_carps_run(carps_run_dir: str, task_name: str, seed: int, output_path: str) -> dict:
    trial_logs_file = os.path.join(carps_run_dir, "trial_logs.jsonl")
    trials = []
    if os.path.exists(trial_logs_file):
        with open(trial_logs_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trials.append(json.loads(line))
                    except Exception:
                        pass

    telemetry = {
        "task_name": task_name,
        "seed": seed,
        "extractor_name": "smac3_bo",
        "trials": trials
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(telemetry, f, indent=2)

    return telemetry

def run_single_smac3_task(task_args: str, output_dir: str = "results/epistemic_ei_highdim/baseline", logs_dir: str = "results") -> str:
    # Parse task name and seed
    task_match = re.search(r"\+task/[^=\s]+=([^\s]+)", task_args)
    task_name = task_match.group(1) if task_match else "unknown_task"

    seed_match = re.search(r"\bseed=(\d+)", task_args)
    seed = int(seed_match.group(1)) if seed_match else 1

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, f"array_smac3_{task_name}_seed{seed}.log")
    telemetry_file = os.path.join(output_dir, f"telemetry_smac3_{task_name}_seed{seed}.json")

    cmd = f"PYTHONPATH=. .venv/bin/python scripts/run_carps_patched.py {task_args}"

    with open(log_file, "w") as lf:
        lf.write(f"Running arguments: {task_args}\n")
        lf.flush()
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=lf,
            stderr=subprocess.STDOUT
        )
        process.wait()

    # Find CARP-S run directory
    # Runs are saved in runs/SMAC3-HPOFacade/...
    possible_runs = glob.glob(f"runs/SMAC3-HPOFacade/**/{seed}", recursive=True)
    matching_run = None
    for pr in possible_runs:
        if task_name in pr or task_name.replace("cfg_rbv2_super_", "") in pr or task_name.replace("cfg_nb301_", "") in pr:
            matching_run = pr
            break

    if matching_run:
        extract_telemetry_from_carps_run(
            carps_run_dir=matching_run,
            task_name=task_name,
            seed=seed,
            output_path=telemetry_file
        )

    return f"Completed {task_name} (seed {seed}) -> Exit code {process.returncode}"

def main():
    parser = argparse.ArgumentParser(description="Run SMAC3 High-Dim Baseline Tasks Locally in Parallel")
    parser.add_argument("--tasks_file", type=str, default="results/smac3_highdim_array_tasks.txt")
    parser.add_argument("--output_dir", type=str, default="results/epistemic_ei_highdim/baseline")
    parser.add_argument("--logs_dir", type=str, default="results")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.tasks_file):
        print(f"Generating tasks file '{args.tasks_file}'...")
        from scripts.generate_smac3_highdim_array_tasks import generate_smac3_highdim_array_tasks
        generate_smac3_highdim_array_tasks(output_path=args.tasks_file)

    with open(args.tasks_file, "r") as f:
        tasks = [line.strip() for line in f if line.strip()]

    print(f"Executing {len(tasks)} SMAC3 High-Dim tasks locally using {args.workers} workers...")
    print(f"Output telemetry directory: {args.output_dir}")
    print(f"Temporary log directory: {args.logs_dir}")

    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_smac3_task, task, args.output_dir, args.logs_dir): task for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            print(f"[{completed:2d}/{len(tasks):2d}] {res}")

    print("All SMAC3 High-Dim baseline tasks completed locally!")

if __name__ == "__main__":
    main()
