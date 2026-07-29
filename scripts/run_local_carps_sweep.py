#!/usr/bin/env python3
"""Local Multiprocessing Runner for CARP-S Epistemic BO Sweeps."""

import os
import sys
import argparse
import subprocess
import concurrent.futures
import time

def get_default_python_exec():
    venv_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable

def load_tasks_from_file(tasks_file: str) -> list:
    if not os.path.exists(tasks_file):
        raise FileNotFoundError(f"Tasks file not found: {tasks_file}")

    tasks = []
    with open(tasks_file, "r") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if line_str:
                tasks.append((idx, line_str))
    return tasks

def execute_single_carps_task(task_info: tuple, log_dir: str, dry_run: bool = False, python_exec: str = None) -> dict:
    if python_exec is None:
        python_exec = get_default_python_exec()

    task_idx, task_args = task_info
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"task_{task_idx}.log")

    if dry_run:
        return {"task_idx": task_idx, "status": "success_dry_run", "elapsed": 0.0}

    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    cmd = f"{python_exec} scripts/run_carps_patched.py --config-dir carps_integration/configs {task_args}"

    try:
        with open(log_file, "w") as out:
            subprocess.run(
                cmd,
                shell=True,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                check=True
            )
        elapsed = time.time() - start_time
        return {"task_idx": task_idx, "status": "success", "elapsed": elapsed}
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        return {"task_idx": task_idx, "status": "failed", "elapsed": elapsed, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Local Multiprocessing Runner for CARP-S Epistemic BO Sweep")
    parser.add_argument(
        "--tasks_file",
        type=str,
        default="results/epistemic_full_acq_array_tasks.txt",
        help="Path to tasks text file"
    )
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum worker processes (default: 8)")
    parser.add_argument("--python_exec", type=str, default=None, help="Python executable path")
    parser.add_argument("--dry_run", action="store_true", help="Dry run test without running evaluations")
    args = parser.parse_args()

    python_exec = args.python_exec if args.python_exec else get_default_python_exec()

    print(f"\n==================================================")
    print(f"STARTING LOCAL MULTIPROCESSING CARP-S SWEEP: {args.tasks_file}")
    print(f"==================================================")

    tasks = load_tasks_from_file(args.tasks_file)
    total_tasks = len(tasks)
    log_dir = os.path.join(os.path.dirname(args.tasks_file), "logs")
    print(f"Loaded {total_tasks} tasks. Output log directory: {log_dir}")
    print(f"Running with max_workers={args.max_workers} using Python: {python_exec}")

    completed = 0
    failed = 0
    start_all = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(execute_single_carps_task, t, log_dir, args.dry_run, python_exec): t[0]
            for t in tasks
        }

        for future in concurrent.futures.as_completed(futures):
            t_idx = futures[future]
            try:
                res = future.result()
                completed += 1
                if res["status"] == "failed":
                    failed += 1
                    print(f"  [Task {t_idx}/{total_tasks}] FAILED (Elapsed: {res['elapsed']:.1f}s)")
                else:
                    pct = (completed / total_tasks) * 100
                    print(f"  Progress: {completed}/{total_tasks} ({pct:.1f}%) completed", end="\r", flush=True)
            except Exception as exc:
                failed += 1
                completed += 1
                print(f"  [Task {t_idx}/{total_tasks}] EXCEPTION: {exc}")

    total_elapsed = time.time() - start_all
    print(f"\nCompleted sweep: {completed - failed}/{total_tasks} succeeded ({failed} failed) in {total_elapsed:.1f}s")

if __name__ == "__main__":
    main()
