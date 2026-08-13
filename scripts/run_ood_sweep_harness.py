#!/usr/bin/env python3
"""Multi-Worker Local Execution Harness for OOD Benchmark Sweep (630 Tasks).

Executes all 630 tasks in results/ood_sweep_tasks.txt in parallel using
multiprocessing with N-2 CPU cores, displaying rich progress telemetry and error handling.
"""

import os
import sys
import time
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_single_ood_task(cmd_str: str) -> tuple[str, int, float, str]:
    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    # Replace 'python ' with current sys.executable
    if cmd_str.startswith("python "):
        cmd_str = f"{sys.executable} " + cmd_str[7:]

    try:
        res = subprocess.run(
            cmd_str,
            shell=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=300
        )

        elapsed = time.time() - start_time
        err_msg = res.stderr if res.returncode != 0 else ""
        return cmd_str, res.returncode, elapsed, err_msg
    except Exception as e:
        elapsed = time.time() - start_time
        return cmd_str, -1, elapsed, str(e)

def main():
    task_file = "results/ood_sweep_tasks.txt"
    if not os.path.exists(task_file):
        print(f"Task file {task_file} missing. Generating now...")
        from scripts.generate_ood_sweep_tasks import generate_ood_sweep_tasks
        generate_ood_sweep_tasks()

    with open(task_file, "r") as f:
        tasks = [l.strip() for l in f if l.strip()]

    total_tasks = len(tasks)
    num_workers = max(1, mp.cpu_count() - 2)
    print(f"==================================================")
    print(f"Starting OOD Benchmark Multi-Worker Local Harness")
    print(f"Total Tasks: {total_tasks} | CPU Workers: {num_workers}")
    print(f"==================================================")

    succeeded = 0
    failed = 0
    start_all = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_cmd = {executor.submit(run_single_ood_task, cmd): cmd for cmd in tasks}
        
        for idx, future in enumerate(as_completed(future_to_cmd), 1):
            cmd, code, elapsed, err = future.result()
            if code == 0:
                succeeded += 1
            else:
                failed += 1
                print(f"\n[FAIL] Task {idx}/{total_tasks} failed (code {code}):\n  Cmd: {cmd}\n  Err: {err[:300]}")

            if idx % 10 == 0 or idx == total_tasks:
                pct = idx / total_tasks * 100
                elapsed_total = time.time() - start_all
                rate = idx / elapsed_total
                eta = (total_tasks - idx) / rate if rate > 0 else 0
                print(f"Progress: {idx}/{total_tasks} ({pct:.1f}%) | Pass: {succeeded} | Fail: {failed} | ETA: {eta:.1f}s", flush=True)

    total_elapsed = time.time() - start_all
    print(f"\n==================================================", flush=True)
    print(f"OOD Benchmark Local Sweep Complete!", flush=True)
    print(f"Total Time: {total_elapsed:.1f}s | Pass: {succeeded}/{total_tasks} | Fail: {failed}", flush=True)
    print(f"==================================================", flush=True)


if __name__ == "__main__":
    main()
