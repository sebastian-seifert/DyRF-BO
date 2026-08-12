#!/usr/bin/env python3
"""Local Multi-Worker Crash Interception Harness.

Executes local task command lines, auto-detecting N-2 CPU cores, capturing detailed
stdout/stderr tracebacks on failure into local_results/crashes/<timestamp>/, and auto-generating
RED pytest reproduction stubs.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import concurrent.futures

def get_default_python_exec():
    # Priority: local .venv -> project parent .venv -> sys.executable
    local_venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")
    parent_venv = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".venv", "bin", "python")
    
    if os.path.exists(local_venv):
        return local_venv
    if os.path.exists(parent_venv):
        return parent_venv
    return sys.executable

def get_auto_workers() -> int:
    cpus = os.cpu_count() or 4
    return max(1, cpus - 2)

def execute_single_task(task_idx: int, command_str: str, python_exec: str = None, crash_out_dir: str = "local_results/crashes") -> dict:
    if python_exec is None:
        python_exec = get_default_python_exec()

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    cmd = f"{python_exec} scripts/run_carps_patched.py {command_str}"

    start_time = time.time()
    proc = subprocess.run(
        cmd,
        shell=True,
        env=env,
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start_time

    if proc.returncode == 0:
        return {
            "status": "success",
            "task_idx": task_idx,
            "elapsed": elapsed
        }

    # Intercept crash details
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    crash_dir = os.path.join(crash_out_dir, timestamp, f"crash_task_{task_idx}")
    os.makedirs(crash_dir, exist_ok=True)

    crash_json_path = os.path.join(crash_dir, "crash_report.json")
    report_data = {
        "task_idx": task_idx,
        "command": command_str,
        "full_cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "timestamp": timestamp,
        "elapsed": elapsed
    }

    with open(crash_json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # Automatically write RED pytest stub file
    pytest_stub_path = os.path.join("tests", f"test_reproduce_crash_task_{task_idx}.py")
    test_code = f'''import os
import sys
import subprocess
import pytest

def test_reproduce_crash_task_{task_idx}():
    """Auto-generated RED reproduction test for Task {task_idx}."""
    python_exec = "{python_exec}"
    cmd = f"{{python_exec}} scripts/run_carps_patched.py {command_str}"
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    res = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Task {task_idx} failed with exit code {{res.returncode}}:\\nSTDOUT:\\n{{res.stdout}}\\nSTDERR:\\n{{res.stderr}}"
'''

    with open(pytest_stub_path, "w") as f:
        f.write(test_code)

    print(f"\n[CRASH DETECTED] Task {task_idx} failed! Crash report: {crash_json_path}")
    print(f"[TDD STUB GENERATED] Created RED test: {pytest_stub_path}")

    return {
        "status": "failed",
        "task_idx": task_idx,
        "elapsed": elapsed,
        "crash_dir": crash_dir,
        "pytest_stub": pytest_stub_path,
        "returncode": proc.returncode,
        "error_summary": proc.stderr.splitlines()[-5:] if proc.stderr else []
    }

def main():
    parser = argparse.ArgumentParser(description="Local Multi-Worker Crash Interception Harness")
    parser.add_argument("--task_file", type=str, default="results/local_smoke_tasks.txt", help="Task file path")
    parser.add_argument("--max_workers", type=int, default=None, help="Parallel worker processes (default: N-2 cores)")
    parser.add_argument("--python_exec", type=str, default=None, help="Python executable path")
    parser.add_argument("--crash_out_dir", type=str, default="local_results/crashes", help="Directory for crash logs")
    args = parser.parse_args()

    python_exec = args.python_exec if args.python_exec else get_default_python_exec()
    max_workers = args.max_workers if args.max_workers else get_auto_workers()

    if not os.path.exists(args.task_file):
        print(f"Task file not found: {args.task_file}. Generating local smoke task file...")
        from scripts.generate_local_smoke_tasks import generate_local_smoke_tasks
        generate_local_smoke_tasks(args.task_file)

    tasks = []
    with open(args.task_file, "r") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if line_str:
                tasks.append((idx, line_str))

    total_tasks = len(tasks)
    print(f"==================================================")
    print(f"STARTING LOCAL CRASH HARNESS")
    print(f"Task File: {args.task_file} ({total_tasks} tasks)")
    print(f"Workers: {max_workers} (leaving 2 cores free for system)")
    print(f"Python Exec: {python_exec}")
    print(f"Crash Output Dir: {args.crash_out_dir}")
    print(f"==================================================")

    completed = 0
    failed = 0
    crashes = []
    start_all = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(execute_single_task, t[0], t[1], python_exec, args.crash_out_dir): t[0]
            for t in tasks
        }

        for future in concurrent.futures.as_completed(futures):
            t_idx = futures[future]
            completed += 1
            try:
                res = future.result()
                if res["status"] == "failed":
                    failed += 1
                    crashes.append(res)
                    print(f"  [Task {t_idx}/{total_tasks}] FAILED (Exit Code {res['returncode']})")
                else:
                    pct = (completed / total_tasks) * 100
                    print(f"  Progress: {completed}/{total_tasks} ({pct:.1f}%) completed", end="\r", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  [Task {t_idx}/{total_tasks}] HARNESS EXCEPTION: {exc}")

    total_elapsed = time.time() - start_all
    print(f"\n\n==================================================")
    print(f"SUMMARY: {completed - failed}/{total_tasks} succeeded ({failed} failed) in {total_elapsed:.1f}s")
    if failed > 0:
        print(f"Failed task pytest stubs generated:")
        for c in crashes:
            print(f"  - Task {c['task_idx']}: {c.get('pytest_stub', 'N/A')}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
