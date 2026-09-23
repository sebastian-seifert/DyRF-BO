#!/usr/bin/env python3
"""Multi-process parallel local executor for Extrapolation UQ sweep tasks.

Executes command-line tasks in parallel using a process pool while respecting
resource bounds (thread pinning, process count).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def compute_chunks(total_tasks: int, chunk_size: int = 200) -> List[Tuple[int, int]]:
    """Compute (start, end) 1-based index chunks for batch submission.

    Parameters
    ----------
    total_tasks : int
        Total number of tasks.
    chunk_size : int, default=200
        Maximum number of tasks per chunk.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) task ranges (inclusive, 1-indexed).
    """
    if total_tasks <= 0 or chunk_size <= 0:
        return []

    chunks: List[Tuple[int, int]] = []
    for start in range(1, total_tasks + 1, chunk_size):
        end = min(start + chunk_size - 1, total_tasks)
        chunks.append((start, end))
    return chunks


def execute_command(
    task_id: int,
    cmd: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute a single task command line.

    Parameters
    ----------
    task_id : int
        1-based index of the task.
    cmd : str
        Command string to execute.
    dry_run : bool, default=False
        If True, skips actual command execution.

    Returns
    -------
    dict[str, Any]
        Dictionary with task_id, cmd, status ('dry_run', 'success', 'failed'),
        exit_code, and elapsed_seconds.
    """
    if dry_run:
        return {
            "task_id": task_id,
            "cmd": cmd,
            "status": "dry_run",
            "exit_code": 0,
            "elapsed_seconds": 0.0,
            "error": None,
        }

    env = os.environ.copy()
    # Constrain single-thread execution per worker
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"

    start_time = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - start_time
        if proc.returncode == 0:
            return {
                "task_id": task_id,
                "cmd": cmd,
                "status": "success",
                "exit_code": 0,
                "elapsed_seconds": elapsed,
                "error": None,
            }
        else:
            return {
                "task_id": task_id,
                "cmd": cmd,
                "status": "failed",
                "exit_code": proc.returncode,
                "elapsed_seconds": elapsed,
                "error": proc.stderr.strip() or proc.stdout.strip(),
            }
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return {
            "task_id": task_id,
            "cmd": cmd,
            "status": "failed",
            "exit_code": -1,
            "elapsed_seconds": elapsed,
            "error": str(e),
        }


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for local multi-processing sweep runner."""
    parser = argparse.ArgumentParser(
        description="Run extrapolation sweep tasks locally in parallel."
    )
    parser.add_argument(
        "--task-file",
        type=str,
        default="results/extrapolation_sweep_tasks.txt",
        help="Path to task file (default: results/extrapolation_sweep_tasks.txt).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help=f"Number of parallel worker processes (default: {max(1, (os.cpu_count() or 1) - 1)}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to execute from the task file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log task commands without executing them.",
    )
    return parser


def run_local_sweep(
    task_file: str | Path,
    workers: Optional[int] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute sweep tasks from file in parallel worker pool.

    Parameters
    ----------
    task_file : str | Path
        Path to file containing newline-separated task commands.
    workers : int, optional
        Number of parallel processes (default: cpu_count - 1).
    limit : int, optional
        Optional limit on number of tasks to run.
    dry_run : bool, default=False
        If True, log without running subprocesses.

    Returns
    -------
    dict[str, Any]
        Summary dictionary with execution metrics and individual results.
    """
    task_path = Path(task_file)
    if not task_path.exists():
        raise FileNotFoundError(f"Task file '{task_path}' not found.")

    with open(task_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    total_tasks = len(lines)
    if total_tasks == 0:
        return {
            "total_tasks": 0,
            "executed_tasks": 0,
            "succeeded_tasks": 0,
            "failed_tasks": 0,
            "results": [],
            "elapsed_seconds": 0.0,
        }

    selected_lines = lines[:limit] if limit is not None else lines
    tasks_to_run = list(enumerate(selected_lines, start=1))
    n_exec = len(tasks_to_run)

    max_workers = workers if workers is not None else max(1, (os.cpu_count() or 1) - 1)
    print(f"Starting local sweep: {n_exec}/{total_tasks} tasks with {max_workers} worker processes (dry_run={dry_run}).")

    start_total = time.perf_counter()
    results: List[Dict[str, Any]] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(execute_command, task_id, cmd, dry_run): task_id
            for task_id, cmd in tasks_to_run
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_map):
            res = future.result()
            results.append(res)
            completed += 1
            if res["status"] == "failed":
                print(f"[FAIL] Task {res['task_id']} exited with code {res['exit_code']}: {res['error']}")
            elif completed % 25 == 0 or completed == n_exec:
                print(f"Progress: [{completed}/{n_exec}] ({completed / n_exec * 100:.1f}%) completed.")

    elapsed_total = time.perf_counter() - start_total
    # Sort results by task_id
    results.sort(key=lambda r: r["task_id"])

    succeeded = sum(1 for r in results if r["status"] in ("success", "dry_run"))
    failed = sum(1 for r in results if r["status"] == "failed")

    print(
        f"Completed sweep in {elapsed_total:.2f}s: "
        f"{succeeded} succeeded, {failed} failed out of {n_exec} executed."
    )

    return {
        "total_tasks": total_tasks,
        "executed_tasks": n_exec,
        "succeeded_tasks": succeeded,
        "failed_tasks": failed,
        "results": results,
        "elapsed_seconds": elapsed_total,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    summary = run_local_sweep(
        task_file=args.task_file,
        workers=args.workers,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    return 0 if summary["failed_tasks"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
