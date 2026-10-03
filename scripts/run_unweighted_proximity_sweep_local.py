#!/usr/bin/env python3
"""Local concurrent executor for Unweighted Proximity Sweep tasks.

Dispatches task lines concurrently using concurrent.futures.ProcessPoolExecutor,
tracks real-time progress, captures execution wall-clock time, and aggregates
success/failure counts with single-thread thread-pinning to prevent CPU oversubscription.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASK_FILE = "results/unweighted_proximity_sweep/tasks.txt"


def _execute_task_worker(args: Tuple[int, str, bool]) -> Dict[str, Any]:
    """Execute a single task command in a worker process.

    Parameters
    ----------
    args : tuple[int, str, bool]
        (task_id, cmd, dry_run)

    Returns
    -------
    dict[str, Any]
        Task execution record with keys 'task_id', 'cmd', 'status', 'exit_code',
        'elapsed_seconds', 'error'.
    """
    task_id, cmd, dry_run = args
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
            err_msg = proc.stderr.strip() or proc.stdout.strip()
            return {
                "task_id": task_id,
                "cmd": cmd,
                "status": "failed",
                "exit_code": proc.returncode,
                "elapsed_seconds": elapsed,
                "error": err_msg,
            }
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        return {
            "task_id": task_id,
            "cmd": cmd,
            "status": "failed",
            "exit_code": -1,
            "elapsed_seconds": elapsed,
            "error": str(exc),
        }


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for local sweep runner."""
    parser = argparse.ArgumentParser(
        description="Run unweighted proximity sweep tasks concurrently on local machine."
    )
    parser.add_argument(
        "--task-file",
        type=str,
        default=DEFAULT_TASK_FILE,
        help=f"Path to file containing newline-separated task commands (default: {DEFAULT_TASK_FILE}).",
    )
    parser.add_argument(
        "--max-workers",
        "--workers",
        dest="max_workers",
        type=int,
        default=None,
        help="Maximum parallel worker processes (default: auto-detected CPU cores - 1).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to execute.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log task commands without actually executing them.",
    )
    return parser


def run_local_sweep(
    task_file: str | Path = DEFAULT_TASK_FILE,
    max_workers: Optional[int] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute sweep tasks concurrently using ProcessPoolExecutor.

    Parameters
    ----------
    task_file : str | Path, default=DEFAULT_TASK_FILE
        Path to file containing task commands.
    max_workers : int, optional
        Concurrency limit. Defaults to max(1, os.cpu_count() - 1).
    limit : int, optional
        Optional cap on executed tasks.
    dry_run : bool, default=False
        If True, marks tasks dry_run without invoking shell subprocesses.

    Returns
    -------
    dict[str, Any]
        Summary containing:
        - 'total_tasks': int
        - 'executed_tasks': int
        - 'succeeded_tasks': int
        - 'failed_tasks': int
        - 'results': list[dict]
        - 'elapsed_seconds': float
    """
    task_path = Path(task_file)
    if not task_path.exists():
        raise FileNotFoundError(f"Task file '{task_path}' not found.")

    with open(task_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    total_tasks = len(lines)
    selected_lines = lines[:limit] if limit is not None else lines
    tasks_to_run = list(enumerate(selected_lines, start=1))
    n_exec = len(tasks_to_run)

    if n_exec == 0:
        return {
            "total_tasks": total_tasks,
            "executed_tasks": 0,
            "succeeded_tasks": 0,
            "failed_tasks": 0,
            "results": [],
            "elapsed_seconds": 0.0,
        }

    detected_cores = os.cpu_count() or 1
    workers = max_workers if max_workers is not None else max(1, detected_cores - 1)

    print(
        f"Starting unweighted proximity sweep: {n_exec}/{total_tasks} tasks "
        f"with {workers} workers (dry_run={dry_run})."
    )

    start_total = time.perf_counter()
    results: List[Dict[str, Any]] = []

    # ProcessPoolExecutor for concurrent multiprocessing execution
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(_execute_task_worker, (task_id, cmd, dry_run)): task_id
            for task_id, cmd in tasks_to_run
        }

        completed = 0
        for future in as_completed(future_to_task):
            res = future.result()
            results.append(res)
            completed += 1

            pct = (completed / n_exec) * 100.0
            if res["status"] == "failed":
                print(
                    f"[FAIL] Task {res['task_id']} failed with code {res['exit_code']}: {res['error']}"
                )
            elif completed % 10 == 0 or completed == n_exec:
                print(f"[Progress: {completed}/{n_exec} ({pct:.1f}%)] Task {res['task_id']} completed.")

    elapsed_total = time.perf_counter() - start_total
    results.sort(key=lambda r: r["task_id"])

    succeeded = sum(1 for r in results if r["status"] in ("success", "dry_run"))
    failed = sum(1 for r in results if r["status"] == "failed")

    print(
        f"Completed unweighted proximity sweep in {elapsed_total:.2f}s: "
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
        max_workers=args.max_workers,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    return 0 if summary["failed_tasks"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
