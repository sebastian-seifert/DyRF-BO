#!/usr/bin/env python3
"""Local concurrent executor for Lower-Quantile UQ Sweep tasks.

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

DEFAULT_TASK_FILE = "results/lower_quantile_sweep/tasks.txt"


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
        description="Run lower-quantile UQ sweep tasks concurrently on local machine."
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
        - 'elapsed_seconds': float
        - 'results': list[dict[str, Any]]
    """
    path = Path(task_file)
    if not path.is_file():
        raise FileNotFoundError(f"Task file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        all_lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    total_tasks = len(all_lines)
    if limit is not None and limit > 0:
        task_lines = all_lines[:limit]
    else:
        task_lines = all_lines

    executed_count = len(task_lines)
    workers = max_workers or max(1, (os.cpu_count() or 2) - 1)

    print("=" * 60)
    print(f"Executing Lower-Quantile UQ Sweep (tasks: {executed_count}/{total_tasks}, workers: {workers}, dry_run={dry_run})")
    print(f"Task file: {path.resolve()}")
    print("=" * 60)

    results: List[Dict[str, Any]] = []
    succeeded = 0
    failed = 0
    t_start = time.perf_counter()

    worker_items = [(idx + 1, cmd, dry_run) for idx, cmd in enumerate(task_lines)]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_execute_task_worker, item): item[0] for item in worker_items}
        done_count = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done_count += 1
            if res["status"] in ("success", "dry_run"):
                succeeded += 1
            else:
                failed += 1
                print(f"[FAILED] Task {res['task_id']}: exit={res['exit_code']} err={res['error'][:120] if res['error'] else ''}")

            if done_count % max(1, executed_count // 10) == 0 or done_count == executed_count:
                pct = int((done_count / executed_count) * 100)
                elapsed_so_far = time.perf_counter() - t_start
                rate = done_count / max(1e-5, elapsed_so_far)
                print(f"[{pct:3d}%] {done_count}/{executed_count} completed | Success: {succeeded} | Failed: {failed} | Rate: {rate:.1f} tasks/s")

    # Sort results by task_id
    results.sort(key=lambda r: r["task_id"])
    total_elapsed = time.perf_counter() - t_start

    print("=" * 60)
    print(f"Sweep Completed in {total_elapsed:.2f}s | Success: {succeeded}/{executed_count} | Failed: {failed}/{executed_count}")
    print("=" * 60)

    return {
        "total_tasks": total_tasks,
        "executed_tasks": executed_count,
        "succeeded_tasks": succeeded,
        "failed_tasks": failed,
        "elapsed_seconds": total_elapsed,
        "results": results,
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
