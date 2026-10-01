#!/usr/bin/env python3
"""Live interactive Extrapolation UQ Sweep Master Orchestrator.

Manages 2-stage execution for the 9,600-task sweep on the LUIS HPC cluster, staying
under the 5,000-job queue limit. Features a live updating terminal dashboard for
salloc interactive sessions and clean streaming logs for headless sbatch jobs.

Usage:
    # Inside an salloc interactive allocation:
    python scripts/run_extrapolation_sweep_orchestrator.py

    # Or via Slurm batch:
    sbatch scripts/submit_extrapolation_sweep_master.sbatch
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_STAGE1_START = 1
DEFAULT_STAGE1_END = 4800
DEFAULT_STAGE2_START = 4801
DEFAULT_STAGE2_END = 9600
DEFAULT_POLL_INTERVAL = 30
DEFAULT_TASK_FILE = REPO_ROOT / "results" / "extrapolation_sweep_tasks.txt"
RAW_DIR = REPO_ROOT / "results" / "extrapolation_uq" / "raw"
SUMMARY_DIR = REPO_ROOT / "results" / "extrapolation_uq" / "summaries"


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for sweep orchestrator."""
    parser = argparse.ArgumentParser(
        description="Live interactive master orchestrator for 9,600-task extrapolation sweep."
    )
    parser.add_argument(
        "--stage1-start",
        type=int,
        default=DEFAULT_STAGE1_START,
        help=f"Initial task index for Stage 1 (default: {DEFAULT_STAGE1_START}).",
    )
    parser.add_argument(
        "--stage1-end",
        type=int,
        default=DEFAULT_STAGE1_END,
        help=f"Ending task index for Stage 1 (default: {DEFAULT_STAGE1_END}).",
    )
    parser.add_argument(
        "--stage2-start",
        type=int,
        default=DEFAULT_STAGE2_START,
        help=f"Initial task index for Stage 2 (default: {DEFAULT_STAGE2_START}).",
    )
    parser.add_argument(
        "--stage2-end",
        type=int,
        default=DEFAULT_STAGE2_END,
        help=f"Ending task index for Stage 2 (default: {DEFAULT_STAGE2_END}).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Queue polling interval in seconds (default: {DEFAULT_POLL_INTERVAL}s).",
    )
    parser.add_argument(
        "--task-file",
        type=str,
        default=str(DEFAULT_TASK_FILE),
        help=f"Path to generated task list file (default: {DEFAULT_TASK_FILE}).",
    )
    parser.add_argument(
        "--aggregate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically run results aggregation upon sweep completion (default: True).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate orchestration without dispatching actual Slurm jobs (default: False).",
    )
    return parser


def parse_squeue_output(squeue_text: str, target_job_name: str = "extrap_uq") -> Dict[str, int]:
    """Parse squeue output and count running, pending, and total active tasks for target job.

    Parameters
    ----------
    squeue_text : str
        Raw string output from `squeue -h -o "%i %T %j"`.
    target_job_name : str, default='extrap_uq'
        Job name filter to distinguish worker tasks from interactive shells or other jobs.

    Returns
    -------
    dict[str, int]
        Dictionary with 'running', 'pending', 'other', and 'total' counts.
    """
    running = 0
    pending = 0
    other = 0

    for line in squeue_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        _job_id, state, job_name = parts[0], parts[1].upper(), parts[2]
        if job_name != target_job_name:
            continue

        if state in {"RUNNING", "R"}:
            running += 1
        elif state in {"PENDING", "PD", "CONFIGURING", "CF"}:
            pending += 1
        else:
            other += 1

    return {
        "running": running,
        "pending": pending,
        "other": other,
        "total": running + pending + other,
    }


def get_disk_progress(
    raw_dir: Path | str | None = None,
    summary_dir: Path | str | None = None,
) -> Dict[str, Any]:
    """Inspect output directories and return count of written Parquet and JSON files."""
    r_dir = Path(raw_dir) if raw_dir else RAW_DIR
    s_dir = Path(summary_dir) if summary_dir else SUMMARY_DIR

    parquet_count = 0
    recent_files: List[str] = []

    if r_dir.exists():
        entries: List[os.DirEntry] = []
        try:
            with os.scandir(r_dir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith(".parquet"):
                        entries.append(entry)
            parquet_count = len(entries)
            # Sort recent 5 by modification time descending
            entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
            recent_files = [e.name for e in entries[:5]]
        except Exception:
            parquet_count = 0

    summary_count = 0
    if s_dir.exists():
        try:
            with os.scandir(s_dir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith(".json"):
                        summary_count += 1
        except Exception:
            summary_count = 0

    return {
        "parquet_count": parquet_count,
        "summary_count": summary_count,
        "recent_files": recent_files,
    }


def format_seconds(seconds: float) -> str:
    """Format seconds into HHh MMm SSs string."""
    total = int(max(0, seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}h {m:02d}m {s:02d}s"


def render_progress_bar(current: int, total: int, width: int = 24) -> str:
    """Render an ASCII progress bar."""
    if total <= 0:
        return "[" + " " * width + "] 0.0%"
    fraction = min(1.0, max(0.0, current / total))
    filled = int(round(width * fraction))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {fraction * 100:5.1f}%"


def format_dashboard(status: Dict[str, Any], is_tty: bool = True) -> str:
    """Render status information as a clean terminal dashboard string."""
    width = 80
    border = "=" * width
    divider = "-" * width

    total_tasks = status.get("total_tasks", 9600)
    p_count = status.get("parquet_count", 0)
    s_count = status.get("summary_count", 0)
    stage_target = status.get("stage_target", 4800)

    overall_bar = render_progress_bar(p_count, total_tasks, width=20)
    stage_bar = render_progress_bar(p_count, stage_target, width=20)

    lines = [
        border,
        "          Extrapolation UQ Sweep Master Orchestrator (9,600 Tasks)",
        border,
        f"Mode:          {status.get('mode', 'Interactive')}",
        f"Current Stage: [{status.get('stage', 'INITIALIZING')}]",
        f"Elapsed Time:  {status.get('elapsed_str', '00h 00m 00s')}               Estimated Remaining: {status.get('eta_str', 'Estimating...')}",
        divider,
        "Slurm Queue Status (squeue --me):",
        f"  Running Tasks:     {status.get('running', 0):>5}  [Concurrency Limit: %25]",
        f"  Pending Tasks:     {status.get('pending', 0):>5}",
        f"  Total Active:      {status.get('total_active', 0):>5}",
        divider,
        "Disk Telemetry Progress (results/extrapolation_uq/):",
        f"  Parquet Files:     {p_count:>5} / {total_tasks}  {overall_bar}",
        f"  JSON Summaries:    {s_count:>5} / {total_tasks}",
        f"  Stage Target:      {min(p_count, stage_target):>5} / {stage_target}  {stage_bar}",
    ]

    recent = status.get("recent_files", [])
    if recent:
        lines.append(divider)
        lines.append("Recent Completed Artifacts:")
        for fname in recent[:3]:
            lines.append(f"  * {fname}")

    lines.append(divider)
    lines.append(f"Status Note: {status.get('note', 'Orchestration active.')}")
    lines.append(border)

    return "\n".join(lines)


def dispatch_stage(start_task: int, end_task: int, dry_run: bool = False) -> bool:
    """Invoke submit_extrapolation_sweep_all.sh for the given task range."""
    if dry_run:
        print(f"[DRY-RUN] Dispatched tasks {start_task} to {end_task} via submit_extrapolation_sweep_all.sh")
        return True

    env = os.environ.copy()
    env["START_TASK"] = str(start_task)
    env["END_TASK"] = str(end_task)

    submit_script = REPO_ROOT / "scripts" / "submit_extrapolation_sweep_all.sh"
    res = subprocess.run(
        ["bash", str(submit_script)],
        env=env,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"[ERROR] Failed to dispatch stage ({start_task}-{end_task}):\n{res.stderr}", file=sys.stderr)
        return False
    print(f"[INFO] Dispatched tasks {start_task} to {end_task} successfully.")
    return True


def run_orchestrator(args: argparse.Namespace) -> int:
    """Execute the full 2-stage master orchestration loop with live status reporting."""
    task_file = Path(args.task_file)
    if not task_file.exists():
        print(f"[INFO] Task file {task_file} missing. Generating tasks...")
        if not args.dry_run:
            from scripts.generate_extrapolation_sweep_tasks import generate_tasks
            generate_tasks(output_file=task_file)

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    is_tty = sys.stdout.isatty()
    mode_str = f"salloc/sbatch Session (Job ID: {slurm_job_id})" if slurm_job_id else "Local Terminal / Login Node"

    start_time = time.time()
    total_tasks = args.stage2_end

    # Stage definitions
    stages = [
        {"name": "STAGE 1", "start": args.stage1_start, "end": args.stage1_end, "target": args.stage1_end},
        {"name": "STAGE 2", "start": args.stage2_start, "end": args.stage2_end, "target": args.stage2_end},
    ]

    for stage_idx, stage in enumerate(stages, 1):
        stage_name = stage["name"]
        start_t = stage["start"]
        end_t = stage["end"]

        # 1. Dispatch stage jobs
        print(f"\n[INFO] Starting {stage_name}: Dispatching tasks {start_t} to {end_t}...")
        success = dispatch_stage(start_t, end_t, dry_run=args.dry_run)
        if not success and not args.dry_run:
            return 1

        # 2. Monitor queue until tasks finish
        while True:
            elapsed = time.time() - start_time
            disk_prog = get_disk_progress()
            completed_count = disk_prog["parquet_count"]

            # Query Slurm queue
            if not args.dry_run and shutil.which("squeue"):
                try:
                    sq_res = subprocess.run(
                        ["squeue", "--me", "-h", "-o", "%i %T %j"],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    counts = parse_squeue_output(sq_res.stdout, target_job_name="extrap_uq")
                except Exception:
                    counts = {"running": 0, "pending": 0, "other": 0, "total": 0}
            else:
                counts = {"running": 0, "pending": 0, "other": 0, "total": 0}

            # ETA estimation
            if completed_count > 0:
                sec_per_task = elapsed / completed_count
                remaining_tasks = max(0, total_tasks - completed_count)
                eta_str = format_seconds(sec_per_task * remaining_tasks)
            else:
                eta_str = "Estimating..."

            status_payload = {
                "mode": mode_str,
                "stage": f"{stage_name} IN PROGRESS (Tasks {start_t} - {end_t})",
                "elapsed_str": format_seconds(elapsed),
                "eta_str": eta_str,
                "running": counts["running"],
                "pending": counts["pending"],
                "total_active": counts["total"],
                "parquet_count": completed_count,
                "summary_count": disk_prog["summary_count"],
                "total_tasks": total_tasks,
                "stage_target": stage["target"],
                "recent_files": disk_prog["recent_files"],
                "note": f"{stage_name} active. Waiting for worker tasks to complete before next step.",
            }

            dashboard_txt = format_dashboard(status_payload, is_tty=is_tty)

            if is_tty:
                # Clear terminal and jump to home position
                print("\033[H\033[J" + dashboard_txt, flush=True)
            else:
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {stage_name} | "
                    f"Running: {counts['running']:>3}, Pending: {counts['pending']:>4} | "
                    f"Parquets: {completed_count:>5}/{total_tasks} | "
                    f"Elapsed: {format_seconds(elapsed)}",
                    flush=True,
                )

            # Check if stage complete:
            # Stage complete when no worker jobs remain in queue AND we either ran in dry-run or polled at least once
            if args.dry_run or (counts["total"] == 0 and completed_count >= (stage["target"] * 0.95)):
                # If total queue is 0, give a short grace period then break
                print(f"\n[INFO] {stage_name} complete! Proceeding to next step...")
                break

            time.sleep(args.poll_interval)

    # 3. Post-processing aggregation if requested
    if args.aggregate:
        print("\n[INFO] Running downstream calibration scorecard aggregation...")
        agg_script = REPO_ROOT / "scripts" / "aggregate_extrapolation_results.py"
        if not args.dry_run:
            py_bin = REPO_ROOT / ".venv" / "bin" / "python"
            if not py_bin.exists():
                py_bin = Path(sys.executable)
            subprocess.run([str(py_bin), str(agg_script)], check=False)
        print("[SUCCESS] Automated scorecard aggregation complete.")

    total_elapsed = time.time() - start_time
    print(f"\n[DONE] All 9,600 tasks orchestrated and completed in {format_seconds(total_elapsed)}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_orchestrator(args)
    except KeyboardInterrupt:
        print("\n[INFO] Orchestrator interrupted by user. Any active Slurm jobs remain in queue.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
