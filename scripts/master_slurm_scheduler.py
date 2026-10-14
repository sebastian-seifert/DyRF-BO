#!/usr/bin/env python3
"""Autonomous Master Slurm Scheduler for High-Throughput Sweeps.

Enforces cluster queue limits:
- Polls `squeue --me` to track active jobs and array tasks.
- Triggers new batch submissions when remaining jobs < min_remaining_jobs_trigger (default: 10).
- Guarantees total active tasks never exceed max_active_tasks (default: 4500).
- Manages persistent state in scheduler_state.json to enable safe restarts and crash recovery.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def parse_squeue_me_output(output: str) -> Tuple[int, int]:
    """Parses output from `squeue --me -h -o '%i %t'` to count remaining jobs and tasks.

    Returns:
        Tuple of (distinct_job_count, total_task_count).
    """
    lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
    if not lines:
        return 0, 0

    job_ids = set()
    total_tasks = 0

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        full_id = parts[0]
        # Strip array index (e.g. 12345_1 -> 12345)
        base_id = full_id.split("_")[0]
        # Also handle bracketed array IDs if present (e.g. 12345_[1-100])
        base_id = base_id.split("[")[0]
        job_ids.add(base_id)
        total_tasks += 1

    return len(job_ids), total_tasks


@dataclass
class SchedulerConfig:
    task_manifest: str = "results/sweep_bbob_full_comparison/tasks_all.txt"
    sbatch_script: str = "scripts/submit_bbob_full_sweep_array.sbatch"
    chunk_size: int = 250
    max_active_tasks: int = 4500
    min_remaining_jobs_trigger: int = 10
    array_concurrency: int = 25
    poll_interval_sec: int = 60
    state_file: str = "results/sweep_bbob_full_comparison/scheduler_state.json"
    job_name_prefix: str = "bbob_swp"


class MasterSlurmScheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config

    def calculate_chunks(self, total_tasks: int) -> List[Tuple[int, int]]:
        """Divides 1-based total_tasks into (start_idx, end_idx) chunk intervals."""
        chunks: List[Tuple[int, int]] = []
        chunk_size = self.config.chunk_size
        for start in range(1, total_tasks + 1, chunk_size):
            end = min(start + chunk_size - 1, total_tasks)
            chunks.append((start, end))
        return chunks

    def load_state(self) -> Dict[str, Any]:
        """Loads state dictionary from state_file, or initializes default state."""
        if os.path.exists(self.config.state_file):
            try:
                with open(self.config.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "completed_or_submitted_chunks": 0,
            "submitted_chunks_info": [],
            "all_chunks_dispatched": False,
        }

    def save_state(self, state: Dict[str, Any]) -> None:
        """Persists state dictionary to state_file."""
        state_dir = os.path.dirname(self.config.state_file)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
        with open(self.config.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def query_squeue_me(self) -> Tuple[int, int]:
        """Runs `squeue --me -h -o '%i %t'` via subprocess to query cluster state."""
        try:
            res = subprocess.run(
                ["squeue", "--me", "-h", "-o", "%i %t"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return parse_squeue_me_output(res.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return 0, 0

    def submit_chunk(
        self,
        chunk_idx: int,
        start: int,
        end: int,
        dry_run: bool = False,
    ) -> Optional[str]:
        """Submits a single array chunk to Slurm via sbatch."""
        job_name = f"{self.config.job_name_prefix}_c{chunk_idx}"
        cmd = [
            "sbatch",
            "--parsable",
            f"--job-name={job_name}",
            f"--export=ALL,TASK_FILE={self.config.task_manifest}",
            f"--array={start}-{end}%{self.config.array_concurrency}",
            self.config.sbatch_script,
        ]
        if dry_run:
            print(f"[DRY-RUN] Executing: {' '.join(cmd)}")
            return f"mock_job_{chunk_idx}"

        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            job_id = res.stdout.strip()
            return job_id
        except Exception as e:
            print(f"[ERROR] Failed to submit chunk {chunk_idx} ({start}-{end}): {e}", file=sys.stderr)
            return None

    def step(
        self,
        mock_remaining_jobs: Optional[int] = None,
        mock_active_tasks: Optional[int] = None,
        dry_run: bool = False,
    ) -> List[Tuple[int, int, int]]:
        """Runs one scheduling decision step.

        Returns:
            List of (chunk_idx, start_task, end_task) submitted during this step.
        """
        if not os.path.exists(self.config.task_manifest):
            out_dir = os.path.dirname(self.config.task_manifest) or "."
            if "dim2" in self.config.task_manifest:
                print(f"[INFO] Manifest '{self.config.task_manifest}' not found. Auto-generating BBOB Dim 2 tasks...")
                try:
                    from scripts.generate_bbob_dim2_sweep_tasks import generate_bbob_dim2_sweep_tasks
                except ModuleNotFoundError:
                    from generate_bbob_dim2_sweep_tasks import generate_bbob_dim2_sweep_tasks
                generate_bbob_dim2_sweep_tasks(seeds=20, trials=100, output_dir=out_dir)
            elif "dim4" in self.config.task_manifest:
                print(f"[INFO] Manifest '{self.config.task_manifest}' not found. Auto-generating BBOB Dim 4 tasks...")
                try:
                    from scripts.generate_bbob_dim4_sweep_tasks import generate_bbob_dim4_sweep_tasks
                except ModuleNotFoundError:
                    from generate_bbob_dim4_sweep_tasks import generate_bbob_dim4_sweep_tasks
                generate_bbob_dim4_sweep_tasks(seeds=20, trials=100, output_dir=out_dir)
            else:
                print(f"[INFO] Manifest '{self.config.task_manifest}' not found. Auto-generating full BBOB sweep tasks...")
                try:
                    from scripts.generate_bbob_full_sweep_tasks import generate_bbob_full_sweep_tasks
                except ModuleNotFoundError:
                    from generate_bbob_full_sweep_tasks import generate_bbob_full_sweep_tasks
                generate_bbob_full_sweep_tasks(seeds=20, trials=100, output_dir=out_dir)

        if not os.path.exists(self.config.task_manifest):
            raise FileNotFoundError(f"Manifest not found: {self.config.task_manifest}")

        with open(self.config.task_manifest, "r", encoding="utf-8") as f:
            total_tasks = sum(1 for line in f if line.strip())

        chunks = self.calculate_chunks(total_tasks)
        total_chunk_count = len(chunks)

        state = self.load_state()
        next_chunk_idx = state.get("completed_or_submitted_chunks", 0)

        if next_chunk_idx >= total_chunk_count:
            state["all_chunks_dispatched"] = True
            self.save_state(state)
            return []

        if mock_remaining_jobs is not None and mock_active_tasks is not None:
            remaining_jobs, active_tasks = mock_remaining_jobs, mock_active_tasks
        else:
            remaining_jobs, active_tasks = self.query_squeue_me()

        print(f"[STATUS] Cluster queue: {remaining_jobs} jobs, {active_tasks} tasks currently active.")

        submitted_in_step: List[Tuple[int, int, int]] = []

        # Trigger logic: submit when remaining jobs < min_remaining_jobs_trigger
        if remaining_jobs < self.config.min_remaining_jobs_trigger:
            available_task_capacity = max(0, self.config.max_active_tasks - active_tasks)
            print(f"[TRIGGER] Remaining jobs ({remaining_jobs}) < {self.config.min_remaining_jobs_trigger}. "
                  f"Available task capacity: {available_task_capacity}.")

            while next_chunk_idx < total_chunk_count:
                start, end = chunks[next_chunk_idx]
                chunk_len = end - start + 1

                if chunk_len > available_task_capacity:
                    print(f"[CAP] Next chunk {next_chunk_idx} requires {chunk_len} tasks, but capacity is {available_task_capacity}. Pausing submissions.")
                    break

                job_id = self.submit_chunk(next_chunk_idx, start, end, dry_run=dry_run)
                if job_id is not None:
                    submitted_in_step.append((next_chunk_idx, start, end))
                    state["submitted_chunks_info"].append({
                        "chunk_idx": next_chunk_idx,
                        "start": start,
                        "end": end,
                        "job_id": job_id,
                    })
                    next_chunk_idx += 1
                    available_task_capacity -= chunk_len
                    active_tasks += chunk_len
                else:
                    break

            state["completed_or_submitted_chunks"] = next_chunk_idx
            if next_chunk_idx >= total_chunk_count:
                state["all_chunks_dispatched"] = True
            self.save_state(state)

        return submitted_in_step

    def run_loop(self, dry_run: bool = False, single_step: bool = False) -> None:
        """Main loop that polls the queue periodically until all chunks are dispatched."""
        print(f"Starting Master Slurm Scheduler (Manifest: {self.config.task_manifest})...")
        while True:
            submitted = self.step(dry_run=dry_run)
            state = self.load_state()
            if submitted:
                print(f"[DISPATCH] Submitted {len(submitted)} chunks. Total dispatched: "
                      f"{state['completed_or_submitted_chunks']} chunks.")
            if state.get("all_chunks_dispatched", False):
                print("[DONE] All task chunks have been dispatched to Slurm!")
                break
            if single_step:
                break
            time.sleep(self.config.poll_interval_sec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Slurm Scheduler for BBOB Sweep.")
    parser.add_argument("--manifest", type=str, default="results/sweep_bbob_full_comparison/tasks_all.txt")
    parser.add_argument("--sbatch", type=str, default="scripts/submit_bbob_full_sweep_array.sbatch")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--max-tasks", type=int, default=4500)
    parser.add_argument("--trigger-jobs", type=int, default=10)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--state-file", type=str, default=None)
    parser.add_argument("--job-prefix", "--job-name-prefix", dest="job_prefix", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")

    args = parser.parse_args()

    default_state_file = (
        args.state_file
        if args.state_file
        else os.path.join(os.path.dirname(args.manifest) or ".", "scheduler_state.json")
    )
    default_job_prefix = (
        args.job_prefix
        if args.job_prefix
        else ("bbob_d2" if "dim2" in args.manifest else ("bbob_d4" if "dim4" in args.manifest else "bbob_swp"))
    )

    cfg = SchedulerConfig(
        task_manifest=args.manifest,
        sbatch_script=args.sbatch,
        chunk_size=args.chunk_size,
        max_active_tasks=args.max_tasks,
        min_remaining_jobs_trigger=args.trigger_jobs,
        array_concurrency=args.concurrency,
        poll_interval_sec=args.interval,
        state_file=default_state_file,
        job_name_prefix=default_job_prefix,
    )
    scheduler = MasterSlurmScheduler(cfg)
    scheduler.run_loop(dry_run=args.dry_run, single_step=args.once)
