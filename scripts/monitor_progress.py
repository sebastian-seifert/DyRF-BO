#!/usr/bin/env python3
import os
import sys
import glob

def monitor_sweep(sweep_dir: str):
    tasks_file = os.path.join(sweep_dir, "tasks.txt")
    raw_dir = os.path.join(sweep_dir, "raw")
    logs_dir = os.path.join(sweep_dir, "logs")

    if not os.path.exists(tasks_file):
        print(f"Error: {tasks_file} does not exist.")
        return

    with open(tasks_file, "r") as f:
        total_tasks = len(f.readlines())

    completed_raw = len(glob.glob(os.path.join(raw_dir, "*.json"))) if os.path.exists(raw_dir) else 0
    progress_pct = (completed_raw / total_tasks * 100) if total_tasks > 0 else 0.0

    print(f"="*60)
    print(f"ADMIN PROGRESS LOG: {os.path.basename(sweep_dir)}")
    print(f"="*60)
    print(f"Total Array Tasks Scheduled : {total_tasks}")
    print(f"Completed Results (JSON)   : {completed_raw}")
    print(f"Progress                    : {progress_pct:.2f}%")
    print(f"="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_progress.py <path_to_sweep_dir>")
        sys.exit(1)
    monitor_sweep(sys.argv[1])
