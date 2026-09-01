#!/usr/bin/env python3
"""Master Log Parser for CARP-S BBsubset Dev Sweep (2,700 Tasks).

Parses all log and error files in results/bbsubset_dev/logs/, classifies
executions by benchmark type (BBOB, YAHPO, HPOBench), detects network/container
pre-caching requirements, and outputs a complete status audit.
"""

import os
import glob
import re
from collections import defaultdict
from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry
from ep_extractors import UQExtractorRegistry

def parse_bbsubset_dev_logs():
    tasks = CarpsBBSubsetRegistry.get_dev_tasks()
    extractors = UQExtractorRegistry.list_registered()
    acquisitions = ["ei", "pi", "lcb"]

    log_dir = "results/bbsubset_dev/logs"
    err_files = sorted(glob.glob(os.path.join(log_dir, "*.err")))

    print(f"Found {len(err_files)} log/err files in {log_dir}")

    total_tasks = len(err_files)
    succeeded = 0
    failed_singularity_network = 0
    failed_other = 0

    benchmark_stats = defaultdict(lambda: {"total": 0, "pass": 0, "fail_net": 0, "fail_other": 0})
    approach_stats = defaultdict(lambda: {"total": 0, "pass": 0, "fail": 0})
    acq_stats = defaultdict(lambda: {"total": 0, "pass": 0, "fail": 0})

    # Task list mapping: index 1..2700
    task_list_file = "results/bbsubset_dev_tasks.txt"
    task_cmds = []
    if os.path.exists(task_list_file):
        with open(task_list_file, "r") as f:
            task_cmds = [l.strip() for l in f if l.strip()]

    for err_path in err_files:
        # Extract task index from filename: dev_<job_id>_<idx>.err
        basename = os.path.basename(err_path)
        match = re.search(r"_(\d+)\.err$", basename)
        if not match:
            continue
        idx = int(match.group(1))

        # Retrieve command metadata if available
        cmd = task_cmds[idx - 1] if 0 < idx <= len(task_cmds) else ""

        # Determine task benchmark name
        benchmark_name = "unknown"
        for t in tasks:
            t_short = t.split("/")[-1]
            if t_short in cmd:
                benchmark_name = t_short
                break

        # Determine approach and acq
        approach = "baseline"
        for ext in extractors:
            if f"uncertainty_func={ext}" in cmd:
                approach = ext
                break

        acq = "ei"
        for a in acquisitions:
            if f"acq_func_name={a}" in cmd:
                acq = a
                break

        # Read err file content
        with open(err_path, "r", errors="ignore") as f:
            content = f.read()

        benchmark_stats[benchmark_name]["total"] += 1
        approach_stats[approach]["total"] += 1
        acq_stats[acq]["total"] += 1

        if "singularity pull" in content or "no route to host" in content or "dial tcp" in content:
            failed_singularity_network += 1
            benchmark_stats[benchmark_name]["fail_net"] += 1
            approach_stats[approach]["fail"] += 1
            acq_stats[acq]["fail"] += 1
        elif "Error executing job" in content or "Traceback" in content:
            failed_other += 1
            benchmark_stats[benchmark_name]["fail_other"] += 1
            approach_stats[approach]["fail"] += 1
            acq_stats[acq]["fail"] += 1
        else:
            succeeded += 1
            benchmark_stats[benchmark_name]["pass"] += 1
            approach_stats[approach]["pass"] += 1
            acq_stats[acq]["pass"] += 1

    print("\n==================================================")
    print("CARP-S BBsubset Dev Sweep Audit Summary")
    print("==================================================")
    print(f"Total Log Files Audited: {total_tasks}")
    print(f"Succeeded Runs:         {succeeded} / {total_tasks} ({succeeded/total_tasks*100:.1f}%)")
    print(f"Singularity/Network:    {failed_singularity_network} / {total_tasks} ({failed_singularity_network/total_tasks*100:.1f}%)")
    print(f"Other Failures:         {failed_other} / {total_tasks} ({failed_other/total_tasks*100:.1f}%)")

    print("\n--------------------------------------------------")
    print("Breakdown by Benchmark Task")
    print("--------------------------------------------------")
    print(f"{'Benchmark Task':<65} | {'Total':<6} | {'Pass':<6} | {'NetFail':<8}")
    print("-" * 90)
    for b_name, s in sorted(benchmark_stats.items()):
        print(f"{b_name:<65} | {s['total']:<6} | {s['pass']:<6} | {s['fail_net']:<8}")

    print("\n--------------------------------------------------")
    print("Breakdown by Optimizer Approach")
    print("--------------------------------------------------")
    print(f"{'Approach':<30} | {'Total':<6} | {'Pass':<6} | {'Fail':<6}")
    print("-" * 55)
    for app_name, s in sorted(approach_stats.items()):
        print(f"{app_name:<30} | {s['total']:<6} | {s['pass']:<6} | {s['fail']:<6}")

    return benchmark_stats

if __name__ == "__main__":
    parse_bbsubset_dev_logs()
