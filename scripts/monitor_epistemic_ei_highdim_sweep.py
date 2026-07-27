#!/usr/bin/env python3
import os
import sys
import argparse

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors import UQExtractorRegistry
from scripts.benchmark_registry import BenchmarkRegistry

def monitor_highdim_sweep(tasks_file: str = "results/epistemic_ei_highdim_array_tasks.txt", results_dir: str = "results"):
    if not os.path.exists(tasks_file):
        print(f"Tasks file '{tasks_file}' not found. Generating array tasks...")
        from scripts.generate_epistemic_ei_highdim_array_tasks import generate_highdim_array_tasks
        generate_highdim_array_tasks(output_path=tasks_file)

    with open(tasks_file, "r") as f:
        tasks = [line.strip() for line in f if line.strip()]

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No tasks found in tasks file.")
        return

    extractors = UQExtractorRegistry.list_registered()
    extractor_stats = {ext: {"total": 0, "completed": 0} for ext in extractors}
    smac_stats = {"total": 0, "completed": 0}

    completed_total = 0

    for line in tasks:
        if "+optimizer/smac20=hpo" in line:
            smac_stats["total"] += 1
        else:
            ext_found = None
            for ext in sorted(extractors, key=len, reverse=True):
                if f"optimizer.extractor_name={ext} " in line:
                    ext_found = ext
                    break
            if ext_found:
                extractor_stats[ext_found]["total"] += 1

            telemetry_path = None
            parts = line.split()
            for part in parts:
                if part.startswith("++optimizer.telemetry_path="):
                    telemetry_path = part.split("=", 1)[1]
                    break

            if telemetry_path and os.path.exists(telemetry_path) and os.path.getsize(telemetry_path) > 0:
                completed_total += 1
                if ext_found:
                    extractor_stats[ext_found]["completed"] += 1

    overall_pct = (completed_total / total_tasks) * 100.0 if total_tasks > 0 else 0.0

    print("================================================================================")
    print("      HIGH-DIMENSIONAL (>20D) EU-GUIDED EI BO SWEEP PROGRESS MONITOR           ")
    print("================================================================================")
    print(f"Overall Progress: [{overall_pct:6.2f}%]  ({completed_total} / {total_tasks} tasks completed)\n")

    print("--- Breakdown by UQ Extractor ---")
    for ext, stats in extractor_stats.items():
        pct = (stats["completed"] / stats["total"]) * 100.0 if stats["total"] > 0 else 0.0
        print(f"  Extractor [{ext:22s}]: [{pct:6.2f}%]  ({stats['completed']:4d} / {stats['total']:4d})")

    print("================================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor High-Dim EI BO Sweep Progress")
    parser.add_argument("--tasks_file", type=str, default="results/epistemic_ei_highdim_array_tasks.txt")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    monitor_highdim_sweep(tasks_file=args.tasks_file, results_dir=args.results_dir)
