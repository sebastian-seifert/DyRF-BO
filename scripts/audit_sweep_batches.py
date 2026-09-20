#!/usr/bin/env python3
"""Batch Completion Auditor and Targeted Rerun Generator for Multi-Suite Sweeps.

Scans run directories against scheduled tasks in `results/sweep_{suite}_proximity/tasks.txt`,
verifies trial count completeness (expected 100 trials), flags truncated/missing tasks,
and generates clean targeted rerun task files and submission scripts.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_BATCHES: List[Dict[str, Any]] = [
    {"batch_id": 1, "suite": "yahpo_rbv2_ranger", "start": 1, "end": 3400},
    {"batch_id": 2, "suite": "yahpo_rbv2_ranger", "start": 3401, "end": 7140},
    {"batch_id": 3, "suite": "yahpo_rbv2_super", "start": 1, "end": 3000},
    {"batch_id": 4, "suite": "yahpo_rbv2_super", "start": 3001, "end": 6180},
    {"batch_id": 5, "suite": "hpobench_ml", "start": 1, "end": 2600},
    {"batch_id": 6, "suite": "hpobench_ml", "start": 2601, "end": 5280},
]


def parse_task_line(line: str) -> Dict[str, Any]:
    """Extracts run configuration metadata from a Hydra CLI command line."""
    tokens = line.strip().split()
    meta = {
        "raw_cmd": line.strip(),
        "optimizer_id": "unknown",
        "task": "unknown",
        "seed": 1,
        "n_trials": 100,
        "baserundir": "runs",
    }
    for tok in tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            k_clean = k.lstrip("+").strip()
            if k_clean == "optimizer_id":
                meta["optimizer_id"] = v
            elif k_clean == "task":
                meta["task"] = v
            elif k_clean == "seed":
                try:
                    meta["seed"] = int(v)
                except ValueError:
                    pass
            elif k_clean == "task.optimization_resources.n_trials":
                try:
                    meta["n_trials"] = int(v)
                except ValueError:
                    pass
            elif k_clean == "baserundir":
                meta["baserundir"] = v
    return meta


def build_suite_runs_index(runs_dir: Path) -> Dict[Tuple[str, str, int], int]:
    """Scans runs_dir ONCE with glob.iglob, indexing (opt_id, task_identifier, seed) -> trial_count."""
    index: Dict[Tuple[str, str, int], int] = {}
    if not runs_dir.is_dir():
        return index

    search_pattern = str(runs_dir / "**" / "trial_logs.jsonl")
    for file_str in glob.iglob(search_pattern, recursive=True):
        log_path = Path(file_str)
        try:
            seed = int(log_path.parent.name)
        except ValueError:
            continue

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                n_trials = sum(1 for line in f if line.strip())
        except Exception:
            continue

        try:
            rel_parts = log_path.relative_to(runs_dir).parts
            opt_id = rel_parts[0]
            # Index all candidate task names from intermediate directory components
            for part in rel_parts[1:-2]:
                if part and part != "None":
                    index[(opt_id, part, seed)] = n_trials
            # Also index immediate parent of seed if valid
            parent_name = log_path.parent.parent.name
            if parent_name and parent_name != "None":
                index[(opt_id, parent_name, seed)] = n_trials
        except Exception:
            pass

    return index


def evaluate_task_status(
    parsed: Dict[str, Any],
    runs_dir: Path,
    runs_index: Optional[Dict[Tuple[str, str, int], int]] = None,
) -> Tuple[str, int]:
    """Determines whether a task is COMPLETE (>= 100 trials), TRUNCATED, or MISSING in O(1) time."""
    opt_id = parsed["optimizer_id"]
    seed = int(parsed["seed"])
    task_raw = parsed["task"]
    task_clean = Path(task_raw).name
    expected = parsed.get("n_trials", 100)

    # 1. Fast index lookup (O(1) in-memory dictionary)
    if runs_index is not None:
        candidate_keys = [
            (opt_id, task_clean, seed),
            (opt_id, task_raw, seed),
            (opt_id, task_clean.split("_")[-1] if "_" in task_clean else task_clean, seed),
        ]
        for key in candidate_keys:
            if key in runs_index:
                n_trials = runs_index[key]
                return ("COMPLETE" if n_trials >= expected else "TRUNCATED"), n_trials
        return "MISSING", 0

    # 2. Fast direct deterministic path check if index not provided
    direct_candidates = [
        runs_dir / opt_id / task_raw / str(seed) / "trial_logs.jsonl",
        runs_dir / opt_id / "blackbox" / task_clean / str(seed) / "trial_logs.jsonl",
        runs_dir / opt_id / task_clean / str(seed) / "trial_logs.jsonl",
    ]
    for p in direct_candidates:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    n_trials = sum(1 for line in f if line.strip())
                return ("COMPLETE" if n_trials >= expected else "TRUNCATED"), n_trials
            except Exception:
                return "MISSING", 0

    # 3. Fallback: build index on the fly and look up
    cached_index = build_suite_runs_index(runs_dir)
    return evaluate_task_status(parsed, runs_dir, runs_index=cached_index)


def audit_batch(
    batch_def: Dict[str, Any],
    results_base: Path = Path("results"),
    runs_base: Path = Path("runs"),
    runs_index: Optional[Dict[Tuple[str, str, int], int]] = None,
) -> Dict[str, Any]:
    """Audits task completeness for a specific batch definition."""
    suite = batch_def["suite"]
    b_id = batch_def["batch_id"]
    start = batch_def["start"]
    end = batch_def["end"]

    suite_results_dir = results_base / f"sweep_{suite}_proximity"
    suite_runs_dir = runs_base / f"sweep_{suite}_proximity"
    task_file = suite_results_dir / "tasks.txt"

    if not task_file.is_file():
        return {
            "batch_id": b_id,
            "suite": suite,
            "start": start,
            "end": end,
            "total": end - start + 1,
            "complete": 0,
            "truncated": 0,
            "missing": end - start + 1,
            "failed_tasks": [],
            "status": "MISSING_TASK_FILE",
        }

    with open(task_file, "r", encoding="utf-8") as f:
        all_lines = [line.strip() for line in f if line.strip()]

    actual_end = min(end, len(all_lines))
    total_in_batch = actual_end - start + 1 if actual_end >= start else 0

    # Build suite runs index once if not provided
    if runs_index is None:
        runs_index = build_suite_runs_index(suite_runs_dir)

    complete_count = 0
    truncated_count = 0
    missing_count = 0
    failed_tasks = []

    for idx in range(start, actual_end + 1):
        raw_cmd = all_lines[idx - 1]
        parsed = parse_task_line(raw_cmd)
        status, n_found = evaluate_task_status(parsed, suite_runs_dir, runs_index=runs_index)

        if status == "COMPLETE":
            complete_count += 1
        elif status == "TRUNCATED":
            truncated_count += 1
            failed_tasks.append({
                "task_idx": idx,
                "raw_cmd": raw_cmd,
                "status": "TRUNCATED",
                "trials_found": n_found,
            })
        else:
            missing_count += 1
            failed_tasks.append({
                "task_idx": idx,
                "raw_cmd": raw_cmd,
                "status": "MISSING",
                "trials_found": 0,
            })

    if complete_count == total_in_batch and total_in_batch > 0:
        batch_status = "COMPLETE"
    elif complete_count > 0:
        batch_status = "PARTIAL"
    else:
        batch_status = "NOT_STARTED"

    return {
        "batch_id": b_id,
        "suite": suite,
        "start": start,
        "end": end,
        "total": total_in_batch,
        "complete": complete_count,
        "truncated": truncated_count,
        "missing": missing_count,
        "failed_tasks": failed_tasks,
        "status": batch_status,
    }


def generate_rerun_files(
    report: Dict[str, Any],
    results_base: Path = Path("results"),
    chunk_size: int = 200,
    concurrency: int = 25,
) -> Dict[str, Path]:
    """Generates tasks_rerun.txt and submit_rerun_all.sh for incomplete tasks in a batch or suite."""
    suite = report["suite"]
    failed = report["failed_tasks"]
    suite_dir = results_base / f"sweep_{suite}_proximity"
    suite_dir.mkdir(parents=True, exist_ok=True)

    rerun_tasks_file = suite_dir / f"tasks_rerun_batch{report['batch_id']}.txt"
    with open(rerun_tasks_file, "w", encoding="utf-8") as f:
        for item in failed:
            f.write(item["raw_cmd"] + "\n")

    # Generate rerun dispatcher script
    total_rerun = len(failed)
    sbatch_runner = f"scripts/submit_sweep_{suite}_proximity_array.sbatch"
    dispatcher_file = suite_dir / f"submit_rerun_batch{report['batch_id']}.sh"

    script_content = f"""#!/bin/bash
# Auto-generated rerun dispatcher for Suite: {suite} (Batch {report['batch_id']})
# Total rerun tasks: {total_rerun}
set -e

TASK_FILE="{rerun_tasks_file}"
SBATCH_FILE="{sbatch_runner}"
TOTAL_TASKS={total_rerun}
CHUNK_SIZE={chunk_size}
CONCURRENCY={concurrency}

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "No failed tasks to rerun!"
    exit 0
fi

echo "=================================================="
echo "Submitting Rerun for {suite} - Batch {report['batch_id']} ($TOTAL_TASKS tasks)"
echo "Chunk Size: $CHUNK_SIZE (%$CONCURRENCY concurrency)"
echo "=================================================="

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    [ $end -gt $TOTAL_TASKS ] && end=$TOTAL_TASKS
    JOB_ID=$(sbatch --parsable --array=${{start}}-${{end}}%${{CONCURRENCY}} "$SBATCH_FILE")
    echo "Submitted Rerun Chunk (${{start}}-${{end}} / ${{TOTAL_TASKS}}) -> Job ID: ${{JOB_ID}}"
done
"""
    with open(dispatcher_file, "w", encoding="utf-8") as f:
        f.write(script_content)

    os.chmod(dispatcher_file, 0o755)

    return {
        "rerun_tasks_file": rerun_tasks_file,
        "dispatcher_file": dispatcher_file,
    }


def audit_all_batches(
    results_base: Path = Path("results"),
    runs_base: Path = Path("runs"),
    suite_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Audits all batches defined in DEFAULT_BATCHES with single-pass suite indexing."""
    suite_indices: Dict[str, Dict[Tuple[str, str, int], int]] = {}
    reports = []
    for b_def in DEFAULT_BATCHES:
        suite = b_def["suite"]
        if suite_filter and suite_filter != "all" and suite != suite_filter:
            continue
        if suite not in suite_indices:
            suite_runs_dir = runs_base / f"sweep_{suite}_proximity"
            suite_indices[suite] = build_suite_runs_index(suite_runs_dir)

        rep = audit_batch(
            b_def,
            results_base=results_base,
            runs_base=runs_base,
            runs_index=suite_indices[suite],
        )
        reports.append(rep)
    return reports


def print_audit_summary(reports: List[Dict[str, Any]]) -> None:
    """Prints a structured ASCII report of batch statuses."""
    print("\n" + "=" * 90)
    print(f"{'BATCH AUDIT REPORT: MULTI-SUITE PROXIMITY SWEEPS':^90}")
    print("=" * 90)
    print(f"{'Batch':<7} | {'Suite':<20} | {'Range':<11} | {'Status':<12} | {'Complete':<10} | {'Trunc':<6} | {'Miss':<6} | {'Pct':<6}")
    print("-" * 90)

    grand_total = 0
    grand_complete = 0
    grand_trunc = 0
    grand_miss = 0

    for r in reports:
        b_id = f"Batch {r['batch_id']}"
        suite = r["suite"]
        rng = f"{r['start']}-{r['end']}"
        status = r["status"]
        comp = f"{r['complete']}/{r['total']}"
        trunc = str(r["truncated"])
        miss = str(r["missing"])
        pct = f"{(r['complete']/r['total'])*100:5.1f}%" if r["total"] > 0 else "0.0%"

        grand_total += r["total"]
        grand_complete += r["complete"]
        grand_trunc += r["truncated"]
        grand_miss += r["missing"]

        print(f"{b_id:<7} | {suite:<20} | {rng:<11} | {status:<12} | {comp:<10} | {trunc:<6} | {miss:<6} | {pct:<6}")

    print("-" * 90)
    overall_pct = (grand_complete / grand_total * 100) if grand_total > 0 else 0.0
    print(f"{'TOTAL':<7} | {'All 3 Suites':<20} | {'18,600':<11} | {'-':<12} | {f'{grand_complete}/{grand_total}':<10} | {grand_trunc:<6} | {grand_miss:<6} | {overall_pct:5.1f}%")
    print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit batch completion and generate rerun scripts.")
    parser.add_argument("--suite", choices=["yahpo_rbv2_ranger", "yahpo_rbv2_super", "hpobench_ml", "all"], default="all")
    parser.add_argument("--results-dir", type=str, default="results", help="Base results directory")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Base runs directory")
    parser.add_argument("--generate-reruns", action="store_true", help="Generate targeted rerun task files and submit scripts for incomplete batches")
    parser.add_argument("--json-out", type=str, default=None, help="Save machine-readable summary to JSON file")
    args = parser.parse_args()

    results_base = Path(args.results_dir)
    runs_base = Path(args.runs_dir)

    reports = audit_all_batches(results_base=results_base, runs_base=runs_base, suite_filter=args.suite)
    print_audit_summary(reports)

    if args.generate_reruns:
        print("Generating rerun task lists and dispatchers for incomplete batches...")
        for r in reports:
            if r["failed_tasks"]:
                files = generate_rerun_files(r, results_base=results_base)
                print(f"  [Batch {r['batch_id']} - {r['suite']}] -> {files['rerun_tasks_file']} ({len(r['failed_tasks'])} tasks)")
                print(f"                                   -> {files['dispatcher_file']}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        print(f"Saved JSON audit report to {args.json_out}")


if __name__ == "__main__":
    main()
