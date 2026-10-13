#!/usr/bin/env python3
"""Identify incomplete or missing tasks from BBOB benchmark sweeps using runs/ or telemetry.

Parses a task manifest (default: results/sweep_bbob_full_comparison/tasks_all.txt)
and verifies whether each task completed successfully by checking:
  1. The run directory (baserundir/optimizer_id/BBOB/bbob/dim/fid/iid/seed/trial_logs.jsonl)
  2. Fallback telemetry files (e.g. telemetry_{opt}_{task}_seed{seed}.json)

Outputs summary statistics and optionally exports a missing_tasks.txt manifest to re-submit
only the incomplete/missing tasks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class TaskSpec:
    line_number: int
    raw_line: str
    optimizer_id: str
    task_id: str
    dimension: int
    seed: int
    n_trials: int
    baserundir: Path
    expected_run_dir: Optional[Path]
    telemetry_path: Optional[Path]


def parse_task_line(line: str, line_number: int) -> Optional[TaskSpec]:
    """Parses a command line string into a TaskSpec."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # 1. Optimizer ID
    opt_match = re.search(r"optimizer_id=([^\s]+)", line)
    optimizer_id = opt_match.group(1) if opt_match else "unknown"

    # 2. Task ID (+task=BBOB/cfg_2_1_0)
    task_match = re.search(r"\+task=([^\s]+)", line)
    task_id = task_match.group(1) if task_match else "unknown"

    # 3. Seed
    seed_match = re.search(r"\bseed=(\d+)", line)
    seed = int(seed_match.group(1)) if seed_match else 1

    # 4. Trials
    trials_match = re.search(r"n_trials=(\d+)", line)
    n_trials = int(trials_match.group(1)) if trials_match else 100

    # 5. Base run directory
    base_match = re.search(r"baserundir=([^\s]+)", line)
    baserundir_str = base_match.group(1) if base_match else "runs/sweep_bbob_full_comparison"
    baserundir = Path(baserundir_str)

    # 6. Telemetry path
    telem_match = re.search(r"\+\+optimizer\.telemetry_path=([^\s]+)", line)
    telemetry_path = Path(telem_match.group(1)) if telem_match else None

    # Construct expected CARPS run directory and extract dimension
    # Format: baserundir / optimizer_id / BBOB / bbob / {dim} / {fid} / {iid} / {seed}
    cfg_match = re.search(r"cfg_(\d+)_(\d+)_(\d+)", task_id)
    if cfg_match:
        dim_str, fid, iid = cfg_match.groups()
        dimension = int(dim_str)
        expected_run_dir = baserundir / optimizer_id / "BBOB" / "bbob" / dim_str / fid / iid / str(seed)
    else:
        # Check alternative format like bbob/2/1/0
        alt_match = re.search(r"bbob/(\d+)/(\d+)/(\d+)", task_id)
        if alt_match:
            dim_str, fid, iid = alt_match.groups()
            dimension = int(dim_str)
            expected_run_dir = baserundir / optimizer_id / "BBOB" / "bbob" / dim_str / fid / iid / str(seed)
        else:
            dimension = -1
            expected_run_dir = baserundir / optimizer_id / task_id / str(seed)

    return TaskSpec(
        line_number=line_number,
        raw_line=line,
        optimizer_id=optimizer_id,
        task_id=task_id,
        dimension=dimension,
        seed=seed,
        n_trials=n_trials,
        baserundir=baserundir,
        expected_run_dir=expected_run_dir,
        telemetry_path=telemetry_path,
    )


def count_trials_in_trial_logs(trial_logs_path: Path) -> int:
    """Counts non-empty trial rows in trial_logs.jsonl."""
    if not trial_logs_path.is_file():
        return 0
    count = 0
    try:
        with open(trial_logs_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        return 0
    return count


def count_trials_in_telemetry(telemetry_path: Path) -> int:
    """Counts recorded trials in telemetry JSON file."""
    if not telemetry_path.is_file():
        return 0
    try:
        with open(telemetry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            history = data.get("history", [])
            return len(history)
    except Exception:
        return 0
    return 0


def is_task_complete(spec: TaskSpec, required_fraction: float = 1.0) -> bool:
    """Checks if a task ran through to completion."""
    target_trials = int(spec.n_trials * required_fraction)

    # 1. Primary check: trial_logs.jsonl in expected_run_dir
    if spec.expected_run_dir:
        trial_logs = spec.expected_run_dir / "trial_logs.jsonl"
        trials_found = count_trials_in_trial_logs(trial_logs)
        if trials_found >= target_trials:
            return True

        # Check alternative nested run structure (smac3_output runhistory fallback)
        runhistory_files = list(spec.expected_run_dir.glob("**/runhistory.json"))
        if runhistory_files:
            try:
                for rh_file in runhistory_files:
                    with open(rh_file, "r", encoding="utf-8") as f:
                        rh_data = json.load(f)
                    data_items = rh_data.get("data", [])
                    if len(data_items) >= target_trials:
                        return True
            except Exception:
                pass

    # 2. Fallback check: telemetry JSON file
    if spec.telemetry_path:
        trials_in_telem = count_trials_in_telemetry(spec.telemetry_path)
        if trials_in_telem >= target_trials:
            return True

    return False


def audit_tasks_by_dimension(
    manifest_path: Path | str,
    required_fraction: float = 1.0,
) -> Dict[int, Dict[str, Any]]:
    """Audits task manifest grouping results by dimensionality."""
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    from collections import defaultdict
    dim_tasks = defaultdict(list)
    dim_incomplete = defaultdict(list)

    with open(manifest, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            spec = parse_task_line(line, line_number=idx)
            if spec is None:
                continue
            dim = spec.dimension
            dim_tasks[dim].append(spec)
            if not is_task_complete(spec, required_fraction=required_fraction):
                dim_incomplete[dim].append(spec)

    results: Dict[int, Dict[str, Any]] = {}
    for dim in sorted(dim_tasks.keys()):
        total = len(dim_tasks[dim])
        incomplete = len(dim_incomplete[dim])
        completed = total - incomplete
        rate = (completed / total * 100.0) if total > 0 else 0.0
        results[dim] = {
            "total": total,
            "completed": completed,
            "incomplete": incomplete,
            "completion_rate": rate,
            "incomplete_tasks": dim_incomplete[dim],
        }

    return results


def find_incomplete_tasks(
    manifest_path: Path | str,
    required_fraction: float = 1.0,
    dimensions: Optional[Sequence[int]] = None,
) -> List[TaskSpec]:
    """Finds all tasks in manifest that did not run to completion, optionally filtered by dimensions."""
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    target_dims = set(dimensions) if dimensions else None
    incomplete_tasks: List[TaskSpec] = []
    with open(manifest, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            spec = parse_task_line(line, line_number=idx)
            if spec is None:
                continue
            if target_dims is not None and spec.dimension not in target_dims:
                continue
            if not is_task_complete(spec, required_fraction=required_fraction):
                incomplete_tasks.append(spec)

    return incomplete_tasks


def count_total_tasks(manifest_path: Path | str) -> int:
    """Counts total tasks in a manifest."""
    manifest = Path(manifest_path)
    count = 0
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.strip().startswith("#"):
                count += 1
    return count


def filter_manifest(
    manifest_path: Path | str,
    output_path: Path | str,
    required_fraction: float = 1.0,
    dimensions: Optional[Sequence[int]] = None,
) -> Tuple[int, int]:
    """Filters manifest to only keep incomplete tasks, writing them to output_path."""
    manifest = Path(manifest_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    incomplete = find_incomplete_tasks(
        manifest,
        required_fraction=required_fraction,
        dimensions=dimensions,
    )
    with open(out_file, "w", encoding="utf-8") as f:
        for task in incomplete:
            f.write(f"{task.raw_line}\n")

    return len(incomplete), count_total_tasks(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify incomplete tasks from runs/ and manifests grouped by dimensionality.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="results/sweep_bbob_full_comparison/tasks_all.txt",
        help="Path to tasks manifest file.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of dimensions to audit (e.g. --dimensions 2, or --dimensions 2 4 8 16 32).",
    )
    parser.add_argument(
        "--output-missing",
        type=str,
        default="results/sweep_bbob_full_comparison/missing_tasks.txt",
        help="Path to output file for missing tasks.",
    )
    parser.add_argument(
        "--export-by-dimension",
        action="store_true",
        help="Export separate missing manifest per dimension (e.g. missing_tasks_dim2.txt).",
    )
    parser.add_argument(
        "--required-fraction",
        type=float,
        default=1.0,
        help="Fraction of n_trials required to mark a task complete (default: 1.0).",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"Error: Manifest {manifest} does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Auditing tasks in {manifest} grouped by dimensionality...")
    dim_audit = audit_tasks_by_dimension(manifest, required_fraction=args.required_fraction)

    # Filter to requested dimensions if provided
    selected_dims = args.dimensions if args.dimensions is not None else sorted(dim_audit.keys())

    print("\n" + "=" * 70)
    print(f"{'Dimension':<12} | {'Total':<10} | {'Completed':<10} | {'Incomplete':<10} | {'Progress':<10}")
    print("-" * 70)

    grand_total = 0
    grand_completed = 0
    grand_incomplete = 0

    all_selected_incomplete: List[TaskSpec] = []

    for d in selected_dims:
        if d in dim_audit:
            stats = dim_audit[d]
            grand_total += stats["total"]
            grand_completed += stats["completed"]
            grand_incomplete += stats["incomplete"]
            all_selected_incomplete.extend(stats["incomplete_tasks"])

            status_str = f"{stats['completion_rate']:.1f}%"
            dim_label = f"{d}D" if d > 0 else "Unknown"
            print(f"{dim_label:<12} | {stats['total']:<10} | {stats['completed']:<10} | {stats['incomplete']:<10} | {status_str:<10}")

            if args.export_by_dimension and args.output_missing:
                out_base = Path(args.output_missing)
                stem = out_base.stem
                dim_out = out_base.parent / f"{stem}_dim{d}.txt"
                dim_out.parent.mkdir(parents=True, exist_ok=True)
                with open(dim_out, "w", encoding="utf-8") as f:
                    for task in stats["incomplete_tasks"]:
                        f.write(f"{task.raw_line}\n")
                print(f"  --> Exported {len(stats['incomplete_tasks'])} incomplete tasks to {dim_out}")

    print("=" * 70)
    grand_rate = (grand_completed / grand_total * 100.0) if grand_total > 0 else 0.0
    print(f"{'OVERALL':<12} | {grand_total:<10} | {grand_completed:<10} | {grand_incomplete:<10} | {grand_rate:.1f}%")
    print("=" * 70 + "\n")

    if args.output_missing:
        out_path = Path(args.output_missing)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for task in all_selected_incomplete:
                f.write(f"{task.raw_line}\n")
        print(f"Exported {len(all_selected_incomplete)} incomplete tasks to {out_path}")


if __name__ == "__main__":
    main()
