#!/usr/bin/env python3
"""Unified Orchestrator for CARP-S and Custom Statistical Analyses on BBOB Sweeps.

Executes:
1. Standard CARP-S reports via `carps.analysis.generate_report`
2. Custom statistical comparison scorecards and markdown reports vs dual baselines
Across BBOB dimensions 2, 4, 8, 16, 32, and full sweep.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.compute_bbob_dim2_sweep_analysis import run_bbob_dim2_analysis, export_dim2_analysis
from scripts.compute_bbob_dim4_sweep_analysis import run_bbob_dim4_analysis, export_dim4_analysis
from scripts.compute_bbob_dim8_sweep_analysis import run_bbob_dim8_analysis, export_dim8_analysis
from scripts.compute_bbob_dim16_sweep_analysis import run_bbob_dim16_analysis, export_dim16_analysis
from scripts.compute_bbob_dim32_sweep_analysis import run_bbob_dim32_analysis, export_dim32_analysis
from scripts.compute_bbob_full_sweep_analysis import (
    run_bbob_statistical_analysis,
    export_analysis_results,
)

BBOB_DIMENSIONS: List[int] = [2, 4, 8, 16, 32]

DIM_HANDLERS = {
    2: (run_bbob_dim2_analysis, export_dim2_analysis),
    4: (run_bbob_dim4_analysis, export_dim4_analysis),
    8: (run_bbob_dim8_analysis, export_dim8_analysis),
    16: (run_bbob_dim16_analysis, export_dim16_analysis),
    32: (run_bbob_dim32_analysis, export_dim32_analysis),
}


def list_available_tasks() -> List[str]:
    """Returns canonical list of individual atomic tasks."""
    tasks = []
    for d in BBOB_DIMENSIONS:
        tasks.append(f"carps_dim{d}")
        tasks.append(f"custom_dim{d}")
    tasks.append("carps_full")
    tasks.append("custom_full")
    return tasks


def resolve_tasks(task_specs: Sequence[str]) -> List[str]:
    """Expands task aliases (e.g. 'all', 'carps_all', 'dim4') into individual task IDs."""
    all_tasks = list_available_tasks()
    resolved: List[str] = []

    for spec in task_specs:
        s = spec.strip().lower()
        if s == "all":
            for t in all_tasks:
                if t not in resolved:
                    resolved.append(t)
        elif s == "carps_all":
            for t in all_tasks:
                if t.startswith("carps_") and t not in resolved:
                    resolved.append(t)
        elif s == "custom_all":
            for t in all_tasks:
                if t.startswith("custom_") and t not in resolved:
                    resolved.append(t)
        elif s == "full":
            for t in ["carps_full", "custom_full"]:
                if t not in resolved:
                    resolved.append(t)
        elif s.startswith("dim") and s[3:].isdigit():
            d = int(s[3:])
            if d in BBOB_DIMENSIONS:
                for t in [f"carps_dim{d}", f"custom_dim{d}"]:
                    if t not in resolved:
                        resolved.append(t)
            else:
                raise ValueError(f"Unknown dimension in task alias '{spec}'. Expected one of {BBOB_DIMENSIONS}")
        elif s in all_tasks:
            if s not in resolved:
                resolved.append(s)
        else:
            raise ValueError(f"Unknown task or alias: '{spec}'. Available tasks: {all_tasks}")

    return resolved


def get_default_paths(task_name: str, base_dir: Path | None = None) -> tuple[Path, Path]:
    """Computes default input log path and output directory for a given task."""
    root = base_dir if base_dir is not None else Path(PROJECT_ROOT)
    results_dir = root / "results"

    if "full" in task_name:
        sweep_dir = results_dir / "sweep_bbob_full_comparison"
        in_path = sweep_dir / "logs.parquet"
        if task_name.startswith("carps_"):
            out_dir = sweep_dir / "carps_report"
        else:
            out_dir = sweep_dir / "analysis"
        return in_path, out_dir

    for d in BBOB_DIMENSIONS:
        if f"dim{d}" in task_name:
            sweep_dir = results_dir / f"sweep_bbob_dim{d}"
            in_path = sweep_dir / "logs.parquet"
            if task_name.startswith("carps_"):
                out_dir = sweep_dir / "carps_report"
            else:
                out_dir = sweep_dir / "analysis"
            return in_path, out_dir

    raise ValueError(f"Cannot determine default paths for task: {task_name}")


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Loads parquet or fallback CSV into DataFrame."""
    target_path = path
    if not target_path.exists():
        alt = target_path.with_suffix(".csv") if target_path.suffix == ".parquet" else target_path.with_suffix(".parquet")
        if alt.exists():
            target_path = alt
        else:
            raise FileNotFoundError(f"Input file not found at {path} or {alt}")

    if target_path.suffix == ".parquet":
        try:
            return pd.read_parquet(target_path)
        except Exception:
            alt_csv = target_path.with_suffix(".csv")
            if alt_csv.exists():
                return pd.read_csv(alt_csv)
            raise
    return pd.read_csv(target_path)


def run_bbob_analysis_task(
    task_name: str,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cost_col: str = "trial_value__cost_inc",
    holm_scope: str = "joint",
    dry_run: bool = False,
    base_dir: Path | None = None,
) -> Dict[str, Any]:
    """Executes a single CARP-S or custom analysis task."""
    default_in, default_out = get_default_paths(task_name, base_dir=base_dir)
    in_file = Path(input_path) if input_path is not None else default_in
    out_path = Path(output_dir) if output_dir is not None else default_out

    if dry_run:
        print(f"[DRY-RUN] Task '{task_name}': Input='{in_file}' -> Output='{out_path}'")
        return {
            "task": task_name,
            "status": "DRY_RUN",
            "input": str(in_file),
            "output": str(out_path),
        }

    print(f"\n=======================================================")
    print(f"Executing Task: {task_name}")
    print(f"Input:  {in_file}")
    print(f"Output: {out_path}")
    print(f"=======================================================")

    out_path.mkdir(parents=True, exist_ok=True)

    try:
        if task_name.startswith("carps_"):
            # CARP-S Report Generation
            from carps.analysis.generate_report import generate_report

            target_in = in_file
            if not target_in.exists():
                alt = target_in.with_suffix(".csv") if target_in.suffix == ".parquet" else target_in.with_suffix(".parquet")
                if alt.exists():
                    target_in = alt
                else:
                    raise FileNotFoundError(f"CARP-S input logs not found at {in_file}")

            report_name = f"{task_name}_report"
            generate_report(
                result_path=str(target_in),
                report_dir=str(out_path),
                report_name=report_name,
                normalize_results=True,
            )
            print(f"[SUCCESS] Generated CARP-S report for {task_name} in {out_path}")

        elif task_name.startswith("custom_"):
            df = _load_dataframe(in_file)

            if task_name == "custom_full":
                results = run_bbob_statistical_analysis(df, cost_col=cost_col, holm_scope=holm_scope)
                export_analysis_results(results, str(out_path))
            else:
                dim_str = task_name.replace("custom_dim", "")
                d = int(dim_str)
                run_fn, export_fn = DIM_HANDLERS[d]
                results = run_fn(df, cost_col=cost_col, holm_scope=holm_scope)
                export_fn(results, str(out_path))

            print(f"[SUCCESS] Completed custom statistical analysis for {task_name} in {out_path}")
        else:
            raise ValueError(f"Unknown task type for task '{task_name}'")

        return {
            "task": task_name,
            "status": "SUCCESS",
            "input": str(in_file),
            "output": str(out_path),
        }

    except Exception as e:
        print(f"[ERROR] Task {task_name} failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return {
            "task": task_name,
            "status": "FAILED",
            "error": str(e),
            "input": str(in_file),
            "output": str(out_path),
        }


def run_orchestrator(
    tasks: Sequence[str],
    cost_col: str = "trial_value__cost_inc",
    holm_scope: str = "joint",
    dry_run: bool = False,
    continue_on_error: bool = False,
    base_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    """Runs a batch of resolved tasks sequentially."""
    resolved = resolve_tasks(tasks)
    print(f"Orchestrator planned {len(resolved)} task(s): {', '.join(resolved)}")

    results = []
    for t in resolved:
        res = run_bbob_analysis_task(
            task_name=t,
            cost_col=cost_col,
            holm_scope=holm_scope,
            dry_run=dry_run,
            base_dir=base_dir,
        )
        results.append(res)
        if res["status"] == "FAILED" and not continue_on_error:
            print(f"[HALT] Aborting remaining tasks due to failure in '{t}'.")
            break

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified BBOB Analysis Runner: Standard CARP-S reports and Custom Statistical Scorecards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help="Tasks or aliases to run (e.g. 'all', 'carps_all', 'custom_all', 'dim2', 'custom_dim4').",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        choices=BBOB_DIMENSIONS,
        help="Convenience filter to run both CARP-S and custom analyses for specific dimensions (e.g. --dims 2 4).",
    )
    parser.add_argument(
        "--cost-col",
        type=str,
        default="trial_value__cost_inc",
        help="Incumbent cost column in logs DataFrame (default: trial_value__cost_inc).",
    )
    parser.add_argument(
        "--holm-scope",
        type=str,
        default="joint",
        choices=["joint", "per_control"],
        help="Holm-Bonferroni correction scope: 'joint' (Scope B) or 'per_control' (Scope A).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned task execution without computing reports.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue executing remaining tasks if one task fails.",
    )
    args = parser.parse_args()

    requested_tasks = list(args.tasks)
    if args.dims:
        requested_tasks = [f"dim{d}" for d in args.dims]

    results = run_orchestrator(
        tasks=requested_tasks,
        cost_col=args.cost_col,
        holm_scope=args.holm_scope,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )

    failures = [r for r in results if r["status"] == "FAILED"]
    if failures:
        print(f"\n[SUMMARY] {len(failures)} task(s) failed.")
        return 1
    print(f"\n[SUMMARY] All {len(results)} task(s) completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
