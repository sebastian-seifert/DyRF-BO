#!/usr/bin/env python3
"""Unified Orchestrator for CARP-S and Custom Statistical Analyses on YAHPO Sweeps.

Executes:
1. Standard CARP-S reports via `carps.analysis.generate_report`
2. Custom statistical comparison scorecards and markdown reports vs dual baselines
Across YAHPO Gym suites rbv2_ranger and rbv2_super.
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

from scripts.compute_yahpo_rbv2_ranger_sweep_analysis import (
    run_yahpo_rbv2_ranger_analysis,
    export_yahpo_rbv2_ranger_analysis,
)
from scripts.compute_yahpo_rbv2_super_sweep_analysis import (
    run_yahpo_rbv2_super_analysis,
    export_yahpo_rbv2_super_analysis,
)

YAHPO_SUITES: List[str] = ["ranger", "super"]

TASK_DISPATCH = {
    "ranger": (run_yahpo_rbv2_ranger_analysis, export_yahpo_rbv2_ranger_analysis),
    "super": (run_yahpo_rbv2_super_analysis, export_yahpo_rbv2_super_analysis),
}


def list_available_tasks() -> List[str]:
    """Returns canonical list of individual atomic tasks."""
    return [
        "ranger_carps",
        "ranger_custom",
        "super_carps",
        "super_custom",
    ]


def resolve_tasks(task_specs: Sequence[str]) -> List[str]:
    """Expands task aliases into individual atomic task IDs.

    Supported aliases:
    - 'all': runs all ranger and super tasks (CARP-S + custom)
    - 'ranger_all' / 'ranger': all rbv2_ranger tasks (carps + custom)
    - 'super_all' / 'super': all rbv2_super tasks (carps + custom)
    - 'carps_all': all CARP-S report tasks across suites
    - 'custom_all': all custom statistical scorecard tasks across suites
    - 'ranger_carps' / 'carps_ranger': ranger CARP-S task
    - 'ranger_custom' / 'custom_ranger': ranger custom analysis task
    - 'super_carps' / 'carps_super': super CARP-S task
    - 'super_custom' / 'custom_super': super custom analysis task
    """
    all_tasks = list_available_tasks()
    alias_map = {
        "all": all_tasks,
        "ranger_all": ["ranger_carps", "ranger_custom"],
        "ranger": ["ranger_carps", "ranger_custom"],
        "super_all": ["super_carps", "super_custom"],
        "super": ["super_carps", "super_custom"],
        "carps_all": ["ranger_carps", "super_carps"],
        "custom_all": ["ranger_custom", "super_custom"],
        "ranger_carps": ["ranger_carps"],
        "carps_ranger": ["ranger_carps"],
        "ranger_custom": ["ranger_custom"],
        "custom_ranger": ["ranger_custom"],
        "super_carps": ["super_carps"],
        "carps_super": ["super_carps"],
        "super_custom": ["super_custom"],
        "custom_super": ["super_custom"],
    }

    resolved: List[str] = []
    for spec in task_specs:
        s = spec.strip().lower()
        if s in alias_map:
            for t in alias_map[s]:
                if t not in resolved:
                    resolved.append(t)
        else:
            raise ValueError(f"Unknown task or alias: '{spec}'. Available aliases: {list(alias_map.keys())}")

    return resolved


def get_default_paths(task_name: str, base_dir: Path | None = None) -> tuple[Path, Path]:
    """Computes default input log path and output directory for a given task."""
    root = base_dir if base_dir is not None else Path(PROJECT_ROOT)
    results_dir = root / "results"

    if "ranger" in task_name:
        suite = "ranger"
    elif "super" in task_name:
        suite = "super"
    else:
        raise ValueError(f"Cannot determine suite for task: {task_name}")

    sweep_dir = results_dir / f"sweep_yahpo_rbv2_{suite}"
    in_path = sweep_dir / "logs.parquet"

    if "carps" in task_name:
        out_dir = sweep_dir / "carps_report"
    else:
        out_dir = sweep_dir / "analysis"

    return in_path, out_dir


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


def run_yahpo_analysis_task(
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
        if "carps" in task_name:
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

        elif "custom" in task_name:
            df = _load_dataframe(in_file)

            suite = "ranger" if "ranger" in task_name else "super"
            run_fn, export_fn = TASK_DISPATCH[suite]
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
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cost_col: str = "trial_value__cost_inc",
    holm_scope: str = "joint",
    dry_run: bool = False,
    fail_fast: bool = False,
    continue_on_error: bool = False,
    base_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    """Runs a batch of resolved tasks sequentially."""
    resolved = resolve_tasks(tasks)
    print(f"Orchestrator planned {len(resolved)} task(s): {', '.join(resolved)}")

    should_fail_fast = fail_fast and not continue_on_error

    results = []
    for t in resolved:
        res = run_yahpo_analysis_task(
            task_name=t,
            input_path=input_path,
            output_dir=output_dir,
            cost_col=cost_col,
            holm_scope=holm_scope,
            dry_run=dry_run,
            base_dir=base_dir,
        )
        results.append(res)
        if res["status"] == "FAILED" and should_fail_fast:
            print(f"[HALT] Aborting remaining tasks due to failure in '{t}'.")
            break

    return results


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for unified YAHPO analysis runner."""
    parser = argparse.ArgumentParser(
        description="Unified YAHPO Analysis Runner: Standard CARP-S reports and Custom Statistical Scorecards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help="Tasks or aliases to run (e.g. 'all', 'ranger_all', 'super_all', 'carps_all', 'custom_all', 'ranger_custom').",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=YAHPO_SUITES,
        help="Convenience filter to run tasks for specific suites (e.g. --suites ranger super).",
    )
    parser.add_argument(
        "--input",
        "--data-path",
        dest="input",
        type=str,
        default=None,
        help="Override input logs path (.parquet or .csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
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
        "--fail-fast",
        action="store_true",
        default=False,
        help="Halt execution immediately on the first task failure.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue executing remaining tasks if one task fails.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    requested_tasks = list(args.tasks)
    if args.suites:
        requested_tasks = [f"{s}_all" for s in args.suites]

    results = run_orchestrator(
        tasks=requested_tasks,
        input_path=args.input,
        output_dir=args.output_dir,
        cost_col=args.cost_col,
        holm_scope=args.holm_scope,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
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
