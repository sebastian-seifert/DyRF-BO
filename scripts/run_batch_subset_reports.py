#!/usr/bin/env python3
"""
Batch Runner for CARP-S Subset Reports across multiple sweep subfolders.

Iteratively executes `scripts/generate_subset_report.py` for each specified subfolder
in `results/` using a defined subset of optimizers and report name.

Example usage:
    python scripts/run_batch_subset_reports.py
    python scripts/run_batch_subset_reports.py --dry-run
    python scripts/run_batch_subset_reports.py --subfolders sweep_bbob_dim2 sweep_bbob_dim4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_SUBFOLDERS: List[str] = [
    "sweep_bbob_dim2",
    "sweep_bbob_dim4",
    "sweep_bbob_dim8",
    "sweep_bbob_dim16",
    "sweep_bbob_dim32",
    "sweep_yahpo_rbv2_ranger",
    "sweep_yahpo_rbv2_super",
    "sweep_bbsubset_test_big_comparison",
    "sweep_hpobench_ml",
]

DEFAULT_OPTIMIZERS: List[str] = [
    "SMAC3_HPOFacade_lcb",
    "SMAC3_HPOFacade_ei",
    "SMAC20_ProximityAC_LCB",
    "SMAC20_ProximityBC_LCB",
    "SMAC20_Entropy_LCB",
]

DEFAULT_REPORT_NAME: str = "subset_report"


def build_command(
    parquet_path: Path,
    optimizers: List[str],
    report_name: str,
    python_bin: str = sys.executable,
    script_path: Path = Path("scripts/generate_subset_report.py"),
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Construct the command line argument list for generate_subset_report.py."""
    cmd = [
        python_bin,
        str(script_path),
        str(parquet_path),
        "--optimizers",
        *optimizers,
        "--report-name",
        report_name,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def run_batch_subset_reports(
    subfolders: Optional[List[str]] = None,
    results_dir: Path | str = Path("results"),
    optimizers: Optional[List[str]] = None,
    report_name: str = DEFAULT_REPORT_NAME,
    dry_run: bool = False,
    skip_missing: bool = True,
    python_bin: str = sys.executable,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Iterate over sweep subfolders and execute generate_subset_report.py for each.

    Returns a dict mapping subfolder name to its outcome status:
    'SUCCESS', 'FAILED', 'SKIPPED_MISSING', or 'DRY_RUN'.
    """
    target_subfolders = subfolders or DEFAULT_SUBFOLDERS
    target_optimizers = optimizers or DEFAULT_OPTIMIZERS
    base_dir = Path(results_dir)

    print("==================================================")
    print(" CARP-S Batch Subset Report Runner")
    print("==================================================")
    print(f"Base Directory:  {base_dir.resolve()}")
    print(f"Total Folders:   {len(target_subfolders)}")
    print(f"Report Name:     {report_name}")
    print(f"Optimizers ({len(target_optimizers)}): {', '.join(target_optimizers)}")
    print(f"Dry Run:         {dry_run}")
    print(f"Skip Missing:    {skip_missing}")
    print("==================================================\n")

    statuses: Dict[str, str] = {}

    for idx, sub in enumerate(target_subfolders, 1):
        folder_path = base_dir / sub
        parquet_path = folder_path / "logs.parquet"

        print(f"[{idx}/{len(target_subfolders)}] Target: {folder_path} ...")

        if not parquet_path.exists():
            if not dry_run:
                if skip_missing:
                    print(f"  [WARNING] Missing {parquet_path} -> Skipping.")
                    statuses[sub] = "SKIPPED_MISSING"
                    continue
                else:
                    raise FileNotFoundError(f"Missing required parquet file: {parquet_path}")
            else:
                print(f"  [NOTE] {parquet_path} does not exist locally yet.")

        cmd = build_command(
            parquet_path=parquet_path,
            optimizers=target_optimizers,
            report_name=report_name,
            python_bin=python_bin,
            extra_args=extra_args,
        )

        cmd_str = " ".join(cmd)
        if dry_run:
            print(f"  [DRY-RUN] Command:\n  {cmd_str}\n")
            statuses[sub] = "DRY_RUN"
            continue

        print(f"  Executing: {cmd_str}")
        start_t = time.time()
        try:
            res = subprocess.run(cmd, check=True)
            elapsed = time.time() - start_t
            print(f"  [SUCCESS] Completed in {elapsed:.1f}s\n")
            statuses[sub] = "SUCCESS"
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_t
            print(f"  [FAILED] Return code {e.returncode} after {elapsed:.1f}s\n", file=sys.stderr)
            statuses[sub] = "FAILED"

    print("==================================================")
    print(" Batch Execution Summary")
    print("==================================================")
    for sub, status in statuses.items():
        print(f"  - {sub:35s}: {status}")
    print("==================================================")

    return statuses


def main():
    parser = argparse.ArgumentParser(
        description="Run generate_subset_report.py sequentially across multiple sweep directories."
    )
    parser.add_argument(
        "--subfolders",
        "-s",
        nargs="+",
        default=DEFAULT_SUBFOLDERS,
        help="Subfolder names in results directory to process.",
    )
    parser.add_argument(
        "--optimizers",
        "-o",
        nargs="+",
        default=DEFAULT_OPTIMIZERS,
        help="Optimizer identifiers to filter and include in the reports.",
    )
    parser.add_argument(
        "--report-name",
        "-r",
        type=str,
        default=DEFAULT_REPORT_NAME,
        help="Report subfolder name (default: 'subset_report').",
    )
    parser.add_argument(
        "--results-dir",
        "-d",
        type=str,
        default="results",
        help="Base results directory (default: 'results').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    parser.add_argument(
        "--fail-fast",
        dest="skip_missing",
        action="store_false",
        help="Fail immediately if logs.parquet is missing instead of skipping.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Path to the python interpreter.",
    )

    args, extra_args = parser.parse_known_args()

    statuses = run_batch_subset_reports(
        subfolders=args.subfolders,
        results_dir=args.results_dir,
        optimizers=args.optimizers,
        report_name=args.report_name,
        dry_run=args.dry_run,
        skip_missing=args.skip_missing,
        python_bin=args.python_bin,
        extra_args=extra_args,
    )

    if any(s == "FAILED" for s in statuses.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
