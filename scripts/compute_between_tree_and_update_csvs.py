#!/usr/bin/env python3
"""Cluster-optimized CLI to compute Between-Tree Disagreement and update cube CSVs.

Evaluates 'hutter_between' (Between-Tree Disagreement) from raw Parquet experiment runs,
aggregates all 8 canonical slices of the 3D OLAP cube, and updates existing slice CSV files
by appending/replacing the 'hutter_between' rows while strictly preserving columns and schemas.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dyrf_bo.extrapolation_uq.cube_analysis import (
    aggregate_between_tree_slices,
    evaluate_single_run_between_tree,
)

# Canonical key columns identifying rows for each cube slice
SLICE_KEY_COLUMNS: Dict[str, List[str]] = {
    "main_global": ["approach"],
    "stratum_global": ["stratum_scope", "approach"],
    "main_global_by_dimension": ["dimension", "approach"],
    "main_global_by_surrogate": ["surrogate", "approach"],
    "stratum_by_dimension": ["stratum_scope", "dimension", "approach"],
    "stratum_by_surrogate": ["stratum_scope", "surrogate", "approach"],
    "main_global_rf_by_dimension": ["surrogate", "dimension", "approach"],
    "full_cube": ["stratum_scope", "surrogate", "dimension", "approach"],
}


def _process_single_run(file_path: Path) -> Optional[pd.DataFrame]:
    """Worker function to read and evaluate between-tree metrics for a single raw Parquet run."""
    try:
        df = pd.read_parquet(file_path)
        eval_df = evaluate_single_run_between_tree(df)
        return eval_df
    except Exception as exc:
        print(f"[ERROR] Failed processing {file_path.name}: {exc}", file=sys.stderr)
        return None


def update_slice_csv(
    csv_path: Path,
    new_rows_df: pd.DataFrame,
    key_cols: Optional[List[str]] = None,
    backup: bool = True,
) -> None:
    """Safely append or replace rows in an existing slice CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the slice CSV file to update.
    new_rows_df : pd.DataFrame
        DataFrame containing new rows (e.g. approach='hutter_between').
    key_cols : list[str] | None
        Columns uniquely identifying rows to replace. If None, defaults to ['approach'].
    backup : bool, default=True
        Whether to create a .bak copy before modifying the CSV.
    """
    if key_cols is None:
        key_cols = ["approach"]

    if csv_path.exists():
        if backup:
            bak_path = csv_path.with_name(f"{csv_path.name}.bak")
            shutil.copy2(csv_path, bak_path)

        orig_df = pd.read_csv(csv_path)

        # Remove existing rows matching the incoming approach to avoid duplicates
        if "approach" in orig_df.columns and "approach" in new_rows_df.columns:
            target_approaches = set(new_rows_df["approach"].unique())
            cleaned_orig = orig_df[~orig_df["approach"].isin(target_approaches)].copy()
        else:
            cleaned_orig = orig_df.copy()

        # Reorder and align columns to match existing CSV exactly
        common_cols = [c for c in orig_df.columns if c in new_rows_df.columns]
        missing_in_new = [c for c in orig_df.columns if c not in new_rows_df.columns]
        aligned_new = new_rows_df[common_cols].copy()
        for c in missing_in_new:
            aligned_new[c] = None

        aligned_new = aligned_new[orig_df.columns]
        combined_df = pd.concat([cleaned_orig, aligned_new], ignore_index=True)
    else:
        combined_df = new_rows_df

    combined_df.to_csv(csv_path, index=False)
    print(f"[INFO] Updated {csv_path.name} ({len(combined_df)} rows total)")


def run_between_tree_update(
    input_dir: Path,
    csv_dir: Path,
    n_jobs: int = 1,
    force: bool = False,
    cache_path: Optional[Path] = None,
    backup: bool = True,
) -> None:
    """Execute between-tree evaluation and update all existing slice CSV files."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    if cache_path is None:
        cache_path = csv_dir / "between_tree_evaluations.parquet"

    runs_df: Optional[pd.DataFrame] = None

    if cache_path.exists() and not force:
        print(f"[INFO] Loading cached evaluations from {cache_path}...")
        try:
            runs_df = pd.read_parquet(cache_path)
            print(f"[INFO] Loaded {len(runs_df)} evaluation rows from cache.")
        except Exception as exc:
            print(f"[WARN] Failed to load cache ({exc}), recomputing...", file=sys.stderr)

    if runs_df is None:
        parquet_files = sorted(list(input_dir.glob("*.parquet")) + list(input_dir.glob("**/*.parquet")))
        parquet_files = sorted(list(set(parquet_files)))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {input_dir}")

        print(f"[INFO] Found {len(parquet_files)} parquet files. Processing with n_jobs={n_jobs}...")

        results: List[pd.DataFrame] = []
        if n_jobs > 1 and len(parquet_files) > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                for res in executor.map(_process_single_run, parquet_files):
                    if res is not None:
                        results.append(res)
        else:
            for f in parquet_files:
                res = _process_single_run(f)
                if res is not None:
                    results.append(res)

        if not results:
            raise RuntimeError("No parquet files were successfully evaluated.")

        runs_df = pd.concat(results, ignore_index=True)

        print(f"[INFO] Saving between-tree evaluations to {cache_path}...")
        runs_df.to_parquet(cache_path, index=False)

    print("[INFO] Computing 3D OLAP Cube Slices for between-tree disagreement...")
    slices = aggregate_between_tree_slices(runs_df)

    # Update each slice CSV in csv_dir
    for slice_name, slice_df in slices.items():
        csv_file = csv_dir / f"{slice_name}.csv"
        key_cols = SLICE_KEY_COLUMNS.get(slice_name, ["approach"])
        update_slice_csv(csv_file, slice_df, key_cols=key_cols, backup=backup)

    print("[INFO] Between-tree update completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Between-Tree Disagreement from raw Parquets and update cube CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw extrapolation UQ Parquet run files.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
        help="Directory containing slice CSVs (e.g. main_global.csv).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel processes for reading parquet files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cached evaluations exist.",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help="Optional custom path to cached evaluations parquet.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable automatic .bak creation for existing CSVs.",
    )

    args = parser.parse_args()
    run_between_tree_update(
        input_dir=args.input_dir,
        csv_dir=args.csv_dir,
        n_jobs=args.n_jobs,
        force=args.force,
        cache_path=args.cache_file,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
