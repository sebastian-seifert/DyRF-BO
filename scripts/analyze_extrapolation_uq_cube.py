#!/usr/bin/env python3
"""Extrapolation UQ 3D Cube Analysis CLI.

Processes raw single-run Parquet files, evaluates metrics per run (across all 10 approaches
and 5 strata), and generates the 8 canonical slices of the 3D OLAP cube
(Stratum x Surrogate x Dimension) with rigorous summary statistics.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq.cube_analysis import aggregate_3d_cube, evaluate_single_run_df


def _process_single_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Worker function to read and evaluate a single raw Parquet run."""
    try:
        df = pd.read_parquet(file_path)
        eval_df = evaluate_single_run_df(df)
        return eval_df
    except Exception as exc:
        print(f"[ERROR] Failed processing {file_path.name}: {exc}", file=sys.stderr)
        return None


def run_cube_analysis(
    input_dir: Path,
    output_dir: Path,
    n_jobs: int = 1,
    force: bool = False,
    cache_path: Optional[Path] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if cache_path is None:
        cache_path = output_dir / "per_run_evaluations.parquet"

    per_run_df: Optional[pd.DataFrame] = None

    if cache_path.exists() and not force:
        print(f"[INFO] Loading cached evaluations from {cache_path}...")
        try:
            per_run_df = pd.read_parquet(cache_path)
            print(f"[INFO] Loaded {len(per_run_df)} evaluation rows from cache.")
        except Exception as exc:
            print(f"[WARN] Failed to load cache ({exc}), recomputing...", file=sys.stderr)

    if per_run_df is None:
        parquet_files = sorted(list(input_dir.glob("*.parquet")) + list(input_dir.glob("**/*.parquet")))
        # Deduplicate files
        parquet_files = sorted(list(set(parquet_files)))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {input_dir}")

        print(f"[INFO] Found {len(parquet_files)} parquet files. Processing with n_jobs={n_jobs}...")

        results: List[pd.DataFrame] = []
        if n_jobs > 1 and len(parquet_files) > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                for res in executor.map(_process_single_file, parquet_files):
                    if res is not None:
                        results.append(res)
        else:
            for f in parquet_files:
                res = _process_single_file(f)
                if res is not None:
                    results.append(res)

        if not results:
            raise RuntimeError("No parquet files were successfully evaluated.")

        per_run_df = pd.concat(results, ignore_index=True)

        print(f"[INFO] Saving per-run evaluations to {cache_path}...")
        per_run_df.to_parquet(cache_path, index=False)
        csv_cache = output_dir / "per_run_evaluations.csv"
        per_run_df.to_csv(csv_cache, index=False)

    print("[INFO] Computing 3D OLAP Cube Slices...")
    slices = aggregate_3d_cube(per_run_df)

    for slice_name, slice_df in slices.items():
        slice_csv = output_dir / f"{slice_name}.csv"
        slice_df.to_csv(slice_csv, index=False)
        print(f"[INFO] Exported {slice_name} ({len(slice_df)} rows) -> {slice_csv.name}")

    print("[INFO] Analysis completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Analyze Extrapolation UQ 3D Cube")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw extrapolation UQ Parquet run files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where evaluation cache and 8 slice CSVs will be saved.",
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
        help="Force recomputation even if per_run_evaluations cache exists.",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help="Optional custom path to cached per_run_evaluations.parquet.",
    )

    args = parser.parse_args()
    run_cube_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        force=args.force,
        cache_path=args.cache_file,
    )


if __name__ == "__main__":
    main()
