#!/usr/bin/env python3
"""
Results Parser & Analysis Reporter for Epistemic OOD Detection Sweep.

Outputs:
  - Table 1: Normal benchmark functions (41 functions) aggregated and averaged across
             each dimension (1D through 15D).
  - Table 2: Special benchmark functions (10 functions) listed individually.

Both tables are exported in CSV and GitHub Flavored Markdown formats.
"""

from __future__ import annotations

import os
import sys
import glob
import json
from typing import Optional, Tuple

import pandas as pd
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from synthetic_functions import (
    get_all_normal_functions,
    get_special_functions,
)

DEFAULT_RESULTS_DIR = "results/epistemic_ood_sweep"
DEFAULT_OUTPUT_DIR = "results/epistemic_ood_sweep/summary"

METRIC_COLUMNS = [
    "auroc",
    "fpr95",
    "aupr",
    "spearman",
    "aurc",
    "oracle_aurc",
    "jsd",
    "mi",
    "nlpd",
    "brier",
]


def format_dataframe_as_markdown(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Formats DataFrame as a GitHub Markdown table cleanly."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        # Fallback manual markdown generator
        formatted_df = df.copy()
        for col in formatted_df.select_dtypes(include=[np.number]).columns:
            formatted_df[col] = formatted_df[col].map(lambda x: float_format.format(x) if pd.notnull(x) else "")

        headers = [str(col) for col in formatted_df.columns]
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_lines = [
            "| " + " | ".join([str(val) for val in row]) + " |"
            for row in formatted_df.values
        ]
        return "\n".join([header_line, sep_line] + body_lines)


def parse_epistemic_ood_sweep_results(
    results_dir: str = DEFAULT_RESULTS_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Parses result JSON files from results_dir, separates normal and special functions,
    and produces Table 1 (Normal by Dim) and Table 2 (Special Individually).

    Parameters:
        results_dir: Path containing result JSON files.
        output_dir: Directory where summary CSV and Markdown files are saved.

    Returns:
        (table1_df, table2_df) tuple of pandas DataFrames.
    """
    json_pattern = os.path.join(results_dir, "*.json")
    json_files = sorted(glob.glob(json_pattern))

    if not json_files:
        print(f"No result JSON files found in '{results_dir}'.")
        return None, None

    normal_names = set(get_all_normal_functions().keys())
    special_names = set(get_special_functions().keys())

    records = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
                records.append(data)
        except Exception as e:
            print(f"Warning: Failed to parse '{jf}': {e}", file=sys.stderr)

    if not records:
        print("No valid records found in JSON files.")
        return None, None

    df = pd.DataFrame(records)
    os.makedirs(output_dir, exist_ok=True)

    present_metrics = [m for m in METRIC_COLUMNS if m in df.columns]

    # Partition into Normal and Special functions
    df_normal = df[df["func_name"].isin(normal_names)].copy()
    df_special = df[df["func_name"].isin(special_names)].copy()

    # If any function was not recognized in either set, fallback partition
    unknown = df[~df["func_name"].isin(normal_names | special_names)]
    if not unknown.empty:
        print(f"Warning: Found {len(unknown)} records with unrecognized function names: {unknown['func_name'].unique()}")

    table1: Optional[pd.DataFrame] = None
    table2: Optional[pd.DataFrame] = None

    # =========================================================================
    # Table 1: Normal Functions Aggregated by Dimension
    # =========================================================================
    if not df_normal.empty:
        # Group by dimension, gap type, and approach across all normal functions & seeds
        table1 = (
            df_normal.groupby(["dim", "gap_type", "approach"])[present_metrics]
            .mean()
            .reset_index()
            .sort_values(by=["dim", "gap_type", "approach"])
        )

        t1_csv = os.path.join(output_dir, "table1_normal_functions_by_dim.csv")
        t1_md = os.path.join(output_dir, "table1_normal_functions_by_dim.md")

        table1.to_csv(t1_csv, index=False)

        md_content_1 = (
            "# Table 1: Normal Synthetic Benchmark Functions Aggregated by Dimension\n\n"
            f"Aggregated across 41 normal benchmark functions (1D to 15D).\n\n"
            + format_dataframe_as_markdown(table1.round(4))
            + "\n"
        )
        with open(t1_md, "w") as f:
            f.write(md_content_1)

        print(f"Saved Table 1 (Normal by Dim) to '{t1_csv}' and '{t1_md}'")

    # =========================================================================
    # Table 2: Special Functions Listed Individually
    # =========================================================================
    if not df_special.empty:
        # Group by func_name (and dim, gap_type, approach) averaging across seeds
        group_cols = ["func_name", "dim", "gap_type", "approach"]
        if "dim" not in df_special.columns:
            group_cols = ["func_name", "gap_type", "approach"]

        table2 = (
            df_special.groupby(group_cols)[present_metrics]
            .mean()
            .reset_index()
            .sort_values(by=["dim", "func_name", "gap_type", "approach"] if "dim" in group_cols else ["func_name", "gap_type", "approach"])
        )

        t2_csv = os.path.join(output_dir, "table2_special_functions_individual.csv")
        t2_md = os.path.join(output_dir, "table2_special_functions_individual.md")

        table2.to_csv(t2_csv, index=False)

        md_content_2 = (
            "# Table 2: Special Named Benchmark Functions Listed Individually\n\n"
            "Averaged across seeds for each of the 10 special named benchmark functions.\n\n"
            + format_dataframe_as_markdown(table2.round(4))
            + "\n"
        )
        with open(t2_md, "w") as f:
            f.write(md_content_2)

        print(f"Saved Table 2 (Special Individually) to '{t2_csv}' and '{t2_md}'")

    return table1, table2


def main():
    res_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    parse_epistemic_ood_sweep_results(results_dir=res_dir, output_dir=out_dir)


if __name__ == "__main__":
    main()
