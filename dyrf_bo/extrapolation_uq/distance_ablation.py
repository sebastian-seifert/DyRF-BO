"""Distance metric ablation study module comparing Euclidean (d_norm) vs Chebyshev (d_inf).

Evaluates whether axis-aligned Chebyshev (L_infinity) projection distances better align
with axis-aligned Random Forest surrogate epistemic uncertainties than standard Euclidean
(L_2) normalized projection distances across dimensions, strategies, and strata.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .metrics_suite import spearman_rank_correlation

PARQUET_FILENAME_PATTERN = re.compile(
    r"^extrapolation_(?P<function_name>.+?)_d(?P<dimension>\d+)_n(?P<n_train>\d+)_(?P<sampling_strategy>[a-zA-Z0-9_]+?)_s(?P<seed>\d+)\.parquet$"
)


def parse_parquet_metadata(filepath: str | Path) -> Dict[str, Any]:
    """Parse experiment configuration metadata from standard Parquet filename.

    Parameters
    ----------
    filepath : str | Path
        Filename or path matching pattern
        'extrapolation_{function}_d{dimension}_n{n_train}_{strategy}_s{seed}.parquet'.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'function_name': str
        - 'dimension': int
        - 'n_train': int
        - 'sampling_strategy': str
        - 'seed': int

    Raises
    ------
    ValueError
        If the filename does not strictly conform to the expected format.
    """
    fname = Path(filepath).name
    match = PARQUET_FILENAME_PATTERN.match(fname)
    if not match:
        raise ValueError(
            f"Invalid filename format: '{fname}'. "
            "Expected 'extrapolation_{function}_d{dimension}_n{n_train}_{strategy}_s{seed}.parquet'."
        )

    gd = match.groupdict()
    return {
        "function_name": str(gd["function_name"]),
        "dimension": int(gd["dimension"]),
        "n_train": int(gd["n_train"]),
        "sampling_strategy": str(gd["sampling_strategy"]),
        "seed": int(gd["seed"]),
    }


def _compute_cliffs_delta(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    """Compute Cliff's Delta effect size between two continuous distributions.

    delta = (#(x_i > y_j) - #(x_i < y_j)) / (n_x * n_y)
    """
    arr_x = np.asarray(x, dtype=np.float64).ravel()
    arr_y = np.asarray(y, dtype=np.float64).ravel()

    arr_x = arr_x[np.isfinite(arr_x)]
    arr_y = arr_y[np.isfinite(arr_y)]

    nx = len(arr_x)
    ny = len(arr_y)

    if nx == 0 or ny == 0:
        return 0.0

    diff_matrix = np.subtract.outer(arr_x, arr_y)
    greater = np.count_nonzero(diff_matrix > 1e-12)
    less = np.count_nonzero(diff_matrix < -1e-12)

    delta = float(greater - less) / float(nx * ny)
    return float(np.clip(delta, -1.0, 1.0))


def _compute_paired_wilcoxon(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    zero_method: str = "pratt",
) -> Tuple[float, float]:
    """Compute paired two-sided Wilcoxon signed-rank test with fallback."""
    arr_x = np.asarray(x, dtype=np.float64).ravel()
    arr_y = np.asarray(y, dtype=np.float64).ravel()

    valid_mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    arr_x = arr_x[valid_mask]
    arr_y = arr_y[valid_mask]

    n = len(arr_x)
    if n == 0:
        return (float("nan"), float("nan"))

    diff = arr_x - arr_y
    if np.all(np.abs(diff) < 1e-12):
        return (0.0, 1.0)

    if n == 1:
        return (0.0, 1.0)

    try:
        res = stats.wilcoxon(arr_x, arr_y, zero_method=zero_method, alternative="two-sided")
        stat = float(res.statistic)
        pval = float(res.pvalue)
        if math.isnan(pval):
            pval = 1.0
        return (stat, pval)
    except Exception:
        return (0.0, 1.0)


def compute_run_distance_ablation(
    df_or_path: pd.DataFrame | str | Path,
) -> Dict[str, Any]:
    """Compute distance metric ablation correlation metrics for a single experiment run.

    Compares Spearman rank correlation of Euclidean normalized distance (d_norm) vs
    Chebyshev distance (d_inf) with surrogate epistemic uncertainties U_SLCB and U_PLCB.

    Parameters
    ----------
    df_or_path : pd.DataFrame | str | Path
        Evaluations DataFrame or Path to Parquet file.

    Returns
    -------
    dict[str, Any]
        Dictionary containing global and per-stratum correlations and differences.
    """
    metadata: Dict[str, Any] = {}
    if isinstance(df_or_path, (str, Path)):
        fpath = Path(df_or_path)
        df = pd.read_parquet(fpath)
        try:
            metadata = parse_parquet_metadata(fpath)
        except ValueError:
            metadata = {}
    elif isinstance(df_or_path, pd.DataFrame):
        df = df_or_path
    else:
        raise TypeError(f"Expected DataFrame, str, or Path; got {type(df_or_path)}")

    results: Dict[str, Any] = dict(metadata)

    d_norm = df["d_norm"].to_numpy(dtype=np.float64) if "d_norm" in df.columns else np.array([])
    d_inf = df["d_inf"].to_numpy(dtype=np.float64) if "d_inf" in df.columns else np.array([])
    u_slcb = df["u_slcb"].to_numpy(dtype=np.float64) if "u_slcb" in df.columns else np.array([])
    u_plcb = df["u_plcb"].to_numpy(dtype=np.float64) if "u_plcb" in df.columns else np.array([])

    # Global Spearman correlations
    spearman_dist_norm_slcb = spearman_rank_correlation(d_norm, u_slcb)
    spearman_dist_inf_slcb = spearman_rank_correlation(d_inf, u_slcb)
    diff_slcb = spearman_dist_inf_slcb - spearman_dist_norm_slcb

    spearman_dist_norm_plcb = spearman_rank_correlation(d_norm, u_plcb)
    spearman_dist_inf_plcb = spearman_rank_correlation(d_inf, u_plcb)
    diff_plcb = spearman_dist_inf_plcb - spearman_dist_norm_plcb

    results["spearman_dist_norm_slcb"] = spearman_dist_norm_slcb
    results["spearman_dist_inf_slcb"] = spearman_dist_inf_slcb
    results["diff_slcb"] = diff_slcb

    results["spearman_dist_norm_plcb"] = spearman_dist_norm_plcb
    results["spearman_dist_inf_plcb"] = spearman_dist_inf_plcb
    results["diff_plcb"] = diff_plcb

    # Per-stratum breakdown if stratum column is present
    if "stratum" in df.columns:
        strata_arr = df["stratum"].to_numpy()
        for s in range(4):
            mask = strata_arr == s
            if np.count_nonzero(mask) >= 2:
                s_d_norm = d_norm[mask]
                s_d_inf = d_inf[mask]
                s_u_slcb = u_slcb[mask]
                s_u_plcb = u_plcb[mask]

                s_norm_slcb = spearman_rank_correlation(s_d_norm, s_u_slcb)
                s_inf_slcb = spearman_rank_correlation(s_d_inf, s_u_slcb)
                s_diff_slcb = s_inf_slcb - s_norm_slcb

                s_norm_plcb = spearman_rank_correlation(s_d_norm, s_u_plcb)
                s_inf_plcb = spearman_rank_correlation(s_d_inf, s_u_plcb)
                s_diff_plcb = s_inf_plcb - s_norm_plcb
            else:
                s_norm_slcb = 0.0
                s_inf_slcb = 0.0
                s_diff_slcb = 0.0
                s_norm_plcb = 0.0
                s_inf_plcb = 0.0
                s_diff_plcb = 0.0

            results[f"stratum_{s}_spearman_dist_norm_slcb"] = s_norm_slcb
            results[f"stratum_{s}_spearman_dist_inf_slcb"] = s_inf_slcb
            results[f"stratum_{s}_diff_slcb"] = s_diff_slcb

            results[f"stratum_{s}_spearman_dist_norm_plcb"] = s_norm_plcb
            results[f"stratum_{s}_spearman_dist_inf_plcb"] = s_inf_plcb
            results[f"stratum_{s}_diff_plcb"] = s_diff_plcb

    return results


def aggregate_distance_ablation(
    records: List[Dict[str, Any]] | pd.DataFrame,
    tol: float = 1e-9,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate distance ablation records across dimensions, strategies, and strata.

    Parameters
    ----------
    records : list[dict[str, Any]] | pd.DataFrame
        List of run result dictionaries or converted DataFrame.
    tol : float, default=1e-9
        Tolerance for classifying wins, ties, and losses.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - df_slices: Grouped scorecard across (dimension, strategy) with Wilcoxon tests,
          Cliff's delta, and Win/Tie/Loss counts.
        - df_strata: Mean and SEM metrics grouped by stratum {0, 1, 2, 3}.
    """
    slice_cols = [
        "dimension",
        "sampling_strategy",
        "n_experiments",
        # SLCB
        "dist_norm_slcb_mean", "dist_norm_slcb_sem",
        "dist_inf_slcb_mean", "dist_inf_slcb_sem",
        "diff_slcb_mean", "pvalue_slcb", "cliffs_delta_slcb",
        "wins_slcb", "ties_slcb", "losses_slcb",
        # PLCB
        "dist_norm_plcb_mean", "dist_norm_plcb_sem",
        "dist_inf_plcb_mean", "dist_inf_plcb_sem",
        "diff_plcb_mean", "pvalue_plcb", "cliffs_delta_plcb",
        "wins_plcb", "ties_plcb", "losses_plcb",
    ]

    strata_cols = [
        "stratum",
        "dist_norm_slcb_mean", "dist_norm_slcb_sem",
        "dist_inf_slcb_mean", "dist_inf_slcb_sem",
        "diff_slcb_mean", "diff_slcb_sem",
        "dist_norm_plcb_mean", "dist_norm_plcb_sem",
        "dist_inf_plcb_mean", "dist_inf_plcb_sem",
        "diff_plcb_mean", "diff_plcb_sem",
    ]

    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif isinstance(records, list):
        df = pd.DataFrame(records) if records else pd.DataFrame()
    else:
        df = pd.DataFrame()

    if df.empty:
        return pd.DataFrame(columns=slice_cols), pd.DataFrame(columns=strata_cols)

    def _process_slice(sub: pd.DataFrame, dim_val: Any, strat_val: Any) -> Dict[str, Any]:
        n = len(sub)
        row: Dict[str, Any] = {
            "dimension": dim_val,
            "sampling_strategy": strat_val,
            "n_experiments": n,
        }

        for method in ["slcb", "plcb"]:
            norm_col = f"spearman_dist_norm_{method}"
            inf_col = f"spearman_dist_inf_{method}"
            diff_col = f"diff_{method}"

            norm_vals = sub[norm_col].dropna().to_numpy(dtype=np.float64)
            inf_vals = sub[inf_col].dropna().to_numpy(dtype=np.float64)
            diff_vals = sub[diff_col].dropna().to_numpy(dtype=np.float64) if diff_col in sub.columns else (inf_vals - norm_vals)

            k = len(diff_vals)
            norm_mean = float(np.mean(norm_vals)) if len(norm_vals) > 0 else float("nan")
            norm_sem = float(np.std(norm_vals, ddof=1) / np.sqrt(k)) if k > 1 else 0.0

            inf_mean = float(np.mean(inf_vals)) if len(inf_vals) > 0 else float("nan")
            inf_sem = float(np.std(inf_vals, ddof=1) / np.sqrt(k)) if k > 1 else 0.0

            diff_mean = float(np.mean(diff_vals)) if k > 0 else float("nan")

            stat, pval = _compute_paired_wilcoxon(inf_vals, norm_vals)
            cliffs_d = _compute_cliffs_delta(inf_vals, norm_vals)

            # Win = Chebyshev > Euclidean (diff > tol)
            wins = int(np.count_nonzero(diff_vals > tol))
            losses = int(np.count_nonzero(diff_vals < -tol))
            ties = int(np.count_nonzero(np.abs(diff_vals) <= tol))

            row[f"dist_norm_{method}_mean"] = norm_mean
            row[f"dist_norm_{method}_sem"] = norm_sem
            row[f"dist_inf_{method}_mean"] = inf_mean
            row[f"dist_inf_{method}_sem"] = inf_sem
            row[f"diff_{method}_mean"] = diff_mean
            row[f"pvalue_{method}"] = pval
            row[f"cliffs_delta_{method}"] = cliffs_d
            row[f"wins_{method}"] = wins
            row[f"ties_{method}"] = ties
            row[f"losses_{method}"] = losses

        return row

    # 1. Grouped slices
    slice_rows: List[Dict[str, Any]] = []

    unique_dims: List[int] = []
    for d in df["dimension"].dropna().unique():
        try:
            unique_dims.append(int(d))
        except (ValueError, TypeError):
            continue
    dimensions = sorted(list(set(unique_dims)))
    strategies = sorted([str(s) for s in df["sampling_strategy"].dropna().unique()])
    numeric_dims = pd.to_numeric(df["dimension"], errors="coerce")

    # (dimension, strategy)
    for d in dimensions:
        for strat in strategies:
            sub = df[(numeric_dims == d) & (df["sampling_strategy"] == strat)]
            if not sub.empty:
                slice_rows.append(_process_slice(sub, int(d), strat))

    # (dimension, "All")
    for d in dimensions:
        sub = df[numeric_dims == d]
        if not sub.empty:
            slice_rows.append(_process_slice(sub, int(d), "All"))

    # ("All", strategy)
    for strat in strategies:
        sub = df[df["sampling_strategy"] == strat]
        if not sub.empty:
            slice_rows.append(_process_slice(sub, "All", strat))

    # ("All", "All") grand total
    slice_rows.append(_process_slice(df, "All", "All"))

    df_slices = pd.DataFrame(slice_rows)[slice_cols]

    # 2. Strata summary
    strata_rows: List[Dict[str, Any]] = []
    for s in range(4):
        s_row: Dict[str, Any] = {"stratum": s}
        for method in ["slcb", "plcb"]:
            norm_col = f"stratum_{s}_spearman_dist_norm_{method}"
            inf_col = f"stratum_{s}_spearman_dist_inf_{method}"
            diff_col = f"stratum_{s}_diff_{method}"

            if norm_col in df.columns and inf_col in df.columns:
                norm_vals = df[norm_col].dropna().to_numpy(dtype=np.float64)
                inf_vals = df[inf_col].dropna().to_numpy(dtype=np.float64)
                diff_vals = (
                    df[diff_col].dropna().to_numpy(dtype=np.float64)
                    if diff_col in df.columns
                    else (inf_vals - norm_vals)
                )

                k = len(norm_vals)
                s_row[f"dist_norm_{method}_mean"] = float(np.mean(norm_vals)) if k > 0 else float("nan")
                s_row[f"dist_norm_{method}_sem"] = (
                    float(np.std(norm_vals, ddof=1) / np.sqrt(k)) if k > 1 else 0.0
                )
                s_row[f"dist_inf_{method}_mean"] = float(np.mean(inf_vals)) if k > 0 else float("nan")
                s_row[f"dist_inf_{method}_sem"] = (
                    float(np.std(inf_vals, ddof=1) / np.sqrt(k)) if k > 1 else 0.0
                )
                s_row[f"diff_{method}_mean"] = float(np.mean(diff_vals)) if k > 0 else float("nan")
                s_row[f"diff_{method}_sem"] = (
                    float(np.std(diff_vals, ddof=1) / np.sqrt(k)) if k > 1 else 0.0
                )
            else:
                s_row[f"dist_norm_{method}_mean"] = float("nan")
                s_row[f"dist_norm_{method}_sem"] = float("nan")
                s_row[f"dist_inf_{method}_mean"] = float("nan")
                s_row[f"dist_inf_{method}_sem"] = float("nan")
                s_row[f"diff_{method}_mean"] = float("nan")
                s_row[f"diff_{method}_sem"] = float("nan")

        strata_rows.append(s_row)

    df_strata = pd.DataFrame(strata_rows)[strata_cols]

    return df_slices, df_strata
