#!/usr/bin/env python3
"""Aggregator and Scorecard Generator for Lower-Quantile UQ Sweep.

Reads raw evaluation Parquets and summary JSONs from lower-quantile UQ sweep runs,
evaluates distance-uncertainty correlation metrics:
- Spearman rho(d_norm, U)
- Spearman rho(d_inf, U)
- Delta(rho)
- Normalized rank monotonicity: (1 + rho) / 2

Generates comprehensive scorecards:
1. Absolute Performance Leaderboard (Rank, Method, Category, rho_norm, rho_inf, Monotonicity)
2. Pairwise Comparative Scorecards comparing:
   - Proximity A (u_prox_a_lower)
   - Proximity B (u_prox_b_lower)
   - Proximity AC (u_prox_ac_lower)
   - Proximity BC (u_prox_bc_lower)
   - Proximity LCB (u_plcb_lower)
   against all non-proximity baselines:
   - SLCB / Hutter Total (u_slcb / u_hutter_total)
   - Hutter Between / Epistemic (u_hutter_between)
   - Hutter Within / Aleatoric (u_hutter_within)
   - Shaker Epistemic (u_shaker_epistemic)
   - Shaker Total (u_shaker_total)
   - Shaker Mutual Information (shaker_mi)
   - Shaker Total Entropy (shaker_total_entropy)
   - RF-FIRE (u_rf_fire_lower)
   as well as internal proximity variants (AC vs A, BC vs B, B vs A, BC vs AC, PLCB vs A).
3. Surrogate Architecture Breakdown (performance across smac_default, mature, shallow, coarse, breiman).

Stratified across:
- Low-D (D <= 5)
- High-D (D >= 16)
- All Dimensions

Outputs:
- table_lower_quantile_scorecard.csv / .md
- table_lower_quantile_leaderboard.csv / .md
- table_lower_quantile_by_surrogate.csv / .md
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RESULTS_DIR = "results/lower_quantile_sweep"
DEFAULT_RAW_DIR = "results/lower_quantile_sweep/raw"
DEFAULT_SUMMARY_DIR = "results/lower_quantile_sweep/summaries"
DEFAULT_OUTPUT_DIR = "results/lower_quantile_sweep/analysis"

PARQUET_PATTERN = re.compile(
    r"^extrapolation_(?P<func>.+?)_d(?P<dim>\d+)_n(?P<n_train>\d+)_(?P<strat>[a-zA-Z0-9_]+?)(?:_(?P<surr>[a-zA-Z0-9_]+?))?_s(?P<seed>\d+)\.parquet$"
)
SUMMARY_PATTERN = re.compile(
    r"^summary_(?P<func>.+?)_d(?P<dim>\d+)_n(?P<n_train>\d+)_(?P<strat>[a-zA-Z0-9_]+?)(?:_(?P<surr>[a-zA-Z0-9_]+?))?_s(?P<seed>\d+)\.json$"
)

# Complete candidate column mapping for all evaluated methods
ESTIMATOR_COL_MAP: Dict[str, List[str]] = {
    # Proximity variants (all unweighted tree walks)
    "prox_a": ["u_prox_a_lower", "u_prox_a_unweighted_lower", "u_prox_a_unweighted", "u_prox_a"],
    "prox_b": ["u_prox_b_lower", "u_prox_b"],
    "prox_ac": ["u_prox_ac_lower", "u_prox_ac"],
    "prox_bc": ["u_prox_bc_lower", "u_prox_bc"],
    "plcb": ["u_plcb_lower", "u_plcb", "u_plcb_unweighted_lower", "u_plcb_unweighted"],
    # Non-proximity baselines
    "slcb": ["u_slcb", "u_hutter_total", "slcb", "hutter_total"],
    "hutter_between": ["u_hutter_between", "hutter_between"],
    "hutter_within": ["u_hutter_within", "hutter_within"],
    "shaker": ["u_shaker_epistemic", "shaker_epistemic"],
    "shaker_total": ["u_shaker_total", "shaker_total"],
    "shaker_mi": ["shaker_mi", "u_shaker_mi"],
    "shaker_total_entropy": ["shaker_total_entropy", "u_shaker_total_entropy"],
    "rf_fire": ["u_rf_fire_lower", "u_rf_fire", "rf_fire"],
}

# Complete key candidates in summary JSON files
ESTIMATOR_KEY_MAP: Dict[str, List[str]] = {
    "prox_a": ["u_prox_a_lower", "prox_a", "u_prox_a_unweighted_lower"],
    "prox_b": ["u_prox_b_lower", "prox_b"],
    "prox_ac": ["u_prox_ac_lower", "prox_ac"],
    "prox_bc": ["u_prox_bc_lower", "prox_bc"],
    "plcb": ["u_plcb_lower", "plcb", "u_plcb_unweighted_lower"],
    "slcb": ["u_slcb", "slcb", "u_hutter_total", "hutter_total"],
    "hutter_between": ["u_hutter_between", "hutter_between"],
    "hutter_within": ["u_hutter_within", "hutter_within"],
    "shaker": ["u_shaker_epistemic", "shaker_epistemic"],
    "shaker_total": ["u_shaker_total", "shaker_total"],
    "shaker_mi": ["shaker_mi", "u_shaker_mi"],
    "shaker_total_entropy": ["shaker_total_entropy", "u_shaker_total_entropy"],
    "rf_fire": ["u_rf_fire_lower", "rf_fire"],
}

# Human-readable labels and categories for methods
ESTIMATOR_META: Dict[str, Dict[str, str]] = {
    "prox_a": {"name": "Proximity A (TNS Top-k)", "category": "Proximity (Unweighted)"},
    "prox_b": {"name": "Proximity B (TWQ Continuous)", "category": "Proximity (Unweighted)"},
    "prox_ac": {"name": "Proximity AC (TNS + Density)", "category": "Proximity (Unweighted)"},
    "prox_bc": {"name": "Proximity BC (TWQ + Density)", "category": "Proximity (Unweighted)"},
    "plcb": {"name": "PLCB (Adaptive Floor)", "category": "Proximity (Unweighted)"},
    "slcb": {"name": "SLCB / Hutter Total", "category": "Baseline (Variance)"},
    "hutter_between": {"name": "Hutter Between (Epistemic)", "category": "Baseline (Tree Disagreement)"},
    "hutter_within": {"name": "Hutter Within (Aleatoric)", "category": "Baseline (Leaf Variance)"},
    "shaker": {"name": "Shaker Epistemic", "category": "Baseline (Numerical Integration)"},
    "shaker_total": {"name": "Shaker Total", "category": "Baseline (Numerical Integration)"},
    "shaker_mi": {"name": "Shaker Mutual Information", "category": "Baseline (Information Theoretic)"},
    "shaker_total_entropy": {"name": "Shaker Total Entropy", "category": "Baseline (Information Theoretic)"},
    "rf_fire": {"name": "RF-FIRE Lower", "category": "Baseline (Volume Expansion)"},
}

# Defined pairwise comparisons: (test, ref, label)
COMPARISONS: List[Tuple[str, str, str]] = [
    # Proximity variants vs SLCB (Hutter Total) baseline
    ("prox_a", "slcb", "prox_a vs slcb"),
    ("prox_b", "slcb", "prox_b vs slcb"),
    ("prox_ac", "slcb", "prox_ac vs slcb"),
    ("prox_bc", "slcb", "prox_bc vs slcb"),
    ("plcb", "slcb", "plcb vs slcb"),
    # Proximity variants vs RF-FIRE
    ("prox_a", "rf_fire", "prox_a vs rf_fire"),
    ("prox_b", "rf_fire", "prox_b vs rf_fire"),
    ("prox_ac", "rf_fire", "prox_ac vs rf_fire"),
    ("prox_bc", "rf_fire", "prox_bc vs rf_fire"),
    ("plcb", "rf_fire", "plcb vs rf_fire"),
    # Proximity variants vs Shaker (Epistemic)
    ("prox_a", "shaker", "prox_a vs shaker"),
    ("prox_b", "shaker", "prox_b vs shaker"),
    ("prox_ac", "shaker", "prox_ac vs shaker"),
    ("prox_bc", "shaker", "prox_bc vs shaker"),
    ("plcb", "shaker", "plcb vs shaker"),
    # Proximity variants vs Hutter Between (Epistemic Disagreement)
    ("prox_a", "hutter_between", "prox_a vs hutter_between"),
    ("prox_b", "hutter_between", "prox_b vs hutter_between"),
    ("prox_ac", "hutter_between", "prox_ac vs hutter_between"),
    ("prox_bc", "hutter_between", "prox_bc vs hutter_between"),
    ("plcb", "hutter_between", "plcb vs hutter_between"),
    # Density multiplier C benefit
    ("prox_ac", "prox_a", "prox_ac vs prox_a"),
    ("prox_bc", "prox_b", "prox_bc vs prox_b"),
    # Continuous quantile (B) vs Top-K (A)
    ("prox_b", "prox_a", "prox_b vs prox_a"),
    ("prox_bc", "prox_ac", "prox_bc vs prox_ac"),
    # PLCB vs Proximity A
    ("plcb", "prox_a", "plcb vs prox_a"),
]


def compute_spearman_rho(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Compute Spearman rank correlation between two vectors."""
    arr_x = np.asarray(x, dtype=np.float64).ravel()
    arr_y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(arr_x) & np.isfinite(arr_y)
    arr_x, arr_y = arr_x[valid], arr_y[valid]
    if len(arr_x) < 2:
        return 0.0
    if np.all(arr_x == arr_x[0]) or np.all(arr_y == arr_y[0]):
        return 0.0
    res, _ = stats.spearmanr(arr_x, arr_y)
    return float(np.nan_to_num(res, nan=0.0))


def compute_cliffs_delta(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Compute Cliff's Delta effect size between two continuous samples."""
    arr_x = np.asarray(x, dtype=np.float64).ravel()
    arr_y = np.asarray(y, dtype=np.float64).ravel()
    arr_x = arr_x[np.isfinite(arr_x)]
    arr_y = arr_y[np.isfinite(arr_y)]
    nx, ny = len(arr_x), len(arr_y)
    if nx == 0 or ny == 0:
        return 0.0
    diff_matrix = np.subtract.outer(arr_x, arr_y)
    greater = np.count_nonzero(diff_matrix > 1e-12)
    less = np.count_nonzero(diff_matrix < -1e-12)
    return float(np.clip((greater - less) / (nx * ny), -1.0, 1.0))


def compute_paired_wilcoxon(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> Tuple[float, float]:
    """Compute paired two-sided Wilcoxon signed-rank test."""
    arr_x = np.asarray(x, dtype=np.float64).ravel()
    arr_y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(arr_x) & np.isfinite(arr_y)
    arr_x, arr_y = arr_x[valid], arr_y[valid]
    diff = arr_x - arr_y
    nonzero_diff = diff[np.abs(diff) > 1e-12]
    if len(nonzero_diff) < 2:
        return 0.0, 1.0
    try:
        res = stats.wilcoxon(arr_x, arr_y, zero_method="pratt")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def extract_record_from_parquet(filepath: Path) -> Optional[Dict[str, Any]]:
    """Extract metrics from a single run Parquet file."""
    m = PARQUET_PATTERN.match(filepath.name)
    dim, n_train, func, strat, surr, seed = None, None, None, None, None, None
    if m:
        gd = m.groupdict()
        dim = int(gd["dim"])
        n_train = int(gd["n_train"])
        func = gd["func"]
        strat = gd["strat"]
        surr = gd.get("surr")
        seed = int(gd["seed"])

    try:
        df = pd.read_parquet(filepath)
    except Exception as exc:
        print(f"Warning: Failed reading parquet {filepath}: {exc}", file=sys.stderr)
        return None

    if "dimension" in df.columns and dim is None:
        dim = int(df["dimension"].iloc[0])
    if dim is None:
        return None

    if surr is None and "surrogate" in df.columns:
        surr = str(df["surrogate"].iloc[0])

    d_norm = df["d_norm"].to_numpy() if "d_norm" in df.columns else None
    d_inf = df["d_inf"].to_numpy() if "d_inf" in df.columns else None
    if d_norm is None:
        return None
    if d_inf is None:
        d_inf = d_norm

    record: Dict[str, Any] = {
        "file": filepath.name,
        "dimension": dim,
        "n_train": n_train,
        "function": func,
        "strategy": strat,
        "surrogate": surr or "smac_default",
        "seed": seed,
    }

    for est_name, col_candidates in ESTIMATOR_COL_MAP.items():
        arr = None
        for c in col_candidates:
            if c in df.columns:
                arr = df[c].to_numpy()
                break
        if arr is not None:
            record[f"{est_name}_norm"] = compute_spearman_rho(d_norm, arr)
            record[f"{est_name}_inf"] = compute_spearman_rho(d_inf, arr)

    return record


def extract_record_from_summary(filepath: Path) -> Optional[Dict[str, Any]]:
    """Extract metrics from a single summary JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Warning: Failed reading JSON {filepath}: {exc}", file=sys.stderr)
        return None

    dim = data.get("dimension")
    surr = data.get("surrogate") or data.get("surrogate_type")
    if dim is None or surr is None:
        m = SUMMARY_PATTERN.match(filepath.name)
        if m:
            if dim is None:
                dim = int(m.group("dim"))
            if surr is None:
                surr = m.group("surr")
    if dim is None:
        return None

    record: Dict[str, Any] = {
        "file": filepath.name,
        "dimension": dim,
        "n_train": data.get("n_train"),
        "function": data.get("function_name"),
        "strategy": data.get("sampling_strategy"),
        "surrogate": surr or "smac_default",
        "seed": data.get("seed"),
    }

    glob = data.get("global", {})
    for est_name, key_candidates in ESTIMATOR_KEY_MAP.items():
        m_dict = None
        for k in key_candidates:
            if k in glob:
                m_dict = glob[k]
                break
        if m_dict:
            rho = m_dict.get("spearman_dist", 0.0)
            record[f"{est_name}_norm"] = rho
            record[f"{est_name}_inf"] = rho

    return record


def collect_experiment_records(
    raw_dir: Optional[str | Path] = None,
    summary_dir: Optional[str | Path] = None,
    prefer_summaries: bool = False,
    workers: int = 4,
) -> pd.DataFrame:
    """Collect metric records across Parquet or summary JSON files.

    Parameters
    ----------
    raw_dir : str | Path, optional
        Directory containing raw Parquet files.
    summary_dir : str | Path, optional
        Directory containing summary JSON files.
    prefer_summaries : bool, default=False
        If True, reads small summary JSON files first, enabling ultra-fast (<5s) aggregation.
    workers : int, default=4
        Number of worker threads for parallel file ingestion.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row is an experiment run with extracted metrics.
    """
    records: List[Dict[str, Any]] = []

    # Fast Mode: Priority Summary JSON files if requested
    if prefer_summaries and summary_dir is not None:
        s_dir = Path(summary_dir)
        if s_dir.is_dir():
            json_files = sorted(s_dir.glob("*.json"))
            total_files = len(json_files)
            if total_files > 0:
                print(f"[Aggregation] Fast Mode: Processing {total_files} summary JSON files from '{s_dir}'...")
                last_bucket = 0
                for idx, f in enumerate(json_files, start=1):
                    rec = extract_record_from_summary(f)
                    if rec is not None:
                        records.append(rec)
                    pct = int((idx / total_files) * 100)
                    bucket = pct // 10
                    if bucket > last_bucket:
                        print(f"[Aggregation] Progress: {bucket * 10}% ({idx}/{total_files} files processed)")
                        last_bucket = bucket
                if last_bucket == 0 and total_files > 0:
                    print(f"[Aggregation] Progress: 100% ({total_files}/{total_files} files processed)")
                return pd.DataFrame(records)

    # Standard Mode: Parquet files (with multi-threaded parallel read)
    if raw_dir is not None:
        p_dir = Path(raw_dir)
        if p_dir.is_dir():
            parquet_files = sorted(p_dir.glob("*.parquet"))
            total_files = len(parquet_files)
            if total_files > 0:
                print(f"[Aggregation] Processing {total_files} Parquet run files with {workers} worker(s)...")
                processed = 0
                last_bucket = 0
                if workers > 1 and total_files > 10:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = {executor.submit(extract_record_from_parquet, f): f for f in parquet_files}
                        for fut in as_completed(futures):
                            processed += 1
                            rec = fut.result()
                            if rec is not None:
                                records.append(rec)
                            pct = int((processed / total_files) * 100)
                            bucket = pct // 10
                            if bucket > last_bucket:
                                print(f"[Aggregation] Progress: {bucket * 10}% ({processed}/{total_files} files processed)")
                                last_bucket = bucket
                else:
                    for idx, f in enumerate(parquet_files, start=1):
                        rec = extract_record_from_parquet(f)
                        if rec is not None:
                            records.append(rec)
                        pct = int((idx / total_files) * 100)
                        bucket = pct // 10
                        if bucket > last_bucket:
                            print(f"[Aggregation] Progress: {bucket * 10}% ({idx}/{total_files} files processed)")
                            last_bucket = bucket
                if last_bucket == 0 and total_files > 0:
                    print(f"[Aggregation] Progress: 100% ({total_files}/{total_files} files processed)")

    # Fallback to summaries if Parquets empty or not found
    if not records and summary_dir is not None:
        s_dir = Path(summary_dir)
        if s_dir.is_dir():
            json_files = sorted(s_dir.glob("*.json"))
            total_files = len(json_files)
            if total_files > 0:
                print(f"[Aggregation] Fallback: Processing {total_files} summary JSON files from '{s_dir}'...")
                last_bucket = 0
                for idx, f in enumerate(json_files, start=1):
                    rec = extract_record_from_summary(f)
                    if rec is not None:
                        records.append(rec)
                    pct = int((idx / total_files) * 100)
                    bucket = pct // 10
                    if bucket > last_bucket:
                        print(f"[Aggregation] Progress: {bucket * 10}% ({idx}/{total_files} files processed)")
                        last_bucket = bucket
                if last_bucket == 0 and total_files > 0:
                    print(f"[Aggregation] Progress: 100% ({total_files}/{total_files} files processed)")

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def _build_scorecard_rows(
    df: pd.DataFrame,
    dim_group_name: str,
) -> List[Dict[str, Any]]:
    """Build pairwise comparison rows for a specific dimension group."""
    rows: List[Dict[str, Any]] = []

    for est_test, est_ref, comp_name in COMPARISONS:
        col_test_norm = f"{est_test}_norm"
        col_ref_norm = f"{est_ref}_norm"
        col_test_inf = f"{est_test}_inf"
        col_ref_inf = f"{est_ref}_inf"

        if col_test_norm in df.columns and col_ref_norm in df.columns:
            valid = df.dropna(subset=[col_test_norm, col_ref_norm])
            if len(valid) > 0:
                norm_test = valid[col_test_norm].to_numpy()
                norm_ref = valid[col_ref_norm].to_numpy()
                inf_test = valid[col_test_inf].to_numpy() if col_test_inf in df.columns else norm_test
                inf_ref = valid[col_ref_inf].to_numpy() if col_ref_inf in df.columns else norm_ref

                delta_norm = norm_test - norm_ref
                delta_inf = inf_test - inf_ref
                mono_test = (1.0 + norm_test) / 2.0
                mono_ref = (1.0 + norm_ref) / 2.0

                w_count = int(np.count_nonzero(delta_norm > 1e-6))
                t_count = int(np.count_nonzero(np.abs(delta_norm) <= 1e-6))
                l_count = int(np.count_nonzero(delta_norm < -1e-6))

                stat_w, p_val = compute_paired_wilcoxon(norm_test, norm_ref)
                c_delta = compute_cliffs_delta(norm_test, norm_ref)

                rows.append({
                    "dimension_group": dim_group_name,
                    "comparison": comp_name,
                    "estimator_test": est_test,
                    "estimator_ref": est_ref,
                    "n_runs": len(valid),
                    "spearman_norm_test": float(np.mean(norm_test)),
                    "spearman_norm_ref": float(np.mean(norm_ref)),
                    "delta_rho_norm": float(np.mean(delta_norm)),
                    "spearman_inf_test": float(np.mean(inf_test)),
                    "spearman_inf_ref": float(np.mean(inf_ref)),
                    "delta_rho_inf": float(np.mean(delta_inf)),
                    "norm_rank_monotonicity_test": float(np.mean(mono_test)),
                    "norm_rank_monotonicity_ref": float(np.mean(mono_ref)),
                    "wilcoxon_stat": float(stat_w),
                    "wilcoxon_pvalue": float(p_val),
                    "cliffs_delta": float(c_delta),
                    "win_tie_loss": f"{w_count}/{t_count}/{l_count}",
                })

    return rows


def build_leaderboard(
    df: pd.DataFrame,
    dim_group_name: str,
) -> pd.DataFrame:
    """Build an absolute performance ranking leaderboard table."""
    entries: List[Dict[str, Any]] = []

    for est, meta in ESTIMATOR_META.items():
        col_norm = f"{est}_norm"
        col_inf = f"{est}_inf"
        if col_norm in df.columns:
            valid_norm = df[col_norm].dropna().to_numpy()
            if len(valid_norm) > 0:
                valid_inf = df[col_inf].dropna().to_numpy() if col_inf in df.columns else valid_norm
                mean_norm = float(np.mean(valid_norm))
                mean_inf = float(np.mean(valid_inf))
                mono = float(np.mean((1.0 + valid_norm) / 2.0))
                entries.append({
                    "dimension_group": dim_group_name,
                    "estimator": est,
                    "name": meta["name"],
                    "category": meta["category"],
                    "n_runs": len(valid_norm),
                    "spearman_norm": mean_norm,
                    "spearman_inf": mean_inf,
                    "norm_rank_monotonicity": mono,
                })

    if not entries:
        return pd.DataFrame()

    res_df = pd.DataFrame(entries)
    res_df = res_df.sort_values(by="spearman_norm", ascending=False).reset_index(drop=True)
    res_df.insert(0, "rank", np.arange(1, len(res_df) + 1))
    return res_df


def build_surrogate_breakdown(df_runs: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Generate a breakdown of estimator performance per surrogate architecture."""
    if df_runs.empty or "surrogate" not in df_runs.columns:
        return pd.DataFrame(), ""

    surrogates = sorted(df_runs["surrogate"].dropna().unique())
    rows: List[Dict[str, Any]] = []

    for s in surrogates:
        sub_df = df_runs[df_runs["surrogate"] == s]
        for est, meta in ESTIMATOR_META.items():
            c_norm = f"{est}_norm"
            if c_norm in sub_df.columns:
                vals = sub_df[c_norm].dropna().to_numpy()
                if len(vals) > 0:
                    rows.append({
                        "surrogate": s,
                        "estimator": est,
                        "name": meta["name"],
                        "category": meta["category"],
                        "n_runs": len(vals),
                        "spearman_norm": float(np.mean(vals)),
                        "norm_rank_monotonicity": float(np.mean((1.0 + vals) / 2.0)),
                    })

    if not rows:
        return pd.DataFrame(), ""

    surr_df = pd.DataFrame(rows)

    md_lines = [
        "# Lower-Quantile UQ: Surrogate Architecture Breakdown",
        "",
        "Empirical calibration performance of each UQ estimator stratified across the 5 Random Forest surrogates:",
        "`smac_default`, `mature`, `shallow`, `coarse`, `breiman`.",
        "",
        "| Surrogate | Estimator | Category | N Runs | Spearman d_norm | Norm Monotonicity |",
        "| :--- | :--- | :--- | ---: | ---: | ---: |",
    ]
    for _, r in surr_df.iterrows():
        md_lines.append(
            f"| `{r['surrogate']}` | `{r['estimator']}` | {r['category']} | {r['n_runs']} | "
            f"{r['spearman_norm']:+.4f} | {r['norm_rank_monotonicity']:.4f} |"
        )

    return surr_df, "\n".join(md_lines)


def generate_lower_quantile_scorecard(
    raw_dir: Optional[str | Path] = None,
    summary_dir: Optional[str | Path] = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    prefer_summaries: bool = False,
    workers: int = 4,
) -> Tuple[pd.DataFrame, str]:
    """Generate comprehensive lower-quantile scorecard, leaderboard, and surrogate tables.

    Parameters
    ----------
    raw_dir : str | Path, optional
        Directory containing raw Parquet files.
    summary_dir : str | Path, optional
        Directory containing summary JSON files.
    output_dir : str | Path, default=DEFAULT_OUTPUT_DIR
        Target directory to write CSV and Markdown scorecards.
    prefer_summaries : bool, default=False
        If True, reads summary JSON files for rapid aggregation.
    workers : int, default=4
        Worker threads for parallel file loading.

    Returns
    -------
    tuple[pd.DataFrame, str]
        Scorecard DataFrame and Markdown content string.
    """
    df_runs = collect_experiment_records(
        raw_dir=raw_dir,
        summary_dir=summary_dir,
        prefer_summaries=prefer_summaries,
        workers=workers,
    )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if df_runs.empty:
        empty_cols = [
            "dimension_group", "comparison", "estimator_test", "estimator_ref",
            "n_runs", "spearman_norm_test", "spearman_norm_ref", "delta_rho_norm",
            "spearman_inf_test", "spearman_inf_ref", "delta_rho_inf",
            "norm_rank_monotonicity_test", "norm_rank_monotonicity_ref",
            "wilcoxon_stat", "wilcoxon_pvalue", "cliffs_delta", "win_tie_loss",
        ]
        scorecard_df = pd.DataFrame(columns=empty_cols)
        md_content = "# Lower-Quantile UQ Calibration Scorecard\n\nNo experimental records found.\n"
        csv_path = out_p / "table_lower_quantile_scorecard.csv"
        md_path = out_p / "table_lower_quantile_scorecard.md"
        scorecard_df.to_csv(csv_path, index=False)
        md_path.write_text(md_content, encoding="utf-8")
        return scorecard_df, md_content

    # Partition by dimension groups
    low_d = df_runs[df_runs["dimension"] <= 5]
    high_d = df_runs[df_runs["dimension"] >= 16]

    # 1. Build Pairwise Scorecards
    all_scorecard_rows: List[Dict[str, Any]] = []
    all_scorecard_rows.extend(_build_scorecard_rows(low_d, "Low-D (D <= 5)"))
    all_scorecard_rows.extend(_build_scorecard_rows(high_d, "High-D (D >= 16)"))
    all_scorecard_rows.extend(_build_scorecard_rows(df_runs, "All Dimensions"))
    scorecard_df = pd.DataFrame(all_scorecard_rows)

    md_lines: List[str] = [
        "# Lower-Quantile UQ Calibration Scorecard",
        "",
        "## Executive Summary",
        "",
        "Empirical calibration scorecard comparing **Lower-Quantile Proximity UQ variants** (A, B, AC, BC, PLCB)",
        "against classical non-proximity baselines (SLCB / Hutter Total, Hutter Between, RF-FIRE, Shaker Epistemic).",
        "Evaluates Spearman rank correlation with Euclidean (d_norm) and Chebyshev (d_inf) projection distances,",
        "difference in distance alignment Delta(rho), and normalized rank monotonicity ((1 + rho) / 2).",
        "",
        "## Calibration Scorecard Table",
        "",
        "| Dimension Group | Comparison | N | Spearman d_norm (Test) | Spearman d_norm (Ref) | Delta rho (Norm) | Spearman d_inf (Test) | Spearman d_inf (Ref) | Delta rho (Inf) | Norm Monotonicity (Test) | Norm Monotonicity (Ref) | W / T / L | Wilcoxon p | Cliff's Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, r in scorecard_df.iterrows():
        p_val_str = f"{r['wilcoxon_pvalue']:.4f}" if r["wilcoxon_pvalue"] >= 0.001 else f"{r['wilcoxon_pvalue']:.1e}"
        md_lines.append(
            f"| **{r['dimension_group']}** | `{r['comparison']}` | {r['n_runs']} | "
            f"{r['spearman_norm_test']:+.4f} | {r['spearman_norm_ref']:+.4f} | "
            f"**{r['delta_rho_norm']:+.4f}** | "
            f"{r['spearman_inf_test']:+.4f} | {r['spearman_inf_ref']:+.4f} | "
            f"**{r['delta_rho_inf']:+.4f}** | "
            f"{r['norm_rank_monotonicity_test']:.4f} | {r['norm_rank_monotonicity_ref']:.4f} | "
            f"{r['win_tie_loss']} | {p_val_str} | {r['cliffs_delta']:+.3f} |"
        )

    md_lines.extend([
        "",
        "## Statistical Methodology Notes",
        "- **Normalized Rank Monotonicity**: Computed as `(1 + rho(d_norm, U)) / 2`, providing a normalized index in [0, 1] where 1.0 indicates perfect monotonic increase of uncertainty with distance.",
        "- **Delta rho**: `rho(test) - rho(ref)`. Positive values indicate that the test proximity estimator exhibits superior monotonic distance alignment.",
        "- **Paired Wilcoxon Signed-Rank Test**: Two-sided test evaluating whether paired differences in rank correlation are statistically significant.",
        "- **Cliff's Delta**: Non-parametric effect size in [-1, +1] where positive values favor the test estimator.",
        "",
    ])

    md_content = "\n".join(md_lines)
    csv_path = out_p / "table_lower_quantile_scorecard.csv"
    md_path = out_p / "table_lower_quantile_scorecard.md"
    scorecard_df.to_csv(csv_path, index=False)
    md_path.write_text(md_content, encoding="utf-8")

    # 2. Build Absolute Performance Leaderboards
    lb_all = build_leaderboard(df_runs, "All Dimensions")
    lb_low = build_leaderboard(low_d, "Low-D (D <= 5)")
    lb_high = build_leaderboard(high_d, "High-D (D >= 16)")

    combined_lb = pd.concat([lb_all, lb_low, lb_high], ignore_index=True)
    lb_csv_path = out_p / "table_lower_quantile_leaderboard.csv"
    lb_md_path = out_p / "table_lower_quantile_leaderboard.md"
    combined_lb.to_csv(lb_csv_path, index=False)

    lb_md_lines = [
        "# Lower-Quantile UQ Absolute Performance Leaderboard",
        "",
        "Absolute ranking of all evaluated UQ estimators sorted by mean Spearman rank correlation with distance.",
        "",
    ]
    for grp_name, grp_df in [("All Dimensions", lb_all), ("Low-D (D <= 5)", lb_low), ("High-D (D >= 16)", lb_high)]:
        lb_md_lines.extend([
            f"### Leaderboard: {grp_name}",
            "",
            "| Rank | Estimator | Method Description | Category | N Runs | Spearman d_norm | Spearman d_inf | Norm Monotonicity |",
            "| ---: | :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
        ])
        for _, r in grp_df.iterrows():
            lb_md_lines.append(
                f"| **{r['rank']}** | `{r['estimator']}` | {r['name']} | {r['category']} | {r['n_runs']} | "
                f"**{r['spearman_norm']:+.4f}** | {r['spearman_inf']:+.4f} | {r['norm_rank_monotonicity']:.4f} |"
            )
        lb_md_lines.append("")

    lb_md_path.write_text("\n".join(lb_md_lines), encoding="utf-8")

    # 3. Build Surrogate Breakdown Table
    surr_df, surr_md = build_surrogate_breakdown(df_runs)
    if not surr_df.empty:
        surr_csv_path = out_p / "table_lower_quantile_by_surrogate.csv"
        surr_md_path = out_p / "table_lower_quantile_by_surrogate.md"
        surr_df.to_csv(surr_csv_path, index=False)
        surr_md_path.write_text(surr_md, encoding="utf-8")

    print(f"Generated scorecard CSV -> {csv_path}")
    print(f"Generated scorecard Markdown -> {md_path}")
    print(f"Generated leaderboard CSV -> {lb_csv_path}")
    print(f"Generated leaderboard Markdown -> {lb_md_path}")
    if not surr_df.empty:
        print(f"Generated surrogate breakdown -> {out_p / 'table_lower_quantile_by_surrogate.md'}")

    return scorecard_df, md_content


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for lower-quantile results aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate lower-quantile sweep results and generate scorecards, leaderboard, and surrogate analysis."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f"Root results directory (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Optional directory containing raw Parquet files (default: <results-dir>/raw).",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default=None,
        help="Optional directory containing summary JSON files (default: <results-dir>/summaries).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated scorecards (default: <results-dir>/analysis).",
    )
    parser.add_argument(
        "--use-summaries",
        action="store_true",
        help="Prioritize reading small JSON summaries for ultra-fast (<5s) scorecard generation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Number of worker threads for parallel file loading (default: min(8, cpu_count)).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else results_dir / "raw"
    summary_dir = Path(args.summary_dir) if args.summary_dir else results_dir / "summaries"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"

    generate_lower_quantile_scorecard(
        raw_dir=raw_dir,
        summary_dir=summary_dir,
        output_dir=output_dir,
        prefer_summaries=args.use_summaries,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
