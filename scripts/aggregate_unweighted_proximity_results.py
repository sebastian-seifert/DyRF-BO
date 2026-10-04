#!/usr/bin/env python3
"""Aggregator and Scorecard Generator for Unweighted Proximity Sweep (Milestone 3).

Reads raw evaluation Parquets and summary JSONs from unweighted proximity sweep runs,
evaluates distance-uncertainty correlation metrics:
- Spearman rho(d_norm, U)
- Spearman rho(d_inf, U)
- Delta(rho)
- Normalized rank monotonicity: (1 + rho) / 2

Generates head-to-head scorecards:
- prox_a_unweighted vs prox_a (weighted)
- plcb_unweighted vs plcb (weighted)
stratified across Low-D (D <= 5), High-D (D >= 16), and All Dimensions.

Outputs:
- table_unweighted_vs_weighted_scorecard.csv
- table_unweighted_vs_weighted_scorecard.md
"""

from __future__ import annotations

import argparse
import json
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

DEFAULT_RESULTS_DIR = "results/unweighted_proximity_sweep"
DEFAULT_RAW_DIR = "results/unweighted_proximity_sweep/raw"
DEFAULT_SUMMARY_DIR = "results/unweighted_proximity_sweep/summaries"
DEFAULT_OUTPUT_DIR = "results/unweighted_proximity_sweep/analysis"

PARQUET_PATTERN = re.compile(
    r"^extrapolation_(?P<func>.+?)_d(?P<dim>\d+)_n(?P<n_train>\d+)_(?P<strat>[a-zA-Z0-9_]+?)(?:_(?P<surr>[a-zA-Z0-9_]+?))?_s(?P<seed>\d+)\.parquet$"
)
SUMMARY_PATTERN = re.compile(
    r"^summary_(?P<func>.+?)_d(?P<dim>\d+)_n(?P<n_train>\d+)_(?P<strat>[a-zA-Z0-9_]+?)(?:_(?P<surr>[a-zA-Z0-9_]+?))?_s(?P<seed>\d+)\.json$"
)


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
    """Extract metrics from single run parquet file."""
    m = PARQUET_PATTERN.match(filepath.name)
    dim, n_train, func, strat, seed = None, None, None, None, None
    if m:
        gd = m.groupdict()
        dim = int(gd["dim"])
        n_train = int(gd["n_train"])
        func = gd["func"]
        strat = gd["strat"]
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

    d_norm = df["d_norm"].to_numpy() if "d_norm" in df.columns else None
    d_inf = df["d_inf"].to_numpy() if "d_inf" in df.columns else None
    if d_norm is None:
        return None
    if d_inf is None:
        d_inf = d_norm

    # Resolve prox_a uncertainty columns
    u_prox_a_unw = None
    for col in ["u_prox_a_unweighted_half", "u_prox_a_unweighted", "u_prox_a_unweighted_lower"]:
        if col in df.columns:
            u_prox_a_unw = df[col].to_numpy()
            break

    u_prox_a_w = None
    for col in ["u_prox_a_weighted_half", "u_prox_a_half", "u_prox_a", "u_prox_a_weighted"]:
        if col in df.columns:
            u_prox_a_w = df[col].to_numpy()
            break

    # Resolve plcb uncertainty columns
    u_plcb_unw = None
    for col in ["u_plcb_unweighted_half", "u_plcb_unweighted", "u_plcb_unweighted_lower"]:
        if col in df.columns:
            u_plcb_unw = df[col].to_numpy()
            break

    u_plcb_w = None
    for col in ["u_plcb_weighted_half", "u_plcb_half", "u_plcb", "u_plcb_weighted"]:
        if col in df.columns:
            u_plcb_w = df[col].to_numpy()
            break

    record: Dict[str, Any] = {
        "file": filepath.name,
        "dimension": dim,
        "n_train": n_train,
        "function": func,
        "strategy": strat,
        "seed": seed,
    }

    if u_prox_a_unw is not None and u_prox_a_w is not None:
        record["prox_a_norm_unw"] = compute_spearman_rho(d_norm, u_prox_a_unw)
        record["prox_a_norm_w"] = compute_spearman_rho(d_norm, u_prox_a_w)
        record["prox_a_inf_unw"] = compute_spearman_rho(d_inf, u_prox_a_unw)
        record["prox_a_inf_w"] = compute_spearman_rho(d_inf, u_prox_a_w)

    if u_plcb_unw is not None and u_plcb_w is not None:
        record["plcb_norm_unw"] = compute_spearman_rho(d_norm, u_plcb_unw)
        record["plcb_norm_w"] = compute_spearman_rho(d_norm, u_plcb_w)
        record["plcb_inf_unw"] = compute_spearman_rho(d_inf, u_plcb_unw)
        record["plcb_inf_w"] = compute_spearman_rho(d_inf, u_plcb_w)

    return record


def extract_record_from_summary(filepath: Path) -> Optional[Dict[str, Any]]:
    """Extract metrics from single summary JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Warning: Failed reading JSON {filepath}: {exc}", file=sys.stderr)
        return None

    dim = data.get("dimension")
    if dim is None:
        m = SUMMARY_PATTERN.match(filepath.name)
        if m:
            dim = int(m.group("dim"))
    if dim is None:
        return None

    record: Dict[str, Any] = {
        "file": filepath.name,
        "dimension": dim,
        "n_train": data.get("n_train"),
        "function": data.get("function_name"),
        "strategy": data.get("sampling_strategy"),
        "seed": data.get("seed"),
    }

    glob = data.get("global", {})
    # prox_a
    unw_pa = glob.get("u_prox_a_unweighted_half") or glob.get("u_prox_a_unweighted")
    w_pa = glob.get("u_prox_a_weighted_half") or glob.get("u_prox_a_half") or glob.get("u_prox_a")
    if unw_pa and w_pa:
        record["prox_a_norm_unw"] = unw_pa.get("spearman_dist", 0.0)
        record["prox_a_norm_w"] = w_pa.get("spearman_dist", 0.0)
        record["prox_a_inf_unw"] = unw_pa.get("spearman_dist", 0.0)
        record["prox_a_inf_w"] = w_pa.get("spearman_dist", 0.0)

    # plcb
    unw_plcb = glob.get("u_plcb_unweighted_half") or glob.get("u_plcb_unweighted")
    w_plcb = glob.get("u_plcb_weighted_half") or glob.get("u_plcb_half") or glob.get("u_plcb")
    if unw_plcb and w_plcb:
        record["plcb_norm_unw"] = unw_plcb.get("spearman_dist", 0.0)
        record["plcb_norm_w"] = w_plcb.get("spearman_dist", 0.0)
        record["plcb_inf_unw"] = unw_plcb.get("spearman_dist", 0.0)
        record["plcb_inf_w"] = w_plcb.get("spearman_dist", 0.0)

    return record


def collect_experiment_records(
    raw_dir: Optional[str | Path] = None,
    summary_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Collect metric records across parquet and summary JSON files."""
    records: List[Dict[str, Any]] = []

    # Priority 1: Parquet files
    if raw_dir is not None:
        p_dir = Path(raw_dir)
        if p_dir.is_dir():
            parquet_files = sorted(p_dir.glob("*.parquet"))
            total_files = len(parquet_files)
            if total_files > 0:
                print(f"[Aggregation] Processing {total_files} Parquet run files from '{p_dir}'...")
                last_bucket = 0
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

    # Priority 2: Summary JSON files (if no parquet records found or supplementary)
    if not records and summary_dir is not None:
        s_dir = Path(summary_dir)
        if s_dir.is_dir():
            json_files = sorted(s_dir.glob("*.json"))
            total_files = len(json_files)
            if total_files > 0:
                print(f"[Aggregation] Processing {total_files} summary JSON files from '{s_dir}'...")
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
    """Build comparison rows for a specific dimension group."""
    rows: List[Dict[str, Any]] = []

    # 1. prox_a_unweighted vs prox_a (weighted)
    if "prox_a_norm_unw" in df.columns and "prox_a_norm_w" in df.columns:
        valid = df.dropna(subset=["prox_a_norm_unw", "prox_a_norm_w"])
        if len(valid) > 0:
            norm_unw = valid["prox_a_norm_unw"].to_numpy()
            norm_w = valid["prox_a_norm_w"].to_numpy()
            inf_unw = valid["prox_a_inf_unw"].to_numpy()
            inf_w = valid["prox_a_inf_w"].to_numpy()

            delta_norm = norm_unw - norm_w
            delta_inf = inf_unw - inf_w
            mono_unw = (1.0 + norm_unw) / 2.0
            mono_w = (1.0 + norm_w) / 2.0

            w_count = int(np.count_nonzero(delta_norm > 1e-6))
            t_count = int(np.count_nonzero(np.abs(delta_norm) <= 1e-6))
            l_count = int(np.count_nonzero(delta_norm < -1e-6))

            stat_w, p_val = compute_paired_wilcoxon(norm_unw, norm_w)
            c_delta = compute_cliffs_delta(norm_unw, norm_w)

            rows.append({
                "dimension_group": dim_group_name,
                "comparison": "prox_a_unweighted vs prox_a (weighted)",
                "estimator_unweighted": "prox_a_unweighted",
                "estimator_weighted": "prox_a",
                "n_runs": len(valid),
                "spearman_norm_unweighted": float(np.mean(norm_unw)),
                "spearman_norm_weighted": float(np.mean(norm_w)),
                "spearman_inf_unweighted": float(np.mean(inf_unw)),
                "spearman_inf_weighted": float(np.mean(inf_w)),
                "delta_rho_norm": float(np.mean(delta_norm)),
                "delta_rho_inf": float(np.mean(delta_inf)),
                "norm_rank_monotonicity_unweighted": float(np.mean(mono_unw)),
                "norm_rank_monotonicity_weighted": float(np.mean(mono_w)),
                "wilcoxon_stat": float(stat_w),
                "wilcoxon_pvalue": float(p_val),
                "cliffs_delta": float(c_delta),
                "win_tie_loss": f"{w_count}/{t_count}/{l_count}",
            })

    # 2. plcb_unweighted vs plcb (weighted)
    if "plcb_norm_unw" in df.columns and "plcb_norm_w" in df.columns:
        valid = df.dropna(subset=["plcb_norm_unw", "plcb_norm_w"])
        if len(valid) > 0:
            norm_unw = valid["plcb_norm_unw"].to_numpy()
            norm_w = valid["plcb_norm_w"].to_numpy()
            inf_unw = valid["plcb_inf_unw"].to_numpy()
            inf_w = valid["plcb_inf_w"].to_numpy()

            delta_norm = norm_unw - norm_w
            delta_inf = inf_unw - inf_w
            mono_unw = (1.0 + norm_unw) / 2.0
            mono_w = (1.0 + norm_w) / 2.0

            w_count = int(np.count_nonzero(delta_norm > 1e-6))
            t_count = int(np.count_nonzero(np.abs(delta_norm) <= 1e-6))
            l_count = int(np.count_nonzero(delta_norm < -1e-6))

            stat_w, p_val = compute_paired_wilcoxon(norm_unw, norm_w)
            c_delta = compute_cliffs_delta(norm_unw, norm_w)

            rows.append({
                "dimension_group": dim_group_name,
                "comparison": "plcb_unweighted vs plcb (weighted)",
                "estimator_unweighted": "plcb_unweighted",
                "estimator_weighted": "plcb",
                "n_runs": len(valid),
                "spearman_norm_unweighted": float(np.mean(norm_unw)),
                "spearman_norm_weighted": float(np.mean(norm_w)),
                "spearman_inf_unweighted": float(np.mean(inf_unw)),
                "spearman_inf_weighted": float(np.mean(inf_w)),
                "delta_rho_norm": float(np.mean(delta_norm)),
                "delta_rho_inf": float(np.mean(delta_inf)),
                "norm_rank_monotonicity_unweighted": float(np.mean(mono_unw)),
                "norm_rank_monotonicity_weighted": float(np.mean(mono_w)),
                "wilcoxon_stat": float(stat_w),
                "wilcoxon_pvalue": float(p_val),
                "cliffs_delta": float(c_delta),
                "win_tie_loss": f"{w_count}/{t_count}/{l_count}",
            })

    return rows


def generate_unweighted_scorecard(
    raw_dir: Optional[str | Path] = None,
    summary_dir: Optional[str | Path] = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Tuple[pd.DataFrame, str]:
    """Generate head-to-head scorecard CSV and Markdown files.

    Parameters
    ----------
    raw_dir : str | Path, optional
        Directory containing raw Parquet files.
    summary_dir : str | Path, optional
        Directory containing summary JSON files.
    output_dir : str | Path, default=DEFAULT_OUTPUT_DIR
        Target directory to write CSV and Markdown scorecards.

    Returns
    -------
    tuple[pd.DataFrame, str]
        Scorecard DataFrame and Markdown content string.
    """
    df_runs = collect_experiment_records(raw_dir=raw_dir, summary_dir=summary_dir)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if df_runs.empty:
        # Create empty template scorecard if no runs found
        empty_cols = [
            "dimension_group", "comparison", "estimator_unweighted", "estimator_weighted",
            "n_runs", "spearman_norm_unweighted", "spearman_norm_weighted",
            "spearman_inf_unweighted", "spearman_inf_weighted",
            "delta_rho_norm", "delta_rho_inf",
            "norm_rank_monotonicity_unweighted", "norm_rank_monotonicity_weighted",
            "wilcoxon_stat", "wilcoxon_pvalue", "cliffs_delta", "win_tie_loss",
        ]
        scorecard_df = pd.DataFrame(columns=empty_cols)
        md_content = "# Unweighted vs Weighted Proximity Calibration Scorecard\n\nNo experimental records found.\n"
        csv_path = out_p / "table_unweighted_vs_weighted_scorecard.csv"
        md_path = out_p / "table_unweighted_vs_weighted_scorecard.md"
        scorecard_df.to_csv(csv_path, index=False)
        md_path.write_text(md_content, encoding="utf-8")
        return scorecard_df, md_content

    # Partition by dimension groups
    low_d = df_runs[df_runs["dimension"] <= 5]
    high_d = df_runs[df_runs["dimension"] >= 16]

    all_rows: List[Dict[str, Any]] = []
    # Stratified evaluations
    all_rows.extend(_build_scorecard_rows(low_d, "Low-D (D <= 5)"))
    all_rows.extend(_build_scorecard_rows(high_d, "High-D (D >= 16)"))
    all_rows.extend(_build_scorecard_rows(df_runs, "All Dimensions"))

    scorecard_df = pd.DataFrame(all_rows)

    # Format Markdown scorecard
    md_lines: List[str] = [
        "# Unweighted vs Weighted Proximity Calibration Scorecard",
        "",
        "## Executive Summary",
        "",
        "Head-to-head empirical comparison of **Unweighted Proximity UQ** vs **Weighted RF-GAP (Legacy)**.",
        "Evaluates Spearman rank correlation with Euclidean (d_norm) and Chebyshev (d_inf) projection distances,",
        "difference in distance alignment Delta(rho), and normalized rank monotonicity ((1 + rho) / 2).",
        "",
        "## Head-to-Head Calibration Scorecard Table",
        "",
        "| Dimension Group | Comparison | N | Spearman d_norm (Unw) | Spearman d_norm (W) | Delta rho (Norm) | Spearman d_inf (Unw) | Spearman d_inf (W) | Delta rho (Inf) | Norm Rank Monotonicity (Unw) | Norm Rank Monotonicity (W) | W / T / L | Wilcoxon p | Cliff's Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, r in scorecard_df.iterrows():
        p_val_str = f"{r['wilcoxon_pvalue']:.4f}" if r["wilcoxon_pvalue"] >= 0.001 else f"{r['wilcoxon_pvalue']:.1e}"
        md_lines.append(
            f"| **{r['dimension_group']}** | `{r['comparison']}` | {r['n_runs']} | "
            f"{r['spearman_norm_unweighted']:+.4f} | {r['spearman_norm_weighted']:+.4f} | "
            f"**{r['delta_rho_norm']:+.4f}** | "
            f"{r['spearman_inf_unweighted']:+.4f} | {r['spearman_inf_weighted']:+.4f} | "
            f"**{r['delta_rho_inf']:+.4f}** | "
            f"{r['norm_rank_monotonicity_unweighted']:.4f} | {r['norm_rank_monotonicity_weighted']:.4f} | "
            f"{r['win_tie_loss']} | {p_val_str} | {r['cliffs_delta']:+.3f} |"
        )

    md_lines.extend([
        "",
        "## Statistical Methodology Notes",
        "- **Normalized Rank Monotonicity**: Computed as `(1 + rho(d_norm, U)) / 2`, providing a normalized index in [0, 1] where 1.0 indicates perfect monotonic increase of uncertainty with distance.",
        "- **Delta rho**: `rho(unweighted) - rho(weighted)`. Positive values indicate that unweighted proximity uncertainty exhibits superior monotonic distance alignment.",
        "- **Paired Wilcoxon Signed-Rank Test**: Two-sided test testing whether the difference between paired unweighted and weighted correlations is significantly different from zero.",
        "- **Cliff's Delta**: Non-parametric effect size in [-1, +1] where positive values favor the unweighted estimator.",
        "",
    ])

    md_content = "\n".join(md_lines)

    csv_path = out_p / "table_unweighted_vs_weighted_scorecard.csv"
    md_path = out_p / "table_unweighted_vs_weighted_scorecard.md"

    scorecard_df.to_csv(csv_path, index=False)
    md_path.write_text(md_content, encoding="utf-8")

    print(f"Generated scorecard CSV -> {csv_path}")
    print(f"Generated scorecard Markdown -> {md_path}")

    return scorecard_df, md_content


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for unweighted proximity results aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate unweighted proximity sweep results and generate scorecards."
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
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else results_dir / "raw"
    summary_dir = Path(args.summary_dir) if args.summary_dir else results_dir / "summaries"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"

    generate_unweighted_scorecard(
        raw_dir=raw_dir,
        summary_dir=summary_dir,
        output_dir=output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
