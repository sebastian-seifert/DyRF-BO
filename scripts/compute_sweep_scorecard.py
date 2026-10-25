#!/usr/bin/env python3
"""Unified Benchmark Sweep Scorecard Engine.

Computes publication-standard statistical scorecards from any CARP-S / SMAC3
optimization run logs (Parquet or CSV):
1. Extracts final incumbents per (task_id, optimizer_id, seed).
2. Applies global per-task Min-Max scaled regret:
       normalized_regret = (cost - min_task_cost) / (max_task_cost - min_task_cost)
3. Computes per-task median normalized regret across paired seeds.
4. Excludes baseline optimizers from evaluated candidate rows.
5. Computes pairwise directional one-sided Wilcoxon signed-rank tests (H_1: cand < baseline).
6. Computes matched-pairs rank-biserial correlation with strict tie-filtering:
       r_rb = (W^+ - W^-) / (W^+ + W^-)
   where pairs with |cand - baseline| <= tie_tolerance are discarded prior to ranking.
7. Applies step-down Holm-Bonferroni correction across candidate approaches for each baseline.
8. Exports a clean, streamlined final table containing solely `optimizer_id` and the three
   essential metrics per baseline: win rate, adjusted p-value, and rank-biserial correlation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

DEFAULT_CONTROLS: List[str] = ["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"]
DEFAULT_TIE_TOLERANCE: float = 1e-6
DEFAULT_COST_COL: str = "trial_value__cost_inc"


def apply_holm_bonferroni(p_vals: Sequence[float]) -> List[float]:
    """Computes Holm-Bonferroni step-down adjusted p-values ensuring FWER control."""
    p = np.array(p_vals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    p = np.where(np.isnan(p), 1.0, p)
    sort_idx = np.argsort(p)
    sorted_p = p[sort_idx]

    adj_sorted = np.zeros(n, dtype=float)
    for i in range(n):
        adj_sorted[i] = min(1.0, sorted_p[i] * (n - i))

    for i in range(1, n):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])

    adj_original = np.zeros(n, dtype=float)
    adj_original[sort_idx] = adj_sorted
    return [float(val) for val in adj_original]


def calculate_rank_biserial_correlation(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    tie_tolerance: float = 1e-6,
) -> float:
    """Computes matched-pairs rank-biserial correlation r_rb = (W^+ - W^-) / (W^+ + W^-).

    Ties with |x_i - y_i| <= tie_tolerance are strictly filtered out prior to ranking.
    Negative r_rb indicates x is systematically lower (better for minimization) than y.
    Returns:
        -1.0 if x is strictly lower on all non-tied pairs.
        +1.0 if y is strictly lower on all non-tied pairs.
         0.0 if all pairs are ties or positive/negative rank sums are equal.
    """
    arr_x = np.asarray(x, dtype=float).ravel()
    arr_y = np.asarray(y, dtype=float).ravel()
    if len(arr_x) != len(arr_y) or len(arr_x) == 0:
        return 0.0

    valid = ~(np.isnan(arr_x) | np.isnan(arr_y))
    arr_x, arr_y = arr_x[valid], arr_y[valid]
    if len(arr_x) == 0:
        return 0.0

    diff = arr_x - arr_y
    nonzero = diff[np.abs(diff) > tie_tolerance]
    if len(nonzero) == 0:
        return 0.0

    abs_diff = np.abs(nonzero)
    ranks = rankdata(abs_diff)
    w_plus = float(np.sum(ranks[nonzero > 0]))   # Baseline wins (x > y)
    w_minus = float(np.sum(ranks[nonzero < 0]))  # Proposed wins (x < y)
    total = w_plus + w_minus
    if total <= 1e-12:
        return 0.0

    return float((w_plus - w_minus) / total)


def compute_min_max_scaled_regret(
    df: pd.DataFrame,
    cost_col: str = "trial_value__cost_inc",
) -> pd.DataFrame:
    """Applies global per-task Min-Max scaling across all evaluated approaches and seeds."""
    df_scaled = df.copy()
    if cost_col not in df_scaled.columns:
        for candidate in ["cost_inc", "trial_value__cost", "cost", "value"]:
            if candidate in df_scaled.columns:
                cost_col = candidate
                break

    task_bounds = df_scaled.groupby("task_id")[cost_col].agg(["min", "max"]).reset_index()
    task_bounds["spread"] = task_bounds["max"] - task_bounds["min"]
    merged = pd.merge(df_scaled, task_bounds, on="task_id", how="left")

    def _scale(row):
        spread = row["spread"]
        if spread > 1e-12:
            return (row[cost_col] - row["min"]) / spread
        return 0.0

    merged["normalized_regret"] = merged.apply(_scale, axis=1)
    return merged


def get_control_prefix(ctrl: str) -> str:
    """Maps control optimizer ID to a standardized concise prefix."""
    ctrl_lower = ctrl.lower()
    if "lcb" in ctrl_lower:
        return "ctrl_lcb"
    elif "ei" in ctrl_lower:
        return "ctrl_ei"
    clean_name = ctrl.replace(" ", "_").replace("-", "_").lower()
    return f"ctrl_{clean_name}"


def compute_sweep_scorecard(
    df: pd.DataFrame,
    controls: Sequence[str] = ("SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"),
    cost_col: str = "trial_value__cost_inc",
    tie_tolerance: float = 1e-6,
    holm_scope: str = "per_control",
) -> Dict[str, Any]:
    """Generates the unified statistical scorecard from benchmark logs.

    Parameters
    ----------
    df : pd.DataFrame
        Optimization run logs containing at least task, optimizer, seed, and cost columns.
    controls : Sequence[str]
        Baseline optimizer IDs used as reference controls.
    cost_col : str
        Target cost column name.
    tie_tolerance : float
        Numerical threshold below which differences are treated as exact ties.
    holm_scope : str
        'per_control' (standard) or 'joint' family for Holm-Bonferroni correction.

    Returns
    -------
    Dict[str, Any] with keys:
        'scorecard': DataFrame containing ONLY `optimizer_id` and [win_rate, wilcoxon_adj_p, rank_biserial]
                     for each baseline, with baseline optimizers strictly excluded from rows.
        'full_scorecard': Extended DataFrame including sample sizes, normalized regret means/medians, etc.
        'scaled_df': Pre-processed and min-max scaled trial log DataFrame.
    """
    working_df = df.copy()

    # Resolve column names
    opt_col = "optimizer_id" if "optimizer_id" in working_df.columns else "optimizer"
    task_col = "task_id" if "task_id" in working_df.columns else "task"
    if opt_col not in working_df.columns or task_col not in working_df.columns:
        raise KeyError(f"DataFrame must contain optimizer and task columns. Found: {working_df.columns.tolist()}")

    working_df = working_df.rename(columns={opt_col: "optimizer_id", task_col: "task_id"})

    if cost_col not in working_df.columns:
        for candidate in ["cost_inc", "trial_value__cost", "cost", "value"]:
            if candidate in working_df.columns:
                cost_col = candidate
                break

    # 1. Final incumbent extraction per run (task_id, optimizer_id, seed)
    seed_col = "seed" if "seed" in working_df.columns else None
    grouper = ["task_id", "optimizer_id"] + ([seed_col] if seed_col else [])
    trial_col = "trial" if "trial" in working_df.columns else ("n_trials" if "n_trials" in working_df.columns else None)

    if trial_col:
        idx = working_df.groupby(grouper)[trial_col].idxmax()
        final_df = working_df.loc[idx].copy()
    else:
        idx = working_df.groupby(grouper)[cost_col].idxmin()
        final_df = working_df.loc[idx].copy()

    # 2. Global per-task Min-Max scaled regret
    scaled_df = compute_min_max_scaled_regret(final_df, cost_col=cost_col)

    # 3. Partition optimizers into candidate approaches and baselines
    all_optimizers = sorted(scaled_df["optimizer_id"].dropna().unique())
    active_controls = [c for c in controls if c in all_optimizers]
    candidate_optimizers = [opt for opt in all_optimizers if opt not in active_controls]

    # 4. Compute metrics per candidate against each baseline
    full_rows: List[Dict[str, Any]] = []

    for opt in candidate_optimizers:
        opt_data = scaled_df[scaled_df["optimizer_id"] == opt]
        n_tasks = len(opt_data["task_id"].unique())
        n_runs = len(opt_data)
        mean_norm = float(opt_data["normalized_regret"].mean())
        median_norm = float(opt_data["normalized_regret"].median())

        row_full: Dict[str, Any] = {
            "optimizer_id": opt,
            "n_tasks": n_tasks,
            "n_runs": n_runs,
            "mean_norm_regret": mean_norm,
            "median_norm_regret": median_norm,
        }

        opt_medians = opt_data.groupby("task_id")["normalized_regret"].median()

        for ctrl in active_controls:
            ctrl_prefix = get_control_prefix(ctrl)
            ctrl_data = scaled_df[scaled_df["optimizer_id"] == ctrl]
            ctrl_medians = ctrl_data.groupby("task_id")["normalized_regret"].median()

            common_tasks = sorted(set(opt_medians.index).intersection(set(ctrl_medians.index)))
            if not common_tasks:
                row_full[f"{ctrl_prefix}_win_rate"] = np.nan
                row_full[f"{ctrl_prefix}_rank_biserial"] = np.nan
                row_full[f"{ctrl_prefix}_wilcoxon_p"] = np.nan
                continue

            raw_opt = np.array([opt_medians[t] for t in common_tasks], dtype=float)
            raw_ctrl = np.array([ctrl_medians[t] for t in common_tasks], dtype=float)

            valid_mask = ~(np.isnan(raw_opt) | np.isnan(raw_ctrl))
            vec_opt = raw_opt[valid_mask]
            vec_ctrl = raw_ctrl[valid_mask]

            if len(vec_opt) == 0:
                row_full[f"{ctrl_prefix}_win_rate"] = np.nan
                row_full[f"{ctrl_prefix}_rank_biserial"] = np.nan
                row_full[f"{ctrl_prefix}_wilcoxon_p"] = np.nan
                continue

            diff = vec_opt - vec_ctrl
            wins = int(np.sum(diff < -tie_tolerance))
            losses = int(np.sum(diff > tie_tolerance))
            ties = len(diff) - wins - losses
            win_rate = (wins + 0.5 * ties) / len(diff)

            # Matched-pairs rank-biserial with strict tie filtering
            rb = calculate_rank_biserial_correlation(vec_opt, vec_ctrl, tie_tolerance=tie_tolerance)

            # One-sided Wilcoxon signed-rank test
            nonzeros = int(np.sum(np.abs(diff) > tie_tolerance))
            if nonzeros > 0:
                try:
                    res = wilcoxon(vec_opt, vec_ctrl, alternative="less", zero_method="wilcox")
                    p_val = float(res.pvalue)
                except Exception:
                    p_val = 1.0
            else:
                p_val = 1.0

            row_full[f"{ctrl_prefix}_win_rate"] = float(win_rate)
            row_full[f"{ctrl_prefix}_rank_biserial"] = float(rb)
            row_full[f"{ctrl_prefix}_wilcoxon_p"] = float(p_val)

        full_rows.append(row_full)

    full_df = pd.DataFrame(full_rows)

    if not full_df.empty:
        # 5. Apply Holm-Bonferroni correction
        if holm_scope == "joint":
            p_entries: List[Tuple[float, int, str]] = []
            for ctrl in active_controls:
                ctrl_prefix = get_control_prefix(ctrl)
                p_col = f"{ctrl_prefix}_wilcoxon_p"
                adj_col = f"{ctrl_prefix}_wilcoxon_adj_p"
                full_df[adj_col] = np.nan
                if p_col in full_df.columns:
                    for idx_row, r_val in full_df.iterrows():
                        if pd.notna(r_val[p_col]):
                            p_entries.append((float(r_val[p_col]), idx_row, adj_col))
            if p_entries:
                adj_p_list = apply_holm_bonferroni([e[0] for e in p_entries])
                for (_, row_idx, adj_col), adj_val in zip(p_entries, adj_p_list):
                    full_df.loc[row_idx, adj_col] = adj_val
        else:
            # Per-control family
            for ctrl in active_controls:
                ctrl_prefix = get_control_prefix(ctrl)
                p_col = f"{ctrl_prefix}_wilcoxon_p"
                adj_col = f"{ctrl_prefix}_wilcoxon_adj_p"
                full_df[adj_col] = np.nan
                if p_col in full_df.columns:
                    mask = full_df[p_col].notna()
                    if mask.any():
                        adj_p = apply_holm_bonferroni(full_df.loc[mask, p_col].tolist())
                        full_df.loc[mask, adj_col] = adj_p

        # Sort candidate approaches by performance against first control or mean_norm_regret
        if f"{get_control_prefix(active_controls[0])}_wilcoxon_adj_p" in full_df.columns:
            first_ctrl_prefix = get_control_prefix(active_controls[0])
            full_df = full_df.sort_values(
                [f"{first_ctrl_prefix}_wilcoxon_adj_p", "mean_norm_regret"]
            ).reset_index(drop=True)
        else:
            full_df = full_df.sort_values("mean_norm_regret").reset_index(drop=True)

    # 6. Build the clean, streamlined final table with ONLY requested columns
    final_cols = ["optimizer_id"]
    for ctrl in active_controls:
        ctrl_prefix = get_control_prefix(ctrl)
        final_cols.extend([
            f"{ctrl_prefix}_win_rate",
            f"{ctrl_prefix}_wilcoxon_adj_p",
            f"{ctrl_prefix}_rank_biserial",
        ])

    available_final_cols = [c for c in final_cols if c in full_df.columns]
    scorecard_df = full_df[available_final_cols].copy() if not full_df.empty else pd.DataFrame(columns=final_cols)

    return {
        "scorecard": scorecard_df,
        "full_scorecard": full_df,
        "scaled_df": scaled_df,
    }


def format_scorecard_markdown(
    scorecard_df: pd.DataFrame,
    title: str = "Benchmark Sweep Scorecard",
    controls: Sequence[str] = ("SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"),
) -> str:
    """Formats the clean scorecard DataFrame into a publication-ready Markdown table."""
    if scorecard_df.empty:
        return f"# {title}\n\n*No candidate approaches evaluated.*"

    display_df = scorecard_df.copy()

    # Create user-friendly column headers
    rename_map: Dict[str, str] = {"optimizer_id": "Optimizer"}
    for ctrl in controls:
        ctrl_prefix = get_control_prefix(ctrl)
        ctrl_short = "LCB" if "lcb" in ctrl.lower() else ("EI" if "ei" in ctrl.lower() else ctrl)
        rename_map[f"{ctrl_prefix}_win_rate"] = f"{ctrl_short} Win Rate"
        rename_map[f"{ctrl_prefix}_wilcoxon_adj_p"] = f"{ctrl_short} Adj. p"
        rename_map[f"{ctrl_prefix}_rank_biserial"] = f"{ctrl_short} $r_{{rb}}$"

    for col in display_df.columns:
        if col.endswith("_win_rate"):
            display_df[col] = display_df[col].apply(lambda v: f"{v:.1%}" if pd.notna(v) else "N/A")
        elif col.endswith("_wilcoxon_adj_p"):
            display_df[col] = display_df[col].apply(
                lambda v: f"{v:.2e}" if (pd.notna(v) and v < 0.001) else (f"{v:.4f}" if pd.notna(v) else "N/A")
            )
        elif col.endswith("_rank_biserial"):
            display_df[col] = display_df[col].apply(lambda v: f"{v:+.3f}" if pd.notna(v) else "N/A")

    display_df = display_df.rename(columns=rename_map)

    try:
        table_md = display_df.to_markdown(index=False)
    except Exception:
        table_md = display_df.to_string(index=False)

    lines = [
        f"# {title}",
        "",
        "**Statistical Notes**:",
        "- **Baselines**: Baseline optimizers serve solely as reference controls and are excluded from evaluated rows.",
        "- **Win Rate**: Proportion of benchmark tasks where candidate achieved lower normalized regret ($W + 0.5 T) / N$.",
        "- **Adj. p**: One-sided Wilcoxon signed-rank test ($H_1: \\text{cand} < \\text{baseline}$) with step-down Holm-Bonferroni correction.",
        "- **$r_{rb}$**: Matched-pairs rank-biserial correlation ($W^+ - W^-) / (W^+ + W^-$) with exact ties ($|\\Delta| \\le 10^{-6}$) strictly filtered out prior to ranking.",
        "",
        table_md,
        "",
    ]
    return "\n".join(lines)


def export_scorecard(
    results: Dict[str, Any],
    output_dir: str | Path,
    title: str = "Benchmark Sweep Scorecard",
    controls: Sequence[str] = ("SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"),
) -> Tuple[Path, Path]:
    """Exports scorecard to CSV and Markdown in the output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "scorecard.csv"
    md_path = out_dir / "scorecard.md"

    results["scorecard"].to_csv(csv_path, index=False)

    # Also save full diagnostic scorecard for archival purposes
    if "full_scorecard" in results:
        results["full_scorecard"].to_csv(out_dir / "scorecard_full_diagnostics.csv", index=False)

    md_content = format_scorecard_markdown(results["scorecard"], title=title, controls=controls)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return csv_path, md_path


def load_logs_file(file_path: str | Path) -> pd.DataFrame:
    """Loads run logs from any Parquet or CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Logs file does not exist: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        # Fallback inspection
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute standardized benchmark scorecard for any CARP-S / SMAC3 run logs."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to logs.parquet or logs.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for scorecard.csv and scorecard.md (defaults to input file parent).",
    )
    parser.add_argument(
        "--controls",
        nargs="+",
        default=["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"],
        help="Baseline optimizer IDs to use as reference controls.",
    )
    parser.add_argument(
        "--cost-col",
        type=str,
        default="trial_value__cost_inc",
        help="Column representing incumbent loss/cost (default: trial_value__cost_inc).",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=1e-6,
        help="Numerical threshold to filter out exact ties (default: 1e-6).",
    )
    parser.add_argument(
        "--holm-scope",
        type=str,
        choices=["per_control", "joint"],
        default="per_control",
        help="Multiple testing correction scope ('per_control' or 'joint').",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Benchmark Sweep Scorecard",
        help="Report title for Markdown output.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    print(f"[INFO] Loading logs from {input_path}...")
    df = load_logs_file(input_path)

    print(f"[INFO] Running statistical analysis against controls: {args.controls}...")
    results = compute_sweep_scorecard(
        df,
        controls=args.controls,
        cost_col=args.cost_col,
        tie_tolerance=args.tie_tolerance,
        holm_scope=args.holm_scope,
    )

    csv_path, md_path = export_scorecard(
        results,
        output_dir=output_dir,
        title=args.title,
        controls=args.controls,
    )

    print(f"[SUCCESS] Scorecard saved to:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Markdown: {md_path}\n")

    # Print table directly to console
    print(format_scorecard_markdown(results["scorecard"], title=args.title, controls=args.controls))


if __name__ == "__main__":
    main()


