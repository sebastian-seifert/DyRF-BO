#!/usr/bin/env python3
"""Statistical Analysis and Scorecard Generator for BBOB High-D & Extreme Proximity LCB Sweep.

Computes:
1. Scale-invariant Normalized Final Regret per task and dimension stratum (D=16, D=32, overall).
2. Non-parametric paired tests: Wilcoxon signed-rank test (two-sided, Demšar 2006 compliant).
3. Family-Wise Error Rate (FWER) control via Holm-Bonferroni step-down correction.
4. Effect size: Cliff's delta on paired evaluation runs.
5. Stratified scorecards (Markdown + CSV) saved to results directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def calculate_cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Computes Cliff's delta non-parametric effect size between two distributions.

    Uses the Mann-Whitney U relationship (delta = 2*U / (n_x * n_y) - 1) for
    O((N+M) log(N+M)) performance and minimal memory consumption.

    Returns:
        float in [-1, 1], where negative indicates x tends to be smaller (better for minimization).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(x, y, alternative="two-sided")
        return float((2.0 * res.statistic) / (n_x * n_y) - 1.0)
    except Exception:
        greater = 0
        less = 0
        for val_x in x:
            greater += int(np.sum(val_x > y))
            less += int(np.sum(val_x < y))
        return float((greater - less) / (n_x * n_y))


def apply_holm_bonferroni(p_vals: Sequence[float]) -> List[float]:
    """Computes Holm-Bonferroni step-down adjusted p-values ensuring FWER control and monotonicity.

    Args:
        p_vals: List or array of unadjusted p-values.

    Returns:
        List of Holm-Bonferroni adjusted p-values in the original input order.
    """
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


def _extract_dimension(task_name: str) -> int:
    """Extracts dimensionality from task name (e.g. cfg_16_1_0 -> 16)."""
    if "16_" in task_name or "/16/" in task_name:
        return 16
    if "32_" in task_name or "/32/" in task_name:
        return 32
    return 0


def compute_statistical_comparison(
    df: pd.DataFrame,
    proposed_id: str = "SMAC20_ProximityLCB",
    baseline_id: str = "SMAC3_HPOFacade_lcb",
    cost_col: Optional[str] = None,
    task_col: Optional[str] = None,
    opt_col: Optional[str] = None,
    seed_col: str = "seed",
    tie_tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Computes stratified paired statistical comparisons across all tasks, D=16, and D=32."""
    if task_col is None:
        task_col = "task" if "task" in df.columns else "task_id"
    if opt_col is None:
        opt_col = "optimizer" if "optimizer" in df.columns else "optimizer_id"
    if cost_col is None:
        for c in ["final_cost", "trial_value__cost_inc", "cost"]:
            if c in df.columns:
                cost_col = c
                break
        if cost_col is None:
            raise KeyError(f"Could not find cost column in df. Available: {list(df.columns)}")

    # Detect trial column to collapse multiple evaluation steps per run down to final incumbent
    trial_col = None
    for cand in ["n_trials", "n_function_calls", "trial", "trial_number", "trial_idx", "iteration", "step"]:
        if cand in df.columns:
            trial_col = cand
            break

    # Select only required columns to minimize memory footprint
    needed_cols = [opt_col, task_col, seed_col, cost_col]
    if "dimension" in df.columns:
        needed_cols.append("dimension")
    if trial_col and trial_col not in needed_cols:
        needed_cols.append(trial_col)

    df_clean = df[[c for c in needed_cols if c in df.columns]].copy()

    if "dimension" not in df_clean.columns:
        df_clean["dimension"] = df_clean[task_col].apply(_extract_dimension)

    # CRITICAL OOM & LOGIC FIX:
    # If df contains multiple trials per run (e.g. n_trials = 1..100), extract the final trial
    # per (optimizer, task, seed) to prevent an explosive Cartesian product during pairing.
    if trial_col is not None:
        df_clean = df_clean.sort_values(by=trial_col).drop_duplicates(
            subset=[opt_col, task_col, seed_col], keep="last"
        ).reset_index(drop=True)
    elif df_clean.duplicated(subset=[opt_col, task_col, seed_col]).any():
        df_clean = df_clean.drop_duplicates(
            subset=[opt_col, task_col, seed_col], keep="last"
        ).reset_index(drop=True)

    df_p = df_clean[df_clean[opt_col] == proposed_id]
    df_b = df_clean[df_clean[opt_col] == baseline_id]

    if df_p.empty or df_b.empty:
        # Check if IDs match prefix
        p_match = [o for o in df_clean[opt_col].unique() if "ProximityLCB" in str(o)]
        b_match = [o for o in df_clean[opt_col].unique() if "SMAC3_HPOFacade" in str(o) or "hpo" in str(o).lower()]
        if p_match and b_match:
            proposed_id = p_match[0]
            baseline_id = b_match[0]
            df_p = df_clean[df_clean[opt_col] == proposed_id]
            df_b = df_clean[df_clean[opt_col] == baseline_id]

    merged = pd.merge(
        df_p[[task_col, seed_col, "dimension", cost_col]],
        df_b[[task_col, seed_col, "dimension", cost_col]],
        on=[task_col, seed_col, "dimension"],
        suffixes=("_proposed", "_baseline"),
    )

    if merged.empty:
        raise ValueError(f"No paired runs found between '{proposed_id}' and '{baseline_id}'.")

    results: Dict[str, Any] = {}

    def analyze_subset(sub_df: pd.DataFrame, label: str) -> Dict[str, Any]:
        p_vals = sub_df[f"{cost_col}_proposed"].to_numpy(dtype=float)
        b_vals = sub_df[f"{cost_col}_baseline"].to_numpy(dtype=float)
        valid = np.isfinite(p_vals) & np.isfinite(b_vals)
        p_v = p_vals[valid]
        b_v = b_vals[valid]

        diff = p_v - b_v
        wins = int(np.sum(diff < -tie_tolerance))
        losses = int(np.sum(diff > tie_tolerance))
        ties = int(np.sum(np.abs(diff) <= tie_tolerance))
        total = len(diff)
        win_rate = (wins / total) if total > 0 else 0.0

        cliffs_d = calculate_cliffs_delta(p_v, b_v)

        try:
            nonzero = diff[np.abs(diff) > tie_tolerance]
            if len(nonzero) >= 5:
                stat, p_val = wilcoxon(p_v, b_v, alternative="two-sided")
            else:
                stat, p_val = np.nan, 1.0
        except Exception:
            stat, p_val = np.nan, 1.0

        # Per-task normalized regret aggregation
        task_norm_p: List[float] = []
        task_norm_b: List[float] = []
        task_details: List[Dict[str, Any]] = []

        # Vectorized groupby iteration instead of quadratic sub_df[sub_df[task_col] == task_name]
        for task_name, t_sub in sub_df.groupby(task_col):
            tp = t_sub[f"{cost_col}_proposed"].to_numpy(dtype=float)
            tb = t_sub[f"{cost_col}_baseline"].to_numpy(dtype=float)
            v_mask = np.isfinite(tp) & np.isfinite(tb)
            tp_v = tp[v_mask]
            tb_v = tb[v_mask]

            t_all = np.concatenate([tp_v, tb_v])
            if len(t_all) > 0:
                t_min = float(np.min(t_all))
                t_max = float(np.max(t_all))
                spread = t_max - t_min
            else:
                t_min, t_max, spread = 0.0, 0.0, 0.0

            if spread > 1e-12:
                np_mean = float(np.mean((tp_v - t_min) / spread))
                nb_mean = float(np.mean((tb_v - t_min) / spread))
            else:
                np_mean, nb_mean = 0.0, 0.0

            task_norm_p.append(np_mean)
            task_norm_b.append(nb_mean)

            t_diff = tp_v - tb_v
            t_w = int(np.sum(t_diff < -tie_tolerance))
            t_l = int(np.sum(t_diff > tie_tolerance))
            t_t = int(np.sum(np.abs(t_diff) <= tie_tolerance))

            try:
                if len(t_diff[np.abs(t_diff) > tie_tolerance]) >= 5:
                    _, t_pval = wilcoxon(tp_v, tb_v, alternative="two-sided")
                else:
                    t_pval = 1.0
            except Exception:
                t_pval = 1.0

            task_details.append({
                "task": task_name,
                "dimension": int(t_sub["dimension"].iloc[0]),
                "n_seeds": len(tp_v),
                "norm_regret_proposed": np_mean,
                "norm_regret_baseline": nb_mean,
                "wins": t_w,
                "losses": t_l,
                "ties": t_t,
                "cliffs_delta": calculate_cliffs_delta(tp_v, tb_v),
                "raw_p_value": float(t_pval),
            })

        # Apply Holm-Bonferroni across tasks in this stratum
        raw_ps = [td["raw_p_value"] for td in task_details]
        adj_ps = apply_holm_bonferroni(raw_ps)
        for td, adj_p in zip(task_details, adj_ps):
            td["adj_p_value"] = adj_p

        # Demšar task-level Wilcoxon test on mean normalized regrets
        if len(task_norm_p) >= 5:
            regret_diff = np.array(task_norm_p) - np.array(task_norm_b)
            if np.any(np.abs(regret_diff) > tie_tolerance):
                try:
                    demsar_stat, demsar_p = wilcoxon(task_norm_p, task_norm_b, alternative="two-sided")
                except Exception:
                    demsar_stat, demsar_p = np.nan, 1.0
            else:
                demsar_stat, demsar_p = 0.0, 1.0
        else:
            demsar_stat, demsar_p = np.nan, 1.0

        return {
            "stratum": label,
            "n_tasks": len(sub_df[task_col].unique()),
            "n_paired_runs": total,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate_proposed": win_rate,
            "cliffs_delta_all_runs": cliffs_d,
            "wilcoxon_all_runs_p": float(p_val),
            "demsar_task_wilcoxon_p": float(demsar_p),
            "mean_norm_regret_proposed": float(np.mean(task_norm_p)) if task_norm_p else 0.0,
            "mean_norm_regret_baseline": float(np.mean(task_norm_b)) if task_norm_b else 0.0,
            "task_details": task_details,
        }

    results["overall"] = analyze_subset(merged, "Overall (D=16 & D=32)")

    merged_16 = merged[merged["dimension"] == 16]
    if not merged_16.empty:
        results["dim_16"] = analyze_subset(merged_16, "High-D (D=16)")
    else:
        results["dim_16"] = {"n_tasks": 0}

    merged_32 = merged[merged["dimension"] == 32]
    if not merged_32.empty:
        results["dim_32"] = analyze_subset(merged_32, "Extreme (D=32)")
    else:
        results["dim_32"] = {"n_tasks": 0}

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Statistical Comparison for BBOB High-D & Extreme Proximity Sweep"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/sweep_bbob_highdim_proximity/logs.parquet",
        help="Input parquet or CSV file containing run logs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sweep_bbob_highdim_proximity/analysis",
        help="Directory to save analysis scorecards and CSVs",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.input)
    if not in_path.exists():
        csv_alt = in_path.with_suffix(".csv")
        if csv_alt.exists():
            in_path = csv_alt
        else:
            print(f"[ERROR] Input file not found: {in_path}")
            sys.exit(1)

    print(f"Reading logs from {in_path}...")
    if in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        # Memory optimization: read only candidate analysis columns if reading CSV
        try:
            sample_cols = pd.read_csv(in_path, nrows=0).columns.tolist()
            cand_cols = [
                "task", "task_id", "optimizer", "optimizer_id", "seed", "dimension",
                "final_cost", "trial_value__cost_inc", "trial_value__cost", "cost",
                "n_trials", "n_function_calls", "trial", "step", "iteration"
            ]
            load_cols = [c for c in cand_cols if c in sample_cols]
            if len(load_cols) >= 4:
                df = pd.read_csv(in_path, usecols=load_cols)
            else:
                df = pd.read_csv(in_path)
        except Exception:
            df = pd.read_csv(in_path)

    print(f"Loaded {len(df)} records. Computing statistical comparison...")
    res = compute_statistical_comparison(df)

    # Save summary markdown
    md_path = out_dir / "bbob_highdim_proximity_scorecard.md"
    with open(md_path, "w") as f:
        f.write("# BBOB High-D & Extreme Evaluation Scorecard\n\n")
        f.write("Evaluation comparing **SMAC20_ProximityLCB** ($k=25, \\lambda=1.345, \\epsilon=0.16$) ")
        f.write("vs **SMAC3_HPOFacade_lcb** ($\\beta=3.8416$) across 30 seeds.\n\n")
        for strat_key, strat_data in [("overall", res["overall"]), ("dim_16", res["dim_16"]), ("dim_32", res["dim_32"])]:
            if strat_data.get("n_tasks", 0) == 0:
                continue
            f.write(f"## {strat_data.get('stratum', strat_key)}\n")
            f.write(f"- **Tasks Evaluated**: {strat_data['n_tasks']}\n")
            f.write(f"- **Paired Runs**: {strat_data['n_paired_runs']}\n")
            f.write(f"- **Win / Loss / Tie**: {strat_data['wins']} / {strat_data['losses']} / {strat_data['ties']} ")
            f.write(f"(Win Rate: {strat_data['win_rate_proposed']:.1%})\n")
            f.write(f"- **Cliff's Delta**: {strat_data['cliffs_delta_all_runs']:.4f}\n")
            f.write(f"- **Demšar Task-Level Wilcoxon p-value**: {strat_data['demsar_task_wilcoxon_p']:.4e}\n")
            f.write(f"- **Mean Normalized Regret (Proposed)**: {strat_data['mean_norm_regret_proposed']:.4f}\n")
            f.write(f"- **Mean Normalized Regret (Baseline)**: {strat_data['mean_norm_regret_baseline']:.4f}\n\n")

    print(f"[SUCCESS] Scorecard saved to {md_path}")

    # Save stratified summary CSV
    strat_rows = []
    for strat_key, strat_data in [("overall", res["overall"]), ("dim_16", res["dim_16"]), ("dim_32", res["dim_32"])]:
        if strat_data.get("n_tasks", 0) > 0:
            strat_rows.append({
                "stratum": strat_data.get("stratum", strat_key),
                "n_tasks": strat_data["n_tasks"],
                "n_paired_runs": strat_data["n_paired_runs"],
                "wins": strat_data["wins"],
                "losses": strat_data["losses"],
                "ties": strat_data["ties"],
                "win_rate_proposed": strat_data["win_rate_proposed"],
                "cliffs_delta_all_runs": strat_data["cliffs_delta_all_runs"],
                "demsar_task_wilcoxon_p": strat_data["demsar_task_wilcoxon_p"],
                "mean_norm_regret_proposed": strat_data["mean_norm_regret_proposed"],
                "mean_norm_regret_baseline": strat_data["mean_norm_regret_baseline"],
            })
    if strat_rows:
        strat_csv = out_dir / "stratified_scorecard.csv"
        pd.DataFrame(strat_rows).to_csv(strat_csv, index=False)
        print(f"[SUCCESS] Stratified scorecard saved to {strat_csv}")

    # Save detailed per-task CSV
    if "task_details" in res.get("overall", {}):
        task_csv = out_dir / "task_details.csv"
        pd.DataFrame(res["overall"]["task_details"]).to_csv(task_csv, index=False)
        print(f"[SUCCESS] Task details saved to {task_csv}")


if __name__ == "__main__":
    main()
