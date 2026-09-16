#!/usr/bin/env python3
"""Statistical Analysis Suite for BBOB High-D & Extreme Evaluation (Proximity LCB vs SMAC3 EI).

Compares Tuned Proximity LCB (SMAC20_ProximityLCB) vs. Standard SMAC3 EI Baseline (SMAC3_HPOFacade_ei)
across 144 BBOB tasks (72 at D=16, 72 at D=32) and 30 seeds.

Key Methodological Features:
1. Memory-Safe Pre-Merge Trial Reduction (prevents multi-step Cartesian joins and OOM kills).
2. O((N+M)log(N+M)) Cliff's Delta computation via Mann-Whitney U.
3. Scale-Invariant Normalized Regret calculation per task.
4. Step-down Holm-Bonferroni FWER correction across all 144 individual task hypotheses.
5. Function-Aggregated Wilcoxon Signed-Rank Test across the 24 canonical BBOB functions
   (satisfies Demšar 2006 sample independence and eliminates instance pseudo-replication).
6. Dimensionality-Stratified Scorecards (Overall, D=16, D=32) and 5 BBOB Landscape Groups.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu, wilcoxon


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame to a clean GitHub-flavored markdown table."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        row_str = [str(val) for val in row.values]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)


def calculate_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cliff's delta non-parametric effect size in O((N+M) log(N+M))."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    u_stat, _ = mannwhitneyu(x, y, alternative="two-sided")
    return float((2.0 * u_stat / (n_x * n_y)) - 1.0)


def apply_holm_bonferroni(p_vals: Sequence[float]) -> List[float]:
    """Computes Holm-Bonferroni step-down adjusted p-values ensuring FWER control and monotonicity."""
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
    """Extracts dimensionality from task name (e.g. cfg_16_1_0 or /16/ -> 16)."""
    if "16_" in str(task_name) or "/16/" in str(task_name):
        return 16
    if "32_" in str(task_name) or "/32/" in str(task_name):
        return 32
    return 0


def _extract_function_id(task_name: str) -> int:
    """Extracts canonical BBOB function number (1-24) from task string."""
    m = re.search(r"bbob/\d+/(\d+)/", str(task_name)) or re.search(r"cfg_\d+_(\d+)_", str(task_name))
    if m:
        return int(m.group(1))
    return 0


def _get_bbob_group(fid: int) -> str:
    """Maps BBOB function ID to its standard landscape group."""
    if 1 <= fid <= 5:
        return "1. Separable (F1-F5)"
    elif 6 <= fid <= 9:
        return "2. Moderate Conditioning (F6-F9)"
    elif 10 <= fid <= 14:
        return "3. Ill-Conditioned (F10-F14)"
    elif 15 <= fid <= 19:
        return "4. Multi-modal w/ Structure (F15-F19)"
    elif 20 <= fid <= 24:
        return "5. Multi-modal w/ Weak Structure (F20-F24)"
    return "Unknown"


def compute_bbob_highdim_proximity_vs_ei_analysis(
    input_file: str = "results/sweep_bbob_highdim_proximity_vs_ei/logs.csv",
    output_dir: str = "results/sweep_bbob_highdim_proximity_vs_ei/analysis",
    proposed_id: str = "SMAC20_ProximityLCB",
    baseline_id: str = "SMAC3_HPOFacade_ei",
    cost_col: Optional[str] = None,
    tie_tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """Computes statistical comparison between Proximity LCB and SMAC3 EI on BBOB High-D suite."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    in_path = Path(input_file)
    if in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    # Standardize column names
    trial_col = None
    for c in ["n_trials", "n_function_calls", "trial", "trial_number", "trial_idx", "step"]:
        if c in df.columns:
            trial_col = c
            break

    if cost_col is None:
        for c in ["trial_value__cost_inc", "cost_inc", "trial_value__cost", "cost", "value"]:
            if c in df.columns:
                cost_col = c
                break

    task_col = "task_id" if "task_id" in df.columns else "task"

    # Memory-Safe Pre-Merge Reduction: select final trial incumbent per run
    if trial_col:
        df = df.sort_values(by=["optimizer_id", task_col, "seed", trial_col])
        df = df.drop_duplicates(subset=["optimizer_id", task_col, "seed"], keep="last")
    else:
        df = df.drop_duplicates(subset=["optimizer_id", task_col, "seed"], keep="last")

    p_df = df[df["optimizer_id"] == proposed_id]
    b_df = df[df["optimizer_id"] == baseline_id]

    if p_df.empty or b_df.empty:
        raise ValueError(f"Missing optimizer data: found {p_df['optimizer_id'].nunique()} proposed and {b_df['optimizer_id'].nunique()} baseline runs.")

    # Merge strictly paired runs
    merged = pd.merge(
        p_df[[task_col, "seed", cost_col]],
        b_df[[task_col, "seed", cost_col]],
        on=[task_col, "seed"],
        suffixes=("_proposed", "_baseline"),
    )

    if merged.empty:
        raise ValueError("No paired runs found between proposed and baseline.")

    merged["dim"] = merged[task_col].apply(_extract_dimension)
    merged["fid"] = merged[task_col].apply(_extract_function_id)

    # Compute per-task statistics
    task_details: List[Dict[str, Any]] = []
    unique_tasks = sorted(merged[task_col].unique())

    for task in unique_tasks:
        sub = merged[merged[task_col] == task]
        p_vals = sub[f"{cost_col}_proposed"].to_numpy(dtype=float)
        b_vals = sub[f"{cost_col}_baseline"].to_numpy(dtype=float)
        valid = np.isfinite(p_vals) & np.isfinite(b_vals)
        pv, bv = p_vals[valid], b_vals[valid]

        diff = pv - bv
        w = int(np.sum(diff < -tie_tolerance))
        l = int(np.sum(diff > tie_tolerance))
        t = int(np.sum(np.abs(diff) <= tie_tolerance))

        # Raw seed Wilcoxon per task
        if len(diff[np.abs(diff) > tie_tolerance]) >= 5:
            try:
                _, pval = wilcoxon(pv, bv, alternative="two-sided")
            except Exception:
                pval = 1.0
        else:
            pval = 1.0

        # Scale-invariant normalized regret
        all_c = np.concatenate([pv, bv])
        y_min, y_max = float(np.min(all_c)), float(np.max(all_c))
        spread = y_max - y_min
        if spread > 1e-12:
            norm_p = float(np.mean((pv - y_min) / spread))
            norm_b = float(np.mean((bv - y_min) / spread))
        else:
            norm_p, norm_b = 0.0, 0.0

        delta = calculate_cliffs_delta(pv, bv)

        task_details.append({
            "task": task,
            "dimension": _extract_dimension(task),
            "fid": _extract_function_id(task),
            "n_seeds": len(pv),
            "norm_regret_proposed": norm_p,
            "norm_regret_baseline": norm_b,
            "wins": w,
            "losses": l,
            "ties": t,
            "cliffs_delta": delta,
            "raw_p_value": float(pval),
        })

    # Apply Holm-Bonferroni correction across the 144 tasks
    raw_ps = [td["raw_p_value"] for td in task_details]
    adj_ps = apply_holm_bonferroni(raw_ps)
    for td, adj_p in zip(task_details, adj_ps):
        td["adj_p_value"] = adj_p

    task_df = pd.DataFrame(task_details)
    task_df.to_csv(out_path / "task_details.csv", index=False)

    # Function-Level Aggregation (Demšar-compliant: N=24 independent functions)
    fn_df = task_df.groupby("fid").agg({
        "norm_regret_proposed": "mean",
        "norm_regret_baseline": "mean",
        "wins": "sum",
        "losses": "sum",
        "ties": "sum",
    }).reset_index()

    fn_df["prop_wins"] = fn_df["norm_regret_proposed"] < fn_df["norm_regret_baseline"]
    fn_w_stat, fn_w_p = wilcoxon(fn_df["norm_regret_proposed"], fn_df["norm_regret_baseline"], alternative="two-sided")
    fn_binom = binomtest(k=int(fn_df["prop_wins"].sum()), n=len(fn_df), p=0.5, alternative="greater")

    fn_df["group"] = fn_df["fid"].apply(_get_bbob_group)
    fn_df.to_csv(out_path / "function_details.csv", index=False)

    # Dimensionality Stratifications
    strata = [
        ("Overall (D=16 & D=32)", task_df),
        ("High-D (D=16)", task_df[task_df["dimension"] == 16]),
        ("Extreme (D=32)", task_df[task_df["dimension"] == 32]),
    ]

    scorecard_rows = []
    for label, sub_tdf in strata:
        n_tasks = len(sub_tdf)
        sub_merged = merged[merged["dim"].isin([16, 32] if "Overall" in label else [16] if "16" in label else [32])]
        diff_all = sub_merged[f"{cost_col}_proposed"].values - sub_merged[f"{cost_col}_baseline"].values
        w_all = int(np.sum(diff_all < -tie_tolerance))
        l_all = int(np.sum(diff_all > tie_tolerance))
        t_all = int(np.sum(np.abs(diff_all) <= tie_tolerance))
        total_runs = len(diff_all)

        m_p = float(sub_tdf["norm_regret_proposed"].mean())
        m_b = float(sub_tdf["norm_regret_baseline"].mean())
        rel_red = float((m_b - m_p) / m_b * 100.0) if m_b > 1e-12 else 0.0

        p_reg = sub_tdf["norm_regret_proposed"].values
        b_reg = sub_tdf["norm_regret_baseline"].values
        if int(np.sum(np.abs(p_reg - b_reg) > 1e-12)) >= 5:
            try:
                _, d_p = wilcoxon(p_reg, b_reg, alternative="two-sided")
            except Exception:
                d_p = 1.0
        else:
            d_p = 1.0

        delta_overall = calculate_cliffs_delta(sub_merged[f"{cost_col}_proposed"].values, sub_merged[f"{cost_col}_baseline"].values)

        scorecard_rows.append({
            "Stratum": label,
            "Tasks": n_tasks,
            "Paired Runs": total_runs,
            "Win / Loss / Tie": f"{w_all} / {l_all} / {t_all}",
            "Win Rate": f"{w_all / total_runs * 100:.1f}%",
            "Mean Regret (Prop vs Base)": f"{m_p:.4f} vs {m_b:.4f}",
            "Regret Reduction": f"{rel_red:+.1f}%",
            "Cliff's Delta": f"{delta_overall:+.4f}",
            "Demšar Wilcoxon p": f"{d_p:.4e}",
        })

    scorecard_df = pd.DataFrame(scorecard_rows)
    scorecard_df.to_csv(out_path / "stratified_scorecard.csv", index=False)

    # 5 BBOB Problem Classes
    task_df["group"] = task_df["fid"].apply(_get_bbob_group)
    group_df = task_df.groupby("group").agg({
        "task": "count",
        "wins": "sum",
        "losses": "sum",
        "ties": "sum",
        "norm_regret_proposed": "mean",
        "norm_regret_baseline": "mean",
        "cliffs_delta": "mean",
    }).reset_index()

    group_df["win_rate"] = group_df["wins"] / (group_df["wins"] + group_df["losses"] + group_df["ties"])
    group_df["rel_red"] = (group_df["norm_regret_baseline"] - group_df["norm_regret_proposed"]) / group_df["norm_regret_baseline"] * 100.0

    # Build Markdown Scorecard
    md_lines = [
        "# BBOB High-D & Extreme Evaluation Scorecard: Proximity LCB vs SMAC3 EI",
        "",
        "Evaluation comparing **SMAC20_ProximityLCB** ($k=25, \\lambda=1.345, \\epsilon=0.16$) vs **SMAC3_HPOFacade_ei** (native Expected Improvement $\\xi=0.0$) across 30 seeds.",
        "",
        "## 1. Aggregate Scorecard by Dimensionality",
        "",
        dataframe_to_markdown(scorecard_df),
        "",
        "## 2. Independent Function-Aggregated Synthesis (Demšar, 2006)",
        "",
        f"- **Canonical BBOB Functions Evaluated**: 24",
        f"- **Functions Won by Proposed**: **{int(fn_df['prop_wins'].sum())} / {len(fn_df)}** ({fn_df['prop_wins'].sum() / len(fn_df) * 100:.1f}%)",
        f"- **Wilcoxon Signed-Rank p-value (across 24 functions)**: `{fn_w_stat:.1f}, p = {fn_w_p:.4e}`",
        f"- **Exact Binomial Sign Test p-value**: `{fn_binom.pvalue:.4e}`",
        "",
        "## 3. Landscape Breakdown by BBOB Problem Class",
        "",
    ]

    disp_group = group_df[["group", "task", "win_rate", "norm_regret_proposed", "norm_regret_baseline", "rel_red", "cliffs_delta"]].copy()
    disp_group.columns = ["Problem Class", "Tasks", "Win Rate", "Regret Proposed", "Regret Baseline", "Rel. Reduction", "Mean Cliff's Delta"]
    disp_group["Win Rate"] = disp_group["Win Rate"].apply(lambda v: f"{v * 100:.1f}%")
    disp_group["Regret Proposed"] = disp_group["Regret Proposed"].apply(lambda v: f"{v:.4f}")
    disp_group["Regret Baseline"] = disp_group["Regret Baseline"].apply(lambda v: f"{v:.4f}")
    disp_group["Rel. Reduction"] = disp_group["Rel. Reduction"].apply(lambda v: f"{v:+.1f}%")
    disp_group["Mean Cliff's Delta"] = disp_group["Mean Cliff's Delta"].apply(lambda v: f"{v:+.3f}")

    md_lines.append(dataframe_to_markdown(disp_group))
    md_lines.append("")

    with open(out_path / "bbob_highdim_proximity_vs_ei_scorecard.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"Analysis successfully generated in {output_dir}:")
    print(f"  - Markdown Scorecard: {out_path / 'bbob_highdim_proximity_vs_ei_scorecard.md'}")
    print(f"  - Stratified Scorecard CSV: {out_path / 'stratified_scorecard.csv'}")
    print(f"  - Task Details CSV: {out_path / 'task_details.csv'}")
    print(f"  - Function Details CSV: {out_path / 'function_details.csv'}")

    return {
        "scorecard_df": scorecard_df,
        "function_df": fn_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze BBOB High-D Proximity LCB vs SMAC3 EI results.")
    parser.add_argument("--input-file", "--input_file", type=str, default="results/sweep_bbob_highdim_proximity_vs_ei/logs.csv")
    parser.add_argument("--output-dir", "--output_dir", type=str, default="results/sweep_bbob_highdim_proximity_vs_ei/analysis")
    parser.add_argument("--proposed-id", "--proposed_id", type=str, default="SMAC20_ProximityLCB")
    parser.add_argument("--baseline-id", "--baseline_id", type=str, default="SMAC3_HPOFacade_ei")
    parser.add_argument("--cost-col", "--cost_col", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_bbob_highdim_proximity_vs_ei_analysis(
        input_file=args.input_file,
        output_dir=args.output_dir,
        proposed_id=args.proposed_id,
        baseline_id=args.baseline_id,
        cost_col=args.cost_col,
    )


if __name__ == "__main__":
    main()
