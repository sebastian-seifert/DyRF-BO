#!/usr/bin/env python3
"""Statistical Analysis and Scorecard Generator for Multi-Suite Proximity LCB Sweeps."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import wilcoxon


def compute_paired_statistics(
    records: List[Dict[str, Any]],
    proposed_id: str = "SMAC20_ProximityLCB",
    baseline_id: str = "SMAC3_HPOFacade_lcb",
) -> Dict[str, Any]:
    """Computes paired win rates, mean differences, and Wilcoxon signed-rank test."""
    pairs: Dict[Tuple[str, int], Dict[str, float]] = {}

    # Flexible matching to tolerate hyphen/underscore variants
    def match_optimizer(opt_name: str) -> Optional[str]:
        opt_lower = opt_name.lower()
        if "proximity" in opt_lower:
            return proposed_id
        if "hpofacade" in opt_lower or "smac3" in opt_lower or "baseline" in opt_lower:
            return baseline_id
        return None

    for r in records:
        raw_opt = str(r.get("optimizer_id", ""))
        matched_opt = match_optimizer(raw_opt)
        task = r.get("task")
        seed = r.get("seed")
        loss = r.get("final_loss")
        if matched_opt and task is not None and seed is not None and loss is not None:
            key = (str(task), int(seed))
            if key not in pairs:
                pairs[key] = {}
            pairs[key][matched_opt] = float(loss)

    diffs = []
    proposed_losses = []
    baseline_losses = []

    wins_prop = 0
    wins_base = 0
    ties = 0

    for (task, seed), vals in pairs.items():
        if proposed_id in vals and baseline_id in vals:
            p_loss = vals[proposed_id]
            b_loss = vals[baseline_id]
            proposed_losses.append(p_loss)
            baseline_losses.append(b_loss)
            diff = p_loss - b_loss  # Negative diff means proposed is better (lower loss)
            diffs.append(diff)

            if np.isclose(p_loss, b_loss, atol=1e-7):
                ties += 1
            elif p_loss < b_loss:
                wins_prop += 1
            else:
                wins_base += 1

    total_pairs = len(diffs)
    if total_pairs == 0:
        found_opts = sorted(list({str(r.get("optimizer_id")) for r in records}))
        return {
            "total_pairs": 0,
            "wins_proposed": 0,
            "wins_baseline": 0,
            "ties": 0,
            "win_rate_proposed": 0.0,
            "win_rate_baseline": 0.0,
            "mean_loss_proposed": 0.0,
            "mean_loss_baseline": 0.0,
            "mean_diff": 0.0,
            "wilcoxon_stat": 0.0,
            "wilcoxon_p": 1.0,
            "found_optimizers": found_opts,
        }

    # Wilcoxon signed-rank test (two-sided)
    try:
        if all(d == 0 for d in diffs):
            wilc_stat, wilc_p = 0.0, 1.0
        else:
            res = wilcoxon(diffs, alternative="two-sided")
            wilc_stat, wilc_p = float(res.statistic), float(res.pvalue)
    except Exception:
        wilc_stat, wilc_p = 0.0, 1.0

    return {
        "total_pairs": total_pairs,
        "wins_proposed": wins_prop,
        "wins_baseline": wins_base,
        "ties": ties,
        "win_rate_proposed": wins_prop / total_pairs,
        "win_rate_baseline": wins_base / total_pairs,
        "mean_loss_proposed": float(np.mean(proposed_losses)),
        "mean_loss_baseline": float(np.mean(baseline_losses)),
        "mean_diff": float(np.mean(diffs)),
        "wilcoxon_stat": wilc_stat,
        "wilcoxon_p": wilc_p,
    }


def load_suite_records(suite_dir: Path) -> List[Dict[str, Any]]:
    """Loads run records from logs.parquet, logs.csv, or gathered_results.jsonl."""
    parquet_path = suite_dir / "logs.parquet"
    csv_path = suite_dir / "logs.csv"
    jsonl_path = suite_dir / "gathered_results.jsonl"

    if parquet_path.is_file() or csv_path.is_file():
        import pandas as pd
        if parquet_path.is_file():
            df = pd.read_parquet(parquet_path)
        else:
            df = pd.read_csv(csv_path)

        task_col = "task_id" if "task_id" in df.columns else ("task" if "task" in df.columns else None)
        opt_col = "optimizer_id" if "optimizer_id" in df.columns else None
        seed_col = "seed" if "seed" in df.columns else None
        cost_inc_col = "trial_value__cost_inc" if "trial_value__cost_inc" in df.columns else (
            "trial_value__cost" if "trial_value__cost" in df.columns else "final_loss"
        )

        if not (task_col and opt_col and seed_col and cost_inc_col in df.columns):
            return []

        grouped = df.groupby([task_col, opt_col, seed_col])[cost_inc_col].min().reset_index()
        records = []
        for _, row in grouped.iterrows():
            records.append({
                "task": str(row[task_col]),
                "optimizer_id": str(row[opt_col]),
                "seed": int(row[seed_col]),
                "final_loss": float(row[cost_inc_col]),
            })
        return records

    elif jsonl_path.is_file():
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return records

    return []


def analyze_suite(suite: str, results_base: str = "results") -> None:
    suite_dir = Path(results_base) / f"sweep_{suite}_proximity"
    records = load_suite_records(suite_dir)
    if not records:
        print(f"[WARN] No gathered results found for {suite} at {suite_dir}")
        return

    stats = compute_paired_statistics(records)
    analysis_dir = suite_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    scorecard_md = analysis_dir / f"{suite}_scorecard.md"
    content = f"""# Benchmark Scorecard: {suite}

## Evaluation Setup
- **Evaluated Pairs**: {stats['total_pairs']} runs
- **Proposed**: `SMAC20_ProximityLCB`
- **Baseline**: `SMAC3_HPOFacade_lcb` ($\\kappa=1.96$)

## Aggregate Results
| Metric | SMAC20_ProximityLCB | SMAC3_HPOFacade_lcb |
| :--- | :--- | :--- |
| **Wins** | {stats['wins_proposed']} ({stats['win_rate_proposed']*100:.1f}%) | {stats['wins_baseline']} ({stats['win_rate_baseline']*100:.1f}%) |
| **Ties** | {stats['ties']} | {stats['ties']} |
| **Mean Loss** | {stats.get('mean_loss_proposed', 0.0):.6f} | {stats.get('mean_loss_baseline', 0.0):.6f} |
| **Wilcoxon $p$-value** | **{stats['wilcoxon_p']:.4e}** | - |

*(Generated automatically from gathered results)*
"""
    with open(scorecard_md, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Generated scorecard for {suite} -> {scorecard_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute analysis scorecards for realworld sweeps.")
    parser.add_argument(
        "--suite",
        choices=["yahpo_rbv2_ranger", "yahpo_rbv2_super", "hpobench_ml", "all"],
        default="all",
    )
    args = parser.parse_args()

    suites = (
        ["yahpo_rbv2_ranger", "yahpo_rbv2_super", "hpobench_ml"]
        if args.suite == "all"
        else [args.suite]
    )

    for suite in suites:
        analyze_suite(suite)


if __name__ == "__main__":
    main()
