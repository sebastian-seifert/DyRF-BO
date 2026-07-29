#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np


def merge_carps_results(prev_tables_dir, auto_tables_dir, output_tables_dir):
    """
    Merges baseline CARP-S results with proximity_auto_lambda results across all 26 tasks.
    Computes overall average ranks, total wins, and top-3 finishes.
    """
    os.makedirs(output_tables_dir, exist_ok=True)
    prev_files = glob.glob(os.path.join(prev_tables_dir, "*_comparison.csv"))
    
    task_tables = {}
    ranks = {}
    wins = {}
    top3 = {}

    for prev_path in prev_files:
        filename = os.path.basename(prev_path)
        task_name = filename.replace("_comparison.csv", "")
        auto_path = os.path.join(auto_tables_dir, filename)

        df_prev = pd.read_csv(prev_path)
        if os.path.exists(auto_path):
            df_auto = pd.read_csv(auto_path)
            df_combined = pd.concat([df_prev, df_auto], ignore_index=True)
        else:
            df_combined = df_prev.copy()

        # Sort by Mean_Final_Cost ascending
        df_combined = df_combined.sort_values(by="Mean_Final_Cost", ascending=True).reset_index(drop=True)
        
        # Save merged table
        out_path = os.path.join(output_tables_dir, filename)
        df_combined.to_csv(out_path, index=False)
        task_tables[task_name] = df_combined

        # Track ranks
        for rank_idx, row in df_combined.iterrows():
            app = row["Approach"]
            r = rank_idx + 1
            ranks.setdefault(app, []).append(r)
            if r == 1:
                wins[app] = wins.get(app, 0) + 1
            if r <= 3:
                top3[app] = top3.get(app, 0) + 1

    summary_data = []
    all_apps = sorted(list(ranks.keys()))
    for app in all_apps:
        app_ranks = ranks[app]
        avg_rank = np.mean(app_ranks)
        summary_data.append({
            "Approach": app,
            "Overall Avg Rank (1=Best)": round(avg_rank, 2),
            "Wins (1st Place)": wins.get(app, 0),
            "Top 3 Finishes": top3.get(app, 0),
            "Tasks Evaluated": len(app_ranks)
        })

    df_summary = pd.DataFrame(summary_data).sort_values(by="Overall Avg Rank (1=Best)", ascending=True).reset_index(drop=True)
    return df_summary, task_tables


def to_markdown_simple(df, index=False):
    """Simple markdown table generator without tabulate dependency."""
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_comparison_report(df_summary, task_tables, report_filepath):
    """
    Generates a clean GitHub Markdown report summarizing the comparison.
    """
    lines = []
    lines.append("# CARP-S Benchmark Comparison: Auto Lambda Proximity vs. Baselines")
    lines.append("")
    lines.append("This report compares the continuous Out-of-Bag (OOB) `proximity_auto_lambda` approach against all 8 previously evaluated Dynamic RF UQ extractors and standard SMAC3 baseline across all 26 CARP-S benchmark tasks.")
    lines.append("")
    lines.append("## Executive Summary: Overall Performance & Ranking")
    lines.append("")
    lines.append(to_markdown_simple(df_summary, index=False))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Per-Task Benchmark Comparison")
    lines.append("")

    for task_name, df_task in task_tables.items():
        lines.append(f"### Benchmark Task: `{task_name}`")
        lines.append("")
        lines.append(to_markdown_simple(df_task, index=False))
        lines.append("")
        lines.append("---")

    report_content = "\n".join(lines)
    os.makedirs(os.path.dirname(report_filepath), exist_ok=True)
    with open(report_filepath, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Comparison report saved to: {report_filepath}")


def main():
    prev_dir = "results/carps_summary_21072026/tables"
    auto_dir = "results/carps_auto_lambda_summary_22072026/tables"
    out_dir = "results/carps_auto_lambda_comparison_22072026/tables"
    report_file = "results/carps_auto_lambda_comparison_22072026/summary_report.md"

    df_summary, task_tables = merge_carps_results(prev_dir, auto_dir, out_dir)
    generate_comparison_report(df_summary, task_tables, report_file)


if __name__ == "__main__":
    main()
