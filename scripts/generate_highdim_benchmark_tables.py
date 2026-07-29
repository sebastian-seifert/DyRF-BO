#!/usr/bin/env python3
"""
Generates per-benchmark comparison tables across all seeds comparing
standard smac3bo against custom DyRF Epistemic UQ approaches for high-dim tasks.
"""

import os
import re
import glob
import json
import numpy as np
import argparse

def normalize_task_name(raw_name: str) -> str:
    if "rbv2_super" in raw_name:
        match = re.search(r"(\d+)", raw_name)
        # Match task ID numbers like 1040, 1049, 15, etc.
        ids = re.findall(r"\b\d+\b", raw_name)
        if ids:
            return f"cfg_rbv2_super_{ids[-1]}"
    if "nb301" in raw_name or "CIFAR10" in raw_name:
        return "cfg_nb301_CIFAR10"
    return raw_name.replace("/", "_")

def parse_highdim_telemetry(dirs=None):
    if dirs is None:
        dirs = ["results/epistemic_ei_highdim/baseline", "results/epistemic_ei_highdim/ei"]

    results = {}
    for d in dirs:
        if not os.path.exists(d):
            continue
        for file_path in glob.glob(os.path.join(d, "telemetry_*.json")):
            filename = os.path.basename(file_path)
            try:
                with open(file_path, "r") as f:
                    content = json.load(f)
                
                raw_task = content.get("task_name", "unknown")
                task = normalize_task_name(raw_task)
                extractor = content.get("extractor_name", "unknown")
                seed = int(content.get("seed", 1))
                trials = content.get("trials", [])

                if not trials:
                    continue

                costs = []
                for t in trials:
                    if isinstance(t, dict):
                        if "cost" in t and isinstance(t["cost"], (int, float)):
                            costs.append(t["cost"])
                        elif "trial_value" in t and isinstance(t["trial_value"], dict) and "cost" in t["trial_value"]:
                            costs.append(t["trial_value"]["cost"])

                if not costs:
                    continue

                best_cost = min(costs)
                results.setdefault(task, {}).setdefault(extractor, {})[seed] = float(best_cost)
            except Exception as e:
                print(f"Warning: Error parsing {filename}: {e}")

    return results

def format_benchmark_table(task_name: str, approaches_dict: dict) -> str:
    lines = []
    lines.append(f"## Benchmark Task: `{task_name}`\n")
    lines.append("| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    stats_list = []
    for app_name, seed_map in approaches_dict.items():
        costs = list(seed_map.values())
        if not costs:
            continue
        mean_c = float(np.mean(costs))
        std_c = float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0
        se_c = std_c / np.sqrt(len(costs)) if len(costs) > 1 else 0.0
        best_c = float(np.min(costs))
        worst_c = float(np.max(costs))
        n_seeds = len(costs)

        stats_list.append((app_name, mean_c, std_c, se_c, f"{n_seeds}/5", best_c, worst_c))

    # Sort ascending by Mean Final Cost (lower cost = better performance)
    stats_list.sort(key=lambda x: x[1])

    for app_name, mean_c, std_c, se_c, n_seeds, best_c, worst_c in stats_list:
        lines.append(
            f"| `{app_name}` | {mean_c:.6f} | {std_c:.6f} | {se_c:.6f} | {n_seeds} | {best_c:.6f} | {worst_c:.6f} |"
        )

    lines.append("\n")
    return "\n".join(lines)

def generate_all_highdim_reports(
    input_dirs=None,
    output_report="results/epistemic_ei_highdim/highdim_benchmark_tables.md",
    output_tables_dir="results/epistemic_ei_highdim/tables"
):
    if input_dirs is None:
        input_dirs = ["results/epistemic_ei_highdim/baseline", "results/epistemic_ei_highdim/ei"]

    results = parse_highdim_telemetry(dirs=input_dirs)
    if not results:
        print("No telemetry results found in input directories.")
        return ""

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    os.makedirs(output_tables_dir, exist_ok=True)

    report_lines = []
    report_lines.append("# High-Dimensional Benchmark Optimization Summary\n")
    report_lines.append("Averaged across 5 seeds per benchmark task. Comparing standard `smac3bo` baseline against DyRF Epistemic UQ approaches.\n")

    for task_name in sorted(results.keys()):
        app_dict = results[task_name]
        table_md = format_benchmark_table(task_name, app_dict)
        report_lines.append(table_md)

        # Write CSV for this task
        csv_path = os.path.join(output_tables_dir, f"{task_name}_comparison.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Approach,Mean_Final_Cost,Std_Dev,Std_Error,Finished_Seeds,Best_Cost,Worst_Cost\n")
            stats_list = []
            for app_name, seed_map in app_dict.items():
                costs = list(seed_map.values())
                if not costs:
                    continue
                mean_c = float(np.mean(costs))
                std_c = float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0
                se_c = std_c / np.sqrt(len(costs)) if len(costs) > 1 else 0.0
                best_c = float(np.min(costs))
                worst_c = float(np.max(costs))
                n_seeds = len(costs)
                stats_list.append((app_name, mean_c, std_c, se_c, f"{n_seeds}/5", best_c, worst_c))
            stats_list.sort(key=lambda x: x[1])

            for app_name, mean_c, std_c, se_c, n_seeds, best_c, worst_c in stats_list:
                f.write(f"{app_name},{mean_c:.6f},{std_c:.6f},{se_c:.6f},{n_seeds},{best_c:.6f},{worst_c:.6f}\n")

    full_report = "\n".join(report_lines)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"Generated High-Dim summary report with {len(results)} tasks at {output_report}")
    return full_report

def main():
    parser = argparse.ArgumentParser(description="Generate High-Dim Benchmark Comparison Tables")
    parser.add_argument("--output_report", type=str, default="results/epistemic_ei_highdim/highdim_benchmark_tables.md")
    parser.add_argument("--output_tables_dir", type=str, default="results/epistemic_ei_highdim/tables")
    args = parser.parse_args()

    generate_all_highdim_reports(output_report=args.output_report, output_tables_dir=args.output_tables_dir)

if __name__ == "__main__":
    main()
