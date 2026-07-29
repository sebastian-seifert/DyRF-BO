#!/usr/bin/env python3
import os
import re
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_telemetry_directory(results_dir="results"):
    """
    Parses all telemetry JSON files in results_dir and aggregates the best cost per task, approach, and seed.
    Returns:
        dict: {task_name: {approach_name: [list_of_best_costs_across_seeds]}}
    """
    results = {}
    if not os.path.exists(results_dir):
        return results

    filename_pattern = re.compile(
        r"^telemetry_(?P<approach>.+?)_(?P<task>cfg_.+?)_seed(?P<seed>\d+)\.json$"
    )

    for filename in os.listdir(results_dir):
        if not filename.endswith(".json") or not filename.startswith("telemetry_"):
            continue
        filepath = os.path.join(results_dir, filename)
        
        match = filename_pattern.match(filename)
        if match:
            meta = match.groupdict()
            task = meta["task"]
            approach = meta["approach"]
        else:
            try:
                with open(filepath, "r") as f:
                    content = json.load(f)
                task = content.get("task_name", "unknown").replace("/", "_")
                approach = content.get("extractor_name", "unknown")
            except Exception:
                continue

        try:
            with open(filepath, "r") as f:
                content = json.load(f)
            trials = content.get("trials", [])
            if not trials:
                continue
            best_cost = min(t["cost"] for t in trials)
            results.setdefault(task, {}).setdefault(approach, []).append(best_cost)
        except Exception as e:
            print(f"Error parsing {filename}: {e}")

    return results

def parse_array_logs(results_dir="results"):
    """
    Extracts the final cost value, task, approach, and seed out of each array_*.log file.
    Returns:
        dict: {task: {approach: {seed: final_cost}}}
    """
    final_costs = {}
    if not os.path.exists(results_dir):
        return final_costs

    log_files = glob.glob(os.path.join(results_dir, "array_*.log"))

    for file_path in log_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            continue

        arg_match = re.search(r"Running arguments:\s+(.*)", content)
        if not arg_match:
            continue
        args = arg_match.group(1)

        # Task extraction
        task_match = re.search(r"\+task/[^=\s]+=([^\s]+)", args)
        if not task_match:
            task_match = re.search(r"ml=([^\s]+)", args)
        if not task_match:
            task_match = re.search(r"\+task/YAHPO/SO=([^\s]+)", args)
        if not task_match:
            continue
        task = task_match.group(1)

        # Seed extraction
        seed_match = re.search(r"\bseed=(\d+)", args)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))

        # Approach extraction
        if "smac20" in args:
            approach = "smac3_bo"
        else:
            app_match = re.search(r"optimizer\.extractor_name=(\w+)", args)
            if app_match:
                approach = app_match.group(1)
            else:
                continue

        # Extract final cost
        cost = None
        sol_match = re.search(r"Solution found:.*?TrialValue\(\s*cost=([0-9\.e\-\+]+)", content, re.DOTALL)
        if sol_match:
            cost = float(sol_match.group(1))
        else:
            # Fallback to telemetry or trial cost logs
            costs = re.findall(r"cost:\s*([0-9\.\-+e]+)", content)
            if costs:
                cost = min(float(c) for c in costs)
            else:
                telem_match = re.search(r"optimizer\.telemetry_path=(\S+)", args)
                if telem_match and os.path.exists(telem_match.group(1)):
                    try:
                        with open(telem_match.group(1), "r") as tf:
                            tdata = json.load(tf)
                        trials = tdata.get("trials", [])
                        if trials:
                            cost = min(t["cost"] for t in trials)
                    except Exception:
                        pass

        if cost is not None:
            final_costs.setdefault(task, {}).setdefault(approach, {})[seed] = cost

    return final_costs

def parse_bo_histories(results_dir="results"):
    """
    Parses trial cost history for each benchmark task, approach, and seed.
    Uses telemetry JSON files when available, and array logs as fallback (e.g. for smac3_bo).
    Returns:
        dict: {task: {approach: {seed: [cost_0, cost_1, ...]}}}
    """
    bo_histories = {}
    if not os.path.exists(results_dir):
        return bo_histories

    # 1. Parse telemetry files
    telemetry_files = glob.glob(os.path.join(results_dir, "telemetry_*.json"))
    filename_pattern = re.compile(
        r"^telemetry_(?P<approach>.+?)_(?P<task>cfg_.+?)_seed(?P<seed>\d+)\.json$"
    )

    for file_path in telemetry_files:
        filename = os.path.basename(file_path)
        match = filename_pattern.match(filename)
        if not match:
            continue
        meta = match.groupdict()
        task = meta["task"]
        approach = meta["approach"]
        seed = int(meta["seed"])

        try:
            with open(file_path, "r") as f:
                tdata = json.load(f)
            trials = tdata.get("trials", [])
            if not trials:
                continue
            # Sort trials by trial_idx
            sorted_trials = sorted(trials, key=lambda x: x.get("trial_idx", 0))
            costs = [t["cost"] for t in sorted_trials]
            bo_histories.setdefault(task, {}).setdefault(approach, {})[seed] = costs
        except Exception as e:
            print(f"Error reading telemetry {filename}: {e}")

    # 2. Parse array logs for approaches missing telemetry (e.g. smac3_bo)
    log_files = glob.glob(os.path.join(results_dir, "array_*.log"))
    for file_path in log_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            continue

        arg_match = re.search(r"Running arguments:\s+(.*)", content)
        if not arg_match:
            continue
        args = arg_match.group(1)

        task_match = re.search(r"\+task/[^=\s]+=([^\s]+)", args)
        if not task_match:
            task_match = re.search(r"ml=([^\s]+)", args)
        if not task_match:
            task_match = re.search(r"\+task/YAHPO/SO=([^\s]+)", args)
        if not task_match:
            continue
        task = task_match.group(1)

        seed_match = re.search(r"\bseed=(\d+)", args)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))

        if "smac20" in args:
            approach = "smac3_bo"
        else:
            app_match = re.search(r"optimizer\.extractor_name=(\w+)", args)
            approach = app_match.group(1) if app_match else None

        if not approach:
            continue

        # Check if already present from telemetry
        if task in bo_histories and approach in bo_histories[task] and seed in bo_histories[task][approach]:
            continue

        # Extract costs sequence from log file
        costs = [float(c) for c in re.findall(r"cost:\s*([0-9\.\-+e]+)", content)]
        if costs:
            bo_histories.setdefault(task, {}).setdefault(approach, {})[seed] = costs

    return bo_histories

def compute_anytime_stats(seed_histories):
    """
    Computes the mean incumbent trajectory, standard error, and standard deviation across seeds.
    Args:
        seed_histories (dict): {seed: [cost_0, cost_1, ...]}
    Returns:
        tuple: (trial_indices, mean_incumbents, standard_errors, standard_deviations)
    """
    if not seed_histories:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Compute incumbent trajectory per seed
    incumbent_trajectories = []
    max_len = max(len(costs) for costs in seed_histories.values())

    for seed, costs in seed_histories.items():
        if not costs:
            continue
        incumbent = np.minimum.accumulate(costs)
        if len(incumbent) < max_len:
            # Pad with last known incumbent value
            incumbent = np.pad(incumbent, (0, max_len - len(incumbent)), mode='edge')
        incumbent_trajectories.append(incumbent)

    incumbent_matrix = np.array(incumbent_trajectories) # shape: (n_seeds, max_len)
    n_seeds = incumbent_matrix.shape[0]

    mean_traj = np.mean(incumbent_matrix, axis=0)
    if n_seeds > 1:
        std_traj = np.std(incumbent_matrix, axis=0, ddof=1)
        se_traj = std_traj / np.sqrt(n_seeds)
    else:
        std_traj = np.zeros_like(mean_traj)
        se_traj = np.zeros_like(mean_traj)

    trial_indices = np.arange(1, max_len + 1)
    return trial_indices, mean_traj, se_traj, std_traj

import csv

def parse_baseline_smac3(baseline_tables_dir="results/carps_summary_21072026/tables"):
    """
    Parses smac3_bo baseline statistics from prior summary tables.
    Returns:
        dict: {task_name: {"smac3_bo": {"mean": float, "std": float, "se": float, "n_seeds": str, "best": float, "worst": float}}}
    """
    smac_data = {}
    if not os.path.exists(baseline_tables_dir):
        return smac_data

    for filename in os.listdir(baseline_tables_dir):
        if not filename.endswith("_comparison.csv"):
            continue
        task_name = filename.replace("_comparison.csv", "")
        filepath = os.path.join(baseline_tables_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Approach") in ["smac3_bo", "smac3", "smac20"]:
                        smac_data.setdefault(task_name, {})["smac3_bo"] = {
                            "mean": float(row["Mean_Final_Cost"]),
                            "std": float(row["Std_Dev"]),
                            "se": float(row["Std_Error"]),
                            "n_seeds": row.get("Finished_Seeds", "5/5"),
                            "best": float(row["Best_Cost"]),
                            "worst": float(row["Worst_Cost"]),
                        }
        except Exception as e:
            print(f"Error reading baseline csv {filename}: {e}")

    return smac_data

def generate_benchmark_tables(final_costs, output_dir="results/carps_summary", baseline_smac_data=None):
    """
    Generates comparison tables per benchmark and a summary markdown report.
    Args:
        final_costs (dict): {task: {approach: {seed: cost}}}
        output_dir (str): directory to save tables and report
        baseline_smac_data (dict): optional pre-parsed smac3_bo baseline metrics per task
    Returns:
        str: full markdown report string
    """
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    report_lines = []
    report_lines.append("# CARP-S Optimization Benchmark Summary Report\n")
    report_lines.append("This report presents the final cost comparison across different BO approaches, standard `smac3_bo` baseline, and Dynamic RF UQ extractors for CARP-S benchmarks.\n")

    for task_name in sorted(final_costs.keys()):
        approaches = final_costs[task_name]
        report_lines.append(f"## Benchmark Task: `{task_name}`\n")
        report_lines.append("| Approach | Mean Final Cost | Std Dev | Std Error | Finished Seeds | Best Seed Cost | Worst Seed Cost |")
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")

        csv_lines = ["Approach,Mean_Final_Cost,Std_Dev,Std_Error,Finished_Seeds,Best_Cost,Worst_Cost"]

        approaches_perf = []
        for app_name, seed_dict in approaches.items():
            costs = list(seed_dict.values())
            if not costs:
                continue
            mean_c = float(np.mean(costs))
            std_c = float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0
            se_c = std_c / np.sqrt(len(costs)) if len(costs) > 1 else 0.0
            best_c = float(np.min(costs))
            worst_c = float(np.max(costs))
            n_seeds = len(costs)

            approaches_perf.append((app_name, mean_c, std_c, se_c, f"{n_seeds}/5", best_c, worst_c))

        # Inject smac3_bo baseline if not already present in final_costs
        if baseline_smac_data and task_name in baseline_smac_data and "smac3_bo" in baseline_smac_data[task_name]:
            if not any("smac3_bo" in app[0] for app in approaches_perf):
                s_info = baseline_smac_data[task_name]["smac3_bo"]
                approaches_perf.append((
                    "smac3_bo (Baseline)",
                    s_info["mean"],
                    s_info["std"],
                    s_info["se"],
                    s_info["n_seeds"],
                    s_info["best"],
                    s_info["worst"]
                ))

        # Sort by mean cost ascending (lower cost = better performance)
        approaches_perf.sort(key=lambda x: x[1])

        for app_name, mean_c, std_c, se_c, n_seeds, best_c, worst_c in approaches_perf:
            report_lines.append(
                f"| `{app_name}` | {mean_c:.6f} | {std_c:.6f} | {se_c:.6f} | {n_seeds} | {best_c:.6f} | {worst_c:.6f} |"
            )
            csv_lines.append(f"{app_name},{mean_c:.6f},{std_c:.6f},{se_c:.6f},{n_seeds},{best_c:.6f},{worst_c:.6f}")

        report_lines.append("\n")

        # Write CSV table for this benchmark
        csv_path = os.path.join(tables_dir, f"{task_name}_comparison.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(csv_lines))

    report_content = "\n".join(report_lines)
    summary_path = os.path.join(output_dir, "summary_report.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Benchmark tables and summary report written to {summary_path}")
    return report_content

def generate_anytime_plots(bo_histories, output_dir="results/carps_summary"):
    """
    Generates anytime performance plots for every benchmark task.
    Plots every approach into the same plot per benchmark.
    Uses right-continuous step functions (where='post' / step='post') for non-continuous progress.
    The main line is the mean trajectory and the shaded band is the standard deviation across seeds.
    Args:
        bo_histories (dict): {task: {approach: {seed: [cost_0, cost_1, ...]}}}
        output_dir (str): directory to save plots
    Returns:
        list: paths of generated plot files
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    created_plots = []

    # Distinct color palette for up to 8 approaches
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ]

    for task_name in sorted(bo_histories.keys()):
        approaches = bo_histories[task_name]
        if not approaches:
            continue

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        for i, app_name in enumerate(sorted(approaches.keys())):
            seed_histories = approaches[app_name]
            trial_indices, mean_traj, se_traj, std_traj = compute_anytime_stats(seed_histories)
            if len(trial_indices) == 0:
                continue

            color = colors[i % len(colors)]

            # Step plot (non-continuous from the left side -> right-continuous step function `where='post'`)
            ax.step(
                trial_indices,
                mean_traj,
                where="post",
                label=app_name,
                color=color,
                linewidth=2.0,
            )

            # Shaded standard deviation band (step='post')
            ax.fill_between(
                trial_indices,
                mean_traj - std_traj,
                mean_traj + std_traj,
                step="post",
                color=color,
                alpha=0.18,
            )

        ax.set_title(f"Anytime Performance: {task_name}", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("BO Trial Index / Function Evaluations", fontsize=12)
        ax.set_ylabel("Incumbent Cost (Lower is Better)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Approach", bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)

        fig.tight_layout()

        png_path = os.path.join(plots_dir, f"{task_name}_anytime.png")
        pdf_path = os.path.join(plots_dir, f"{task_name}_anytime.pdf")
        fig.savefig(png_path)
        fig.savefig(pdf_path)
        plt.close(fig)

        created_plots.append(png_path)

    print(f"Generated {len(created_plots)} anytime performance plots in {plots_dir}")
    return created_plots

import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze CARP-S benchmark results")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing array log files and telemetry JSONs")
    parser.add_argument("--output_dir", type=str, default="results/carps_summary", help="Directory to save summary tables, plots, and report")
    parser.add_argument("--baseline_dir", type=str, default="results/carps_summary_21072026/tables", help="Directory containing prior smac3_bo comparison tables")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir
    baseline_dir = args.baseline_dir

    print(f"Extracting final costs from array logs in {results_dir}...")
    final_costs = parse_array_logs(results_dir)

    print("Parsing BO run histories for anytime performance...")
    bo_histories = parse_bo_histories(results_dir)
    print(f"Parsed BO histories for {len(bo_histories)} benchmark tasks.")

    # Supplement final_costs from telemetry BO histories if array logs are absent or incomplete
    for task_name, app_dict in bo_histories.items():
        for app_name, seed_dict in app_dict.items():
            for seed_id, costs in seed_dict.items():
                if costs and seed_id not in final_costs.get(task_name, {}).get(app_name, {}):
                    final_costs.setdefault(task_name, {}).setdefault(app_name, {})[seed_id] = min(costs)

    print(f"Found final costs for {len(final_costs)} benchmark tasks.")

    print(f"Parsing baseline smac3_bo metrics from {baseline_dir}...")
    baseline_smac_data = parse_baseline_smac3(baseline_dir)
    print(f"Found baseline smac3_bo data for {len(baseline_smac_data)} benchmark tasks.")

    print(f"Generating benchmark comparison tables and summary report in {output_dir}...")
    generate_benchmark_tables(final_costs, output_dir, baseline_smac_data=baseline_smac_data)

    print("Creating anytime performance step plots...")
    generate_anytime_plots(bo_histories, output_dir)

    print("CARP-S analysis complete! Output saved in:", output_dir)

if __name__ == "__main__":
    main()

