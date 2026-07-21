#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
from scipy.stats import rankdata

def parse_logs():
    log_files = glob.glob("results/array_7547051_*.log")
    results = {}
    
    args_pattern = re.compile(r"Running arguments:\s+(.*)")
    
    for file_path in log_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        args_match = args_pattern.search(content)
        if not args_match:
            continue
        args = args_match.group(1)
        
        task_match = re.search(r"ml=(cfg_ml_[a-z0-9_]+)", args)
        if not task_match:
            continue
        task = task_match.group(1)
        
        seed_match = re.search(r"seed=(\d+)", args)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        
        if "smac20" in args:
            approach = "smac3_bo"
        else:
            app_match = re.search(r"optimizer\.extractor_name=(\w+)", args)
            if not app_match:
                continue
            approach = app_match.group(1)
            
        solution_match = re.search(r"Solution found:\s*\(.*TrialValue\(\s*cost=([0-9\.\-+e]+)", content, re.DOTALL)
        if not solution_match:
            solution_match = re.search(r"TrialValue\(\s*cost=([0-9\.\-+e]+)", content, re.DOTALL)
            
        if solution_match:
            cost = float(solution_match.group(1))
        else:
            costs = re.findall(r"cost:\s*([0-9\.\-+e]+)", content)
            if costs:
                cost = min(float(c) for c in costs)
            else:
                continue
                
        results.setdefault(task, {}).setdefault(approach, {})[seed] = cost
        
    return results

def main():
    results = parse_logs()
    
    tasks = sorted(results.keys())
    approaches = sorted(list(next(iter(results.values())).keys()))
    
    # We will compute the mean cost for each approach on each task
    task_approach_means = {}
    for task in tasks:
        task_approach_means[task] = {}
        for app in approaches:
            # Aggregate costs across seeds
            seed_costs = results[task].get(app, {})
            if seed_costs:
                mean_cost = np.mean(list(seed_costs.values()))
            else:
                mean_cost = np.nan
            task_approach_means[task][app] = mean_cost
            
    # Calculate ranks per task (lower cost is better, so lower rank is better)
    # We will use fractional ranking (ties get the average of their ranks)
    task_ranks = {}
    for task in tasks:
        apps_present = []
        means = []
        for app in approaches:
            val = task_approach_means[task][app]
            if not np.isnan(val):
                apps_present.append(app)
                means.append(val)
        
        # Rank the means (1 is best/lowest)
        ranks = rankdata(means)
        task_ranks[task] = {app: r for app, r in zip(apps_present, ranks)}
        
    # Calculate average rank for each approach across all tasks
    app_ranks = {}
    for app in approaches:
        ranks_list = []
        for task in tasks:
            if app in task_ranks[task]:
                ranks_list.append(task_ranks[task][app])
        app_ranks[app] = ranks_list
        
    # Print per-task ranks
    print("# CARP-S HPOBench Rank Analysis\n")
    print("## Per-Task Ranks (Fractional Ranking - lower is better)")
    headers = ["Approach"] + tasks + ["Average Rank"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    
    overall_ranks = []
    for app in approaches:
        row = [app]
        for task in tasks:
            rank_val = task_ranks[task].get(app, np.nan)
            row.append(f"{rank_val:.2f}" if not np.isnan(rank_val) else "N/A")
        
        avg_rank = np.mean(app_ranks[app]) if app_ranks[app] else np.nan
        row.append(f"**{avg_rank:.3f}**" if not np.isnan(avg_rank) else "N/A")
        overall_ranks.append((app, avg_rank))
        print("| " + " | ".join(row) + " |")
        
    print("\n## Overall Leaderboard (by Average Rank)")
    print("| Rank | Approach | Average Rank |")
    print("| --- | --- | --- |")
    overall_ranks.sort(key=lambda x: x[1])
    for idx, (app, avg) in enumerate(overall_ranks):
        print(f"| {idx+1} | **{app}** | {avg:.3f} |")

if __name__ == "__main__":
    main()
