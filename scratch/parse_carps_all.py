#!/usr/bin/env python3
import os
import re
import glob
import numpy as np

def parse_logs():
    log_files = glob.glob("results/array_7547051_*.log")
    print(f"Parsing {len(log_files)} log files...")
    
    results = {}
    
    # regexes
    args_pattern = re.compile(r"Running arguments:\s+(.*)")
    solution_pattern = re.compile(r"TrialValue\(\s*cost=(?:np\.float64\()?([0-9\.e\-\+]+)\)?", re.DOTALL)
    
    for file_path in log_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        args_match = args_pattern.search(content)
        if not args_match:
            continue
        args = args_match.group(1)
        
        # Determine task
        task_match = re.search(r"ml=(cfg_ml_[a-z0-9_]+)", args)
        if not task_match:
            continue
        task = task_match.group(1)
        
        # Determine seed
        seed_match = re.search(r"seed=(\d+)", args)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        
        # Determine approach
        if "smac20" in args:
            approach = "smac3_bo"
        else:
            app_match = re.search(r"optimizer\.extractor_name=(\w+)", args)
            if not app_match:
                continue
            approach = app_match.group(1)
            
        # Determine cost
        solution_match = re.search(r"Solution found:\s*\(.*TrialValue\(\s*cost=([0-9\.\-+e]+)", content, re.DOTALL)
        if not solution_match:
            # try general search for TrialValue cost
            solution_match = re.search(r"TrialValue\(\s*cost=([0-9\.\-+e]+)", content, re.DOTALL)
            
        if solution_match:
            cost = float(solution_match.group(1))
        else:
            # Fall back to finding minimum cost in telemetry if we can't find it in log,
            # or try to extract from "cost: X" patterns printed during runs
            costs = re.findall(r"cost:\s*([0-9\.\-+e]+)", content)
            if costs:
                cost = min(float(c) for c in costs)
            else:
                print(f"Warning: Could not find cost in {file_path}")
                continue
                
        results.setdefault(task, {}).setdefault(approach, {})[seed] = cost
        
    return results

def main():
    results = parse_logs()
    
    for task in sorted(results.keys()):
        print(f"\n## Task: {task}")
        print("| Approach | Mean Best Cost | Std Dev | Seeds | Best Cost |")
        print("| --- | --- | --- | --- | --- |")
        
        approaches = results[task]
        rows = []
        for app, seed_costs in approaches.items():
            costs = list(seed_costs.values())
            mean = np.mean(costs)
            std = np.std(costs)
            best = np.min(costs)
            n_seeds = len(costs)
            rows.append((app, mean, std, n_seeds, best))
            
        # Sort by mean cost ascending (lower is better)
        rows.sort(key=lambda x: x[1])
        for app, mean, std, n_seeds, best in rows:
            print(f"| {app} | {mean:.5f} | {std:.5f} | {n_seeds}/5 | {best:.5f} |")

if __name__ == "__main__":
    main()
