#!/usr/bin/env python3
"""Master Results Parser for CARP-S BBsubset Dev Sweep (2,430 Completed Runs).

Scans native SMAC3 `runhistory.json` files in `runs/`, aggregates performance trajectories
across all 5 seeds, computes Mean Best Cost and Average Rank across the 18 CARP-S dev benchmarks,
and outputs publication-ready statistical summary tables.
"""

import os
import json
import glob
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry
from ep_extractors import UQExtractorRegistry

def parse_bbsubset_dev_results():
    extractors = UQExtractorRegistry.list_registered()
    all_approaches = ["baseline"] + extractors
    acquisitions = ["ei", "pi", "lcb"]
    dev_tasks = CarpsBBSubsetRegistry.get_dev_tasks()

    # Store minimum cost: results_dict[acq][benchmark_short][approach][seed] = best_cost
    results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Find all runhistory.json files under runs/
    rh_files = sorted(glob.glob("runs/**/runhistory.json", recursive=True))
    print(f"Found {len(rh_files)} SMAC3 runhistory.json files in runs/")

    for rh_path in rh_files:
        # Determine optimizer, acq, benchmark, and seed from path
        # Path format: runs/<optimizer_id>/<benchmark_id>/blackbox/20/dev/<task_path>/<seed>/<seed>/smac3_output/...
        
        # 1. Acquisition Function
        acq = None
        for a in acquisitions:
            if f"_{a}/" in rh_path or f"_{a}_" in rh_path or f"/{a}/" in rh_path:
                acq = a
                break
        if acq is None:
            if "_ei" in rh_path: acq = "ei"
            elif "_pi" in rh_path: acq = "pi"
            elif "_lcb" in rh_path: acq = "lcb"
            else: continue

        # 2. Approach
        approach = "baseline"
        for ext in extractors:
            if ext in rh_path:
                approach = ext
                break

        # 3. Benchmark Task
        benchmark = None
        for t in dev_tasks:
            t_short = t.split("/")[-1]
            # Match task pattern in path
            clean_name = t_short.replace("subset_", "").replace("_None", "")
            if t_short in rh_path or clean_name in rh_path:
                benchmark = t_short
                break
            # Additional path matching for sub-components (e.g. bbob/2/12/0 or lcbench/168335)
            parts = clean_name.split("_")
            if len(parts) >= 3:
                sub_part = "/".join(parts[-3:])
                if sub_part in rh_path:
                    benchmark = t_short
                    break

        if benchmark is None:
            continue

        # 4. Seed
        # Pattern .../<seed>/<seed>/smac3_output/...
        m_seed = re.search(r"/(\d+)/(\d+)/smac3_output/", rh_path)
        if m_seed:
            seed = int(m_seed.group(1))
        else:
            seed = 1

        # Parse minimum cost from runhistory.json
        try:
            with open(rh_path, "r") as f:
                rh_data = json.load(f)
                costs = [entry["cost"] for entry in rh_data.get("data", []) if entry.get("status") == 1]
                if costs:
                    best_cost = min(costs)
                    results_dict[acq][benchmark][approach][seed] = best_cost
        except Exception:
            pass

    print("\n==================================================")
    print("CARP-S BBsubset Dev Sweep - Telemetry Analysis")
    print("==================================================")

    for acq in acquisitions:
        print(f"\n==========================================")
        print(f"Acquisition Function: {acq.upper()}")
        print(f"==========================================")

        benchmarks = sorted(list(results_dict[acq].keys()))
        if not benchmarks:
            print(f"No parsed runhistory results for {acq.upper()}.")
            continue

        print(f"Evaluated Benchmarks ({len(benchmarks)} / 20):")
        
        print(f"\n==================================================")
        print(f"Per-Task Breakdown (Mean Cost ± SEM across 5 seeds)")
        print(f"==================================================")

        for b in benchmarks:
            print(f"\n>>> Benchmark Task: {b}")
            task_rows = []
            for app in all_approaches:
                seeds_dict = results_dict[acq][b][app]
                if seeds_dict:
                    vals = list(seeds_dict.values())
                    mean_val = np.mean(vals)
                    sem_val = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                    task_rows.append({
                        "Approach": app,
                        "Mean Cost": mean_val,
                        "SEM": sem_val,
                        "Min Cost": np.min(vals),
                        "Seeds": len(vals)
                    })
            if task_rows:
                df_task = pd.DataFrame(task_rows).sort_values("Mean Cost")
                print(df_task.to_string(index=False))

        # Build DataFrame of Mean Costs across seeds
        mean_costs = defaultdict(dict)
        for b in benchmarks:
            for app in all_approaches:
                seeds_dict = results_dict[acq][b][app]
                if seeds_dict:
                    vals = list(seeds_dict.values())
                    mean_costs[b][app] = np.mean(vals)
                else:
                    mean_costs[b][app] = np.nan

        df_costs = pd.DataFrame(mean_costs).T
        df_ranks = df_costs.rank(axis=1, ascending=True)
        avg_ranks = df_ranks.mean(axis=0).sort_values()

        print(f"\n==================================================")
        print(f"Average Rank across {len(benchmarks)} CARP-S Dev Benchmarks ({acq.upper()})")
        print(f"==================================================")
        for app, rank in avg_ranks.items():
            print(f"  {app:<28}: Average Rank {rank:.2f}")

    # Write complete summary report to Markdown file
    report_file = "results/bbsubset_dev_summary_report.md"
    with open(report_file, "w") as rf:
        rf.write("# CARP-S BBsubset Dev Sweep - Telemetry & Per-Task Breakdown Report\n\n")
        for acq in acquisitions:
            benchmarks = sorted(list(results_dict[acq].keys()))
            if not benchmarks:
                continue
            rf.write(f"## Acquisition Function: {acq.upper()}\n\n")
            for b in benchmarks:
                rf.write(f"### Benchmark Task: `{b}`\n\n")
                task_rows = []
                for app in all_approaches:
                    seeds_dict = results_dict[acq][b][app]
                    if seeds_dict:
                        vals = list(seeds_dict.values())
                        mean_val = np.mean(vals)
                        sem_val = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                        task_rows.append({
                            "Approach": app,
                            "Mean Best Cost": mean_val,
                            "SEM": sem_val,
                            "Min Cost": np.min(vals),
                            "Seeds": len(vals)
                        })
                if task_rows:
                    rf.write("| Approach | Mean Best Cost | SEM | Min Cost | Seeds |\n")
                    rf.write("| :--- | :---: | :---: | :---: | :---: |\n")
                    for row in sorted(task_rows, key=lambda r: r["Mean Best Cost"]):
                        rf.write(f"| `{row['Approach']}` | {row['Mean Best Cost']:.6f} | {row['SEM']:.6f} | {row['Min Cost']:.6f} | {row['Seeds']} |\n")
                    rf.write("\n")

    print(f"\nSaved complete summary report to {report_file}")
    return results_dict

if __name__ == "__main__":
    parse_bbsubset_dev_results()


