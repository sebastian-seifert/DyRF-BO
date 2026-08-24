#!/usr/bin/env python3
"""Parses telemetry files from CARP-S BBsubset Additive Hybrid Sweep and generates comparison tables."""

import os
import glob
import json
import numpy as np
import pandas as pd
from collections import defaultdict

def parse_additive_hybrid_results(base_dir: str = "results/bbsubset_additive_hybrid"):
    telemetry_files = glob.glob(os.path.join(base_dir, "**", "telemetry_*.json"), recursive=True)
    if not telemetry_files:
        print(f"No telemetry files found in {base_dir}")
        return

    # Data structure: data[acq][task][approach][seed] = best_cost
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for fpath in telemetry_files:
        try:
            with open(fpath, "r") as f:
                content = json.load(f)
            
            trials = content.get("trials", [])
            if not trials:
                continue
            
            costs = [t["cost"] for t in trials if "cost" in t and np.isfinite(t["cost"])]
            if not costs:
                continue
            best_cost = min(costs)
            
            # Parse metadata from filename
            # Format: telemetry_additive_{acq}_{approach}_{task_name}_seed{seed}.json
            # Or baseline: telemetry_smac3_{acq}_{task_name}_seed{seed}.json
            fname = os.path.basename(fpath)
            
            if fname.startswith("telemetry_smac3_"):
                parts = fname.replace("telemetry_smac3_", "").replace(".json", "").split("_")
                acq = parts[0]
                seed = int(parts[-1].replace("seed", ""))
                task = "_".join(parts[1:-1])
                approach = "smac3_baseline"
            elif fname.startswith("telemetry_additive_"):
                parts = fname.replace("telemetry_additive_", "").replace(".json", "").split("_")
                acq = parts[0]
                seed = int(parts[-1].replace("seed", ""))
                
                # Check known approaches
                known_approaches = [
                    "standard_disagreement", "shaker_entropy", "likelihood_credal",
                    "standard_proximity", "proximity_b", "proximity_bc", "proximity_auto_lambda"
                ]
                approach = None
                for app in known_approaches:
                    if fname.startswith(f"telemetry_additive_{acq}_{app}_"):
                        approach = app
                        task = fname.replace(f"telemetry_additive_{acq}_{app}_", "").replace(f"_seed{seed}.json", "")
                        break
                if approach is None:
                    continue
            else:
                continue

            data[acq.upper()][task][approach][seed] = best_cost
        except Exception as e:
            print(f"Warning: Failed to parse {fpath} ({e})")

    # Generate Markdown Summary Report
    report_lines = ["# CARP-S BBsubset Additive Hybrid Sweep - Summary Report\n"]
    
    for acq, tasks in sorted(data.items()):
        report_lines.append(f"## Acquisition Function: {acq}\n")
        
        for task_name, approaches in sorted(tasks.items()):
            report_lines.append(f"### Benchmark Task: `{task_name}`\n")
            report_lines.append("| Approach | Mean Best Cost | SEM | Min Cost | Seeds |")
            report_lines.append("| :--- | :---: | :---: | :---: | :---: |")
            
            summary = []
            for approach, seed_dict in approaches.items():
                costs = list(seed_dict.values())
                mean_cost = np.mean(costs)
                sem = np.std(costs, ddof=1) / np.sqrt(len(costs)) if len(costs) > 1 else 0.0
                min_c = np.min(costs)
                summary.append((approach, mean_cost, sem, min_c, len(costs)))
            
            # Sort by mean cost ascending (minimization)
            summary.sort(key=lambda x: x[1])
            for app, mean_c, s, min_c, n_seeds in summary:
                report_lines.append(f"| `{app}` | {mean_c:.6f} | {s:.6f} | {min_c:.6f} | {n_seeds} |")
            report_lines.append("")

    report_content = "\n".join(report_lines)
    out_report_path = os.path.join(base_dir, "additive_hybrid_summary_report.md")
    with open(out_report_path, "w") as f:
        f.write(report_content)
    print(f"Report saved to {out_report_path}")

if __name__ == "__main__":
    parse_additive_hybrid_results()
