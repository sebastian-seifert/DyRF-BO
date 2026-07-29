#!/usr/bin/env python3
"""Generates EXPERIMENT_MANIFEST.json recording exact scientific metadata and protocol."""

import os
import sys
import json
import subprocess
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_registry import BenchmarkRegistry
from ep_extractors import UQExtractorRegistry

def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"

def generate_highdim_manifest(output_path: str = "results/epistemic_ei_highdim/EXPERIMENT_MANIFEST.json") -> dict:
    git_commit = get_git_commit()
    tasks = BenchmarkRegistry.HIGH_DIM_NAS
    extractors = UQExtractorRegistry.list_registered()

    approaches = ["smac3_bo"] + extractors

    from scripts.inspect_approach_hyperparameters import get_all_approach_hyperparameters
    hp_specs = get_all_approach_hyperparameters()

    manifest = {
        "experiment_name": "highdim_ei_epistemic_sweep",
        "date_created": datetime.now().isoformat(),
        "git_commit": git_commit,
        "git_branch": "feat/epistemic-ei-acq",
        "n_seeds": 5,
        "seeds": [1, 2, 3, 4, 5],
        "n_trials": 50,
        "n_init": 10,
        "acquisition_function": "Expected Improvement (EI)",
        "acq_xi": 0.0,
        "surrogate_model": "RandomForestRegressor (100 trees)",
        "benchmark_suite": "High-Dimensional (>20D) & NAS",
        "benchmark_tasks": tasks,
        "n_tasks": len(tasks),
        "approaches": approaches,
        "n_approaches": len(approaches),
        "approach_hyperparameters": hp_specs,
        "total_planned_runs": len(tasks) * len(approaches) * 5,
        "paths": {
            "baseline_telemetry": "results/epistemic_ei_highdim/baseline/",
            "ei_telemetry": "results/epistemic_ei_highdim/ei/",
            "markdown_report": "results/epistemic_ei_highdim/highdim_benchmark_tables.md",
            "csv_tables": "results/epistemic_ei_highdim/tables/",
            "overall_ranks_csv": "results/epistemic_ei_highdim/overall_average_ranks.csv"
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated experiment manifest at {output_path}")
    return manifest

if __name__ == "__main__":
    generate_highdim_manifest()
