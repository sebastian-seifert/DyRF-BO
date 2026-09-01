#!/usr/bin/env python3
import os
import glob
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

def extract_task_and_approach_and_seed(
    filepath: str,
) -> Tuple[Optional[str], Optional[str], Optional[int], List[float]]:
    """Extracts benchmark task name, approach identifier, seed, and trial cost history from a result JSON.

    Args:
        filepath: Absolute or relative path to the trial log JSON file.

    Returns:
        Tuple[Optional[str], Optional[str], Optional[int], List[float]]:
            - Benchmark task name identifier.
            - Approach / uncertainty extractor name.
            - Random seed integer.
            - List of sequential evaluation costs.
    """
    filename = os.path.basename(filepath)
    with open(filepath, "r") as f:
        try:
            data = json.load(f)
        except Exception as e:
            return None, None, None, []
            
    if "baseline" in filepath:
        approach = "smac3_bo"
    else:
        approach = data.get("extractor_name", "unknown")
        
    if "seed" in data and isinstance(data["seed"], int):
        seed = data["seed"]
    else:
        m_seed = re.search(r"seed(\d+)\.json$", filename)
        seed = int(m_seed.group(1)) if m_seed else 1
        
    # Extract task name
    m_task = re.search(r"_(cfg_[^\s_]+(?:_[^\s_]+)*)_seed\d+\.json$", filename)
    if not m_task:
        m_task = re.search(r"_(nb301_[^\s_]+)_seed\d+\.json$", filename)
        
    if m_task:
        task_name = m_task.group(1)
    else:
        task_name = data.get("task_name", "unknown_task").replace("/", "_")
        
    trials = data.get("trials", [])
    costs = []
    for t in trials:
        if "cost" in t:
            costs.append(t["cost"])
        elif "trial_value" in t and "cost" in t["trial_value"]:
            costs.append(t["trial_value"]["cost"])
            
    return task_name, approach, seed, costs

def make_step_fill_vectors(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates right-continuous ('where=post') coordinate vectors for fill_between.

    Args:
        x: Array of x-coordinates (trial iterations).
        y: Array of y-coordinates (metric/cost values).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Interleaved x coordinates.
            - Interleaved y coordinates.
    """
    x_fill = np.empty(2 * len(x) - 1)
    x_fill[0::2] = x
    x_fill[1::2] = x[1:]
    
    y_fill = np.empty(2 * len(y) - 1)
    y_fill[0::2] = y
    y_fill[1::2] = y[:-1]
    
    return x_fill, y_fill

def generate_anytime_plot_for_task(
    task_name: str,
    approaches_data: Dict[str, Dict[int, List[float]]],
    output_dir: str,
    max_trials: int = 50,
) -> Tuple[str, str, Dict[str, Dict[str, Any]]]:
    """Renders and saves anytime performance curves (PNG and PDF) for an individual benchmark task.

    Args:
        task_name: Name identifier of the benchmark problem.
        approaches_data: Nested mapping of approach -> seed -> cost history.
        output_dir: Output directory where figure files will be saved.
        max_trials: Maximum number of BO trial evaluations to evaluate.

    Returns:
        Tuple[str, str, Dict[str, Dict[str, Any]]]:
            - Filepath to the saved PNG image.
            - Filepath to the saved PDF vector graphic.
            - Trajectories dictionary containing mean, standard error, and seed count per approach.
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    colors = {
        "smac3_bo": "#1f77b4",                # Blue (Baseline)
        "standard_proximity": "#d62728",      # Red
        "proximity_auto_lambda": "#2ca02c",  # Green
        "proximity_b": "#ff7f0e",            # Orange
        "proximity_bc": "#9467bd",           # Purple
        "standard_disagreement": "#8c564b",  # Brown
        "chen_variance": "#e377c2",          # Pink
        "likelihood_credal": "#7f7f7f",      # Gray
        "shaker_entropy": "#bcbd22",         # Yellow-Green
    }
    
    labels = {
        "smac3_bo": "SMAC3 Baseline (HPOFacade)",
        "standard_proximity": "Standard Proximity (DyRF)",
        "proximity_auto_lambda": "Proximity Auto-Lambda (DyRF)",
        "proximity_b": "Proximity Method B (DyRF)",
        "proximity_bc": "Proximity Method B+C (DyRF)",
        "standard_disagreement": "Standard Disagreement (DyRF)",
        "chen_variance": "Chen Variance (DyRF)",
        "likelihood_credal": "Likelihood Credal (DyRF)",
        "shaker_entropy": "Shaker Entropy (DyRF)"
    }
    
    trajectories = {}
    for approach, seed_dict in approaches_data.items():
        seed_anytimes = []
        for seed, costs in seed_dict.items():
            costs_arr = np.array(costs[:max_trials])
            anytime_best = np.minimum.accumulate(costs_arr)
            if len(anytime_best) < max_trials:
                anytime_best = np.pad(anytime_best, (0, max_trials - len(anytime_best)), mode='edge')
            seed_anytimes.append(anytime_best)
            
        if seed_anytimes:
            matrix = np.array(seed_anytimes)
            mean = np.mean(matrix, axis=0)
            n_seeds = matrix.shape[0]
            stderr = np.std(matrix, axis=0, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(mean)
            trajectories[approach] = {
                "mean": mean,
                "stderr": stderr,
                "n_seeds": n_seeds
            }
            
    # Sort approaches by final mean best cost
    sorted_approaches = sorted(trajectories.keys(), key=lambda a: trajectories[a]["mean"][-1])
    x = np.arange(1, max_trials + 1)
    
    for app in sorted_approaches:
        data = trajectories[app]
        mean = data["mean"]
        stderr = data["stderr"]
        n_seeds = data["n_seeds"]
        
        color = colors.get(app, "#333333")
        label = labels.get(app, app)
        linewidth = 2.2 if app in ["smac3_bo", "standard_proximity", "proximity_auto_lambda"] else 1.5
        linestyle = "--" if app == "smac3_bo" else "-"
        
        # Plot right-continuous step mean line
        plt.step(x, mean, where='post', label=f"{label} ({n_seeds} seed{'s' if n_seeds > 1 else ''})",
                 color=color, linewidth=linewidth, linestyle=linestyle, zorder=4 if app=="standard_proximity" else 3)
        
        # Plot shaded band ONLY if n_seeds > 1 (e.g. not 1/5 finished runs)
        if n_seeds > 1:
            x_step, lower_step = make_step_fill_vectors(x, mean - stderr)
            _, upper_step = make_step_fill_vectors(x, mean + stderr)
            plt.fill_between(x_step, lower_step, upper_step, color=color, alpha=0.15, zorder=2)
            
    plt.title(f"Anytime Performance on {task_name}", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("BO Evaluation Trials (t)", fontsize=12)
    plt.ylabel("Anytime Best Cost (y_best, lower is better)", fontsize=12)
    plt.xlim(1, max_trials)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, f"anytime_{task_name}.png")
    pdf_path = os.path.join(output_dir, f"anytime_{task_name}.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    return png_path, pdf_path, trajectories

def main() -> None:
    """CLI entry point to scan results directory and generate anytime performance plots."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate Anytime Performance Plots")
    parser.add_argument("--base_dir", type=str, default="results/epistemic_ei_highdim_fix")
    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = os.path.join(base_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    print(f"Scanning {len(json_files)} telemetry JSON files in {base_dir}...")
    
    # Task -> Approach -> Seed -> Costs
    task_data = {}
    
    for filepath in json_files:
        task_name, approach, seed, costs = extract_task_and_approach_and_seed(filepath)
        if task_name and approach and costs:
            task_data.setdefault(task_name, {}).setdefault(approach, {})[seed] = costs
            
    print(f"Found {len(task_data)} unique benchmark tasks:")
    summary_lines = ["# Anytime Performance Plots Summary\n"]
    
    for task_name in sorted(task_data.keys()):
        approaches_data = task_data[task_name]
        png_path, pdf_path, trajectories = generate_anytime_plot_for_task(task_name, approaches_data, output_dir)
        print(f"  [✓] Generated plots for {task_name} ({len(approaches_data)} approaches) -> {os.path.basename(png_path)}")
        
        summary_lines.append(f"## Benchmark: `{task_name}`")
        summary_lines.append(f"Saved plots: [`anytime_{task_name}.png`](file://{os.path.abspath(png_path)}) | [`anytime_{task_name}.pdf`](file://{os.path.abspath(pdf_path)})\n")
        summary_lines.append(f"| Approach | Mean Best Cost (t=50) | Std Error | Seeds | Band Plotted |")
        summary_lines.append(f"| :--- | :---: | :---: | :---: | :---: |")
        for app in sorted(trajectories.keys(), key=lambda a: trajectories[a]["mean"][-1]):
            m = trajectories[app]["mean"][-1]
            se = trajectories[app]["stderr"][-1]
            ns = trajectories[app]["n_seeds"]
            band_str = "Yes" if ns > 1 else "No (1 seed)"
            summary_lines.append(f"| `{app}` | {m:.6f} | {se:.6f} | {ns} | {band_str} |")
        summary_lines.append("\n---\n")
        
    summary_md_path = os.path.join(output_dir, "anytime_plots_summary.md")
    with open(summary_md_path, "w") as f:
        f.write("\n".join(summary_lines))
        
    print(f"\nAll plots generated successfully in {output_dir}/!")
    print(f"Summary written to {summary_md_path}")

if __name__ == "__main__":
    main()
