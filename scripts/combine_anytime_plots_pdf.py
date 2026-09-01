#!/usr/bin/env python3
import os
import glob
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
        except Exception:
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
    """Constructs step-aligned coordinates for fill_between under right-continuous ('where=post') step plotting.

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

def compute_anytime_trajectories(
    approaches_data: Dict[str, Dict[int, List[float]]],
    max_trials: int = 50,
) -> Dict[str, Dict[str, Any]]:
    """Computes anytime best cost trajectories (mean, standard error, sample counts) per approach.

    Args:
        approaches_data: Nested mapping of approach -> seed -> cost history.
        max_trials: Maximum number of BO trial evaluations to truncate or pad to.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of approach -> dictionary containing 'mean', 'stderr', and 'n_seeds'.
    """
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
    return trajectories

def main() -> None:
    """CLI entry point to combine benchmark anytime performance curves into multi-page PDF documents."""
    import argparse
    parser = argparse.ArgumentParser(description="Combine Anytime Performance Plots into PDF")
    parser.add_argument("--base_dir", type=str, default="results/epistemic_ei_highdim_fix")
    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = os.path.join(base_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    
    task_data = {}
    for filepath in json_files:
        task_name, approach, seed, costs = extract_task_and_approach_and_seed(filepath)
        if task_name and approach and costs:
            task_data.setdefault(task_name, {}).setdefault(approach, {})[seed] = costs
            
    sorted_tasks = sorted(task_data.keys())
    print(f"Combining {len(sorted_tasks)} benchmark tasks into PDF layout...")
    
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
        "smac3_bo": "SMAC3 Baseline",
        "standard_proximity": "Standard Proximity",
        "proximity_auto_lambda": "Proximity Auto-Lambda",
        "proximity_b": "Proximity Method B",
        "proximity_bc": "Proximity Method B+C",
        "standard_disagreement": "Standard Disagreement",
        "chen_variance": "Chen Variance",
        "likelihood_credal": "Likelihood Credal",
        "shaker_entropy": "Shaker Entropy"
    }
    
    max_trials = 50
    x = np.arange(1, max_trials + 1)
    
    # ----------------------------------------------------
    # PDF 1: 8 rows x 2 columns per page (Multi-page PDF)
    # ----------------------------------------------------
    pdf_8x2_path = os.path.join(output_dir, "combined_anytime_plots_8x2.pdf")
    plots_per_page = 16  # 8 rows x 2 columns
    
    with PdfPages(pdf_8x2_path) as pdf:
        num_pages = int(np.ceil(len(sorted_tasks) / plots_per_page))
        for page_idx in range(num_pages):
            page_tasks = sorted_tasks[page_idx * plots_per_page : (page_idx + 1) * plots_per_page]
            fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(12, 22), dpi=200)
            axes_flat = axes.flatten()
            
            for idx, task_name in enumerate(page_tasks):
                ax = axes_flat[idx]
                trajectories = compute_anytime_trajectories(task_data[task_name], max_trials=max_trials)
                sorted_apps = sorted(trajectories.keys(), key=lambda a: trajectories[a]["mean"][-1])
                
                for app in sorted_apps:
                    data = trajectories[app]
                    mean = data["mean"]
                    stderr = data["stderr"]
                    n_seeds = data["n_seeds"]
                    
                    color = colors.get(app, "#333333")
                    lbl = labels.get(app, app)
                    lw = 1.8 if app in ["smac3_bo", "standard_proximity", "proximity_auto_lambda"] else 1.2
                    ls = "--" if app == "smac3_bo" else "-"
                    
                    ax.step(x, mean, where='post', label=lbl, color=color, linewidth=lw, linestyle=ls)
                    
                    if n_seeds > 1:
                        x_step, lower_step = make_step_fill_vectors(x, mean - stderr)
                        _, upper_step = make_step_fill_vectors(x, mean + stderr)
                        ax.fill_between(x_step, lower_step, upper_step, color=color, alpha=0.12)
                        
                clean_title = task_name.replace("cfg_", "").replace("rbv2_super_", "rbv2_")
                ax.set_title(clean_title, fontsize=9, fontweight='bold', pad=3)
                ax.tick_params(axis='both', which='major', labelsize=7)
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.set_xlim(1, max_trials)
                
            # Hide unused axes
            for idx in range(len(page_tasks), len(axes_flat)):
                axes_flat[idx].set_visible(False)
                
            # Add shared legend on top of page
            handles, legend_labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, fontsize=8, frameon=True)
            
            fig.suptitle("High-Dimensional Benchmarks (>20D) Anytime Performance (EI Acquisition)", fontsize=13, fontweight='bold', y=1.01)
            plt.subplots_adjust(top=0.95, bottom=0.03, hspace=0.45, wspace=0.25)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
    print(f"[✓] Created 8x2 layout PDF: {pdf_8x2_path}")
    
    # ----------------------------------------------------
    # PDF 2: 9 rows x 2 columns (Single Page PDF for all 18 tasks)
    # ----------------------------------------------------
    pdf_9x2_path = os.path.join(output_dir, "combined_anytime_plots_9x2.pdf")
    fig, axes = plt.subplots(nrows=9, ncols=2, figsize=(12, 25), dpi=200)
    axes_flat = axes.flatten()
    
    for idx, task_name in enumerate(sorted_tasks):
        ax = axes_flat[idx]
        trajectories = compute_anytime_trajectories(task_data[task_name], max_trials=max_trials)
        sorted_apps = sorted(trajectories.keys(), key=lambda a: trajectories[a]["mean"][-1])
        
        for app in sorted_apps:
            data = trajectories[app]
            mean = data["mean"]
            stderr = data["stderr"]
            n_seeds = data["n_seeds"]
            
            color = colors.get(app, "#333333")
            lbl = labels.get(app, app)
            lw = 1.8 if app in ["smac3_bo", "standard_proximity", "proximity_auto_lambda"] else 1.2
            ls = "--" if app == "smac3_bo" else "-"
            
            ax.step(x, mean, where='post', label=lbl, color=color, linewidth=lw, linestyle=ls)
            
            if n_seeds > 1:
                x_step, lower_step = make_step_fill_vectors(x, mean - stderr)
                _, upper_step = make_step_fill_vectors(x, mean + stderr)
                ax.fill_between(x_step, lower_step, upper_step, color=color, alpha=0.12)
                
        clean_title = task_name.replace("cfg_", "").replace("rbv2_super_", "rbv2_")
        ax.set_title(clean_title, fontsize=9, fontweight='bold', pad=3)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(1, max_trials)
        
    for idx in range(len(sorted_tasks), len(axes_flat)):
        axes_flat[idx].set_visible(False)
        
    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, fontsize=8, frameon=True)
    fig.suptitle("High-Dimensional Benchmarks (>20D) Anytime Performance Overview (18 Tasks)", fontsize=13, fontweight='bold', y=1.01)
    
    plt.subplots_adjust(top=0.96, bottom=0.02, hspace=0.45, wspace=0.25)
    plt.savefig(pdf_9x2_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[✓] Created 9x2 layout PDF: {pdf_9x2_path}")

if __name__ == "__main__":
    main()
