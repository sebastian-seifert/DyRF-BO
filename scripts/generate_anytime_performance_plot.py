#!/usr/bin/env python3
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

def make_step_coords(x: np.ndarray, y: np.ndarray):
    """
    Transforms 1D arrays x and y into right-continuous ('rechtsstetig', step-post)
    coordinate arrays for plotting lines and step fill_between uncertainty bands.
    
    For x = [x0, x1, x2] and y = [y0, y1, y2]:
    x_step = [x0, x1, x1, x2, x2]
    y_step = [y0, y0, y1, y1, y2]
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) <= 1:
        return x, y

    x_step = np.repeat(x, 2)[1:]
    y_step = np.repeat(y, 2)[:-1]
    return x_step, y_step

def compute_anytime_trajectories(results_dir="results/epistemic_ei_highdim", max_steps=100, task_filter=None):
    """
    Parses telemetry JSON files under results_dir, extracts cumulative minimum cost
    per trial iteration, computes mean and SEM (std / sqrt(N)) trajectories per extractor.
    If task_filter is specified (e.g. '1111'), filters for tasks containing task_filter.
    """
    json_files = glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True)
    if not json_files:
        raise FileNotFoundError(f"No JSON telemetry files found in {results_dir}")

    # Group runs by (extractor_name, task_name, seed)
    runs = {} # (extractor, task, seed) -> 1D array of best cost at step t
    task_global_min = {}
    task_global_max = {}

    for fpath in json_files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            extractor = data.get("extractor_name")
            task = data.get("task_name")
            seed = data.get("seed")

            # Fallback for seed from filename (e.g. ..._seed1.json)
            if seed is None:
                fname = os.path.basename(fpath)
                if "seed" in fname:
                    try:
                        seed = int(fname.split("seed")[-1].replace(".json", ""))
                    except ValueError:
                        seed = 1
                else:
                    seed = 1

            trials = data.get("trials", [])
            
            if not extractor or not task or not trials:
                continue

            # Filter by benchmark if specified (check both task string and filename)
            if task_filter is not None:
                tf_str = str(task_filter)
                if tf_str not in str(task) and tf_str not in fpath:
                    continue

            # Extract costs per trial iteration (support baseline and ei schemas)
            costs = []
            for t in trials:
                cost = None
                if isinstance(t, dict):
                    if "trial_value" in t and isinstance(t["trial_value"], dict) and "cost" in t["trial_value"]:
                        cost = t["trial_value"]["cost"]
                    elif "cost" in t:
                        cost = t["cost"]
                
                if cost is not None and not np.isnan(cost):
                    costs.append(float(cost))
            
            if not costs:
                continue

            # Compute cumulative minimum (anytime best performance)
            cum_min = np.minimum.accumulate(costs)
            
            # Truncate or pad to max_steps
            if len(cum_min) > max_steps:
                cum_min = cum_min[:max_steps]
            elif len(cum_min) < max_steps:
                cum_min = np.pad(cum_min, (0, max_steps - len(cum_min)), mode='edge')

            key = (extractor, task, seed)
            runs[key] = cum_min

            # Track task min/max for normalization
            min_val = np.min(cum_min)
            max_val = np.max(cum_min)
            if task not in task_global_min:
                task_global_min[task] = min_val
                task_global_max[task] = max_val
            else:
                task_global_min[task] = min(task_global_min[task], min_val)
                task_global_max[task] = max(task_global_max[task], max_val)

        except Exception as e:
            continue

    # Group trajectories by extractor (use raw cumulative minimum if single task_filter is set)
    extractor_trajectories = {} # extractor -> list of 1D arrays
    for (extractor, task, seed), cum_min in runs.items():
        if task_filter is not None:
            traj = cum_min
        else:
            min_v = task_global_min[task]
            max_v = task_global_max[task]
            denom = max_v - min_v
            traj = (cum_min - min_v) / denom if denom > 1e-8 else np.zeros_like(cum_min)

        if extractor not in extractor_trajectories:
            extractor_trajectories[extractor] = []
        extractor_trajectories[extractor].append(traj)

    # Compute mean and SEM for each extractor
    results = {}
    iterations = np.arange(1, max_steps + 1)

    for extractor, trajs in extractor_trajectories.items():
        arr = np.array(trajs) # shape (N_runs, max_steps)
        n_runs = arr.shape[0]
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean)
        sem = std / np.sqrt(n_runs) if n_runs > 0 else np.zeros_like(mean)

        results[extractor] = {
            "iterations": iterations,
            "mean": mean,
            "std": std,
            "sem": sem,
            "n_runs": n_runs
        }

    return results

def plot_anytime_performance(results_dir="results/epistemic_ei_highdim", output_path="results/epistemic_ei_highdim/anytime_performance_1111.png", task_filter="1111"):
    """
    Generates and saves a right-continuous ('rechtsstetig') anytime performance plot
    with SEM error bands across all runs for the specified benchmark task.
    """
    trajectories = compute_anytime_trajectories(results_dir, task_filter=task_filter)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')

    # Color palette & styling for extractors
    colors = {
        "standard_disagreement": "#1f77b4", # blue
        "proximity_b": "#ff7f0e",            # orange
        "proximity_auto_lambda": "#2ca02c",  # green
        "proximity_bc": "#d62728",           # red
        "chen_variance": "#9467bd",          # purple
        "standard_proximity": "#8c564b",     # brown
        "likelihood_credal": "#e377c2",      # pink
        "shaker_entropy": "#7f7f7f",         # gray
        "smac3_bo": "#000000"                # black (baseline)
    }

    labels = {
        "standard_disagreement": "Standard Disagreement",
        "proximity_b": "Proximity Method B",
        "proximity_auto_lambda": "Proximity Auto-Lambda",
        "proximity_bc": "Proximity Method B+C",
        "chen_variance": "Chen Variance",
        "standard_proximity": "Standard Proximity",
        "likelihood_credal": "Likelihood Credal",
        "shaker_entropy": "Shaker Entropy",
        "smac3_bo": "SMAC3 Baseline (Standard)"
    }

    # Sort extractors by final mean performance (lowest cost/regret first)
    sorted_extractors = sorted(trajectories.keys(), key=lambda e: trajectories[e]["mean"][-1])

    for ext in sorted_extractors:
        data = trajectories[ext]
        iters = data["iterations"]
        mean = data["mean"]
        sem = data["sem"]

        # Transform to right-continuous ('rechtsstetig') step coordinates
        x_step, mean_step = make_step_coords(iters, mean)
        _, sem_step = make_step_coords(iters, sem)

        lower_step = mean_step - sem_step
        upper_step = mean_step + sem_step

        color = colors.get(ext, None)
        label = f"{labels.get(ext, ext)} (N={data['n_runs']})"
        linewidth = 2.5 if ext in ["smac3_bo", "standard_disagreement"] else 1.8
        linestyle = "--" if ext == "smac3_bo" else "-"

        # Plot right-continuous step line
        plt.plot(x_step, mean_step, label=label, color=color, linewidth=linewidth, linestyle=linestyle)
        # Plot right-continuous step error band (SEM = std / sqrt(N))
        plt.fill_between(x_step, lower_step, upper_step, color=color, alpha=0.15)

    title_task = f"Benchmark {task_filter}" if task_filter else "High-Dim Benchmarks"
    plt.xlabel("Function Evaluations (Iteration $t$)", fontsize=12, fontweight='bold')
    plt.ylabel("Best Target Cost $y^*_t$", fontsize=12, fontweight='bold')
    plt.title(f"Anytime Optimization Performance ({title_task})", fontsize=14, fontweight='bold', pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path)
    plt.close()

    print(f"[SUCCESS] Anytime performance plot saved to: {output_path} and {pdf_path}")
    return output_path

if __name__ == "__main__":
    plot_anytime_performance(task_filter="1111")

