#!/usr/bin/env python3
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

def load_telemetry_files(base_dir, task_keyword="1040"):
    results = {}
    
    # 1. Baseline runs
    baseline_files = glob.glob(os.path.join(base_dir, "baseline", f"*{task_keyword}*.json"))
    for filepath in baseline_files:
        with open(filepath, "r") as f:
            data = json.load(f)
        approach = "smac3_bo"
        seed = data.get("seed", 1)
        trials = data.get("trials", [])
        costs = []
        for t in trials:
            if "trial_value" in t and "cost" in t["trial_value"]:
                costs.append(t["trial_value"]["cost"])
            elif "cost" in t:
                costs.append(t["cost"])
        if costs:
            results.setdefault(approach, {})[seed] = costs

    # 2. DyRF Epistemic runs
    ei_files = glob.glob(os.path.join(base_dir, "ei", f"*{task_keyword}*.json"))
    for filepath in ei_files:
        with open(filepath, "r") as f:
            data = json.load(f)
        approach = data.get("extractor_name", "unknown")
        # Extract seed from filename if not in root json
        filename = os.path.basename(filepath)
        if "seed" in data and isinstance(data["seed"], int):
            seed = data["seed"]
        else:
            # parse seed from filename pattern seedX.json
            import re
            m = re.search(r"seed(\d+)\.json$", filename)
            seed = int(m.group(1)) if m else 1
            
        trials = data.get("trials", [])
        costs = []
        for t in trials:
            if "cost" in t:
                costs.append(t["cost"])
            elif "trial_value" in t and "cost" in t["trial_value"]:
                costs.append(t["trial_value"]["cost"])
        if costs:
            results.setdefault(approach, {})[seed] = costs

    return results

def compute_anytime_trajectories(results, max_trials=50):
    trajectories = {}
    
    for approach, seed_dict in results.items():
        seed_anytimes = []
        for seed, costs in seed_dict.items():
            # Pad or truncate to max_trials
            costs_arr = np.array(costs[:max_trials])
            # Compute running minimum (anytime best cost)
            anytime_best = np.minimum.accumulate(costs_arr)
            if len(anytime_best) < max_trials:
                # pad with last value if fewer than max_trials
                last_val = anytime_best[-1] if len(anytime_best) > 0 else np.nan
                anytime_best = np.pad(anytime_best, (0, max_trials - len(anytime_best)), mode='edge')
            seed_anytimes.append(anytime_best)
            
        if seed_anytimes:
            matrix = np.array(seed_anytimes) # shape: (n_seeds, max_trials)
            mean = np.mean(matrix, axis=0)
            n_seeds = matrix.shape[0]
            stderr = np.std(matrix, axis=0, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(mean)
            trajectories[approach] = {
                "trials": np.arange(1, max_trials + 1),
                "mean": mean,
                "stderr": stderr,
                "n_seeds": n_seeds,
                "matrix": matrix
            }
            
    return trajectories

def make_step_fill_vectors(x, y):
    """
    Creates right-continuous ('where=post') coordinate vectors for fill_between.
    """
    x_fill = np.empty(2 * len(x) - 1)
    x_fill[0::2] = x
    x_fill[1::2] = x[1:]
    
    y_fill = np.empty(2 * len(y) - 1)
    y_fill[0::2] = y
    y_fill[1::2] = y[:-1]
    
    return x_fill, y_fill

def plot_anytime_performance(trajectories, output_path="results/anytime_rbv2_super_1040.png"):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Custom color palette & line styles
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
    
    # Sort approaches so top performers / baselines stand out
    sorted_approaches = sorted(trajectories.keys(), key=lambda a: trajectories[a]["mean"][-1])
    
    for app in sorted_approaches:
        data = trajectories[app]
        x = data["trials"]
        mean = data["mean"]
        stderr = data["stderr"]
        
        color = colors.get(app, "#333333")
        label = labels.get(app, app)
        linewidth = 2.2 if app in ["smac3_bo", "standard_proximity", "proximity_auto_lambda"] else 1.5
        linestyle = "--" if app == "smac3_bo" else "-"
        
        # 1. Right-continuous step mean line (where='post')
        plt.step(x, mean, where='post', label=f"{label} ({data['n_seeds']} seeds)", 
                 color=color, linewidth=linewidth, linestyle=linestyle, zorder=4 if app=="standard_proximity" else 3)
        
        # 2. Right-continuous step shaded band (mean - stderr to mean + stderr)
        x_step, lower_step = make_step_fill_vectors(x, mean - stderr)
        _, upper_step = make_step_fill_vectors(x, mean + stderr)
        
        plt.fill_between(x_step, lower_step, upper_step, color=color, alpha=0.15, zorder=2)
        
    plt.title("Anytime Performance on YAHPO rbv2_super_1040 (38D Joint HPO Task)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("BO Evaluation Trials (t)", fontsize=12)
    plt.ylabel("Anytime Best Cost (y_best, lower is better)", fontsize=12)
    plt.xlim(1, 50)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path} and {pdf_path}")

def main():
    base_dir = "results/epistemic_ei_highdim"
    results = load_telemetry_files(base_dir, task_keyword="1040")
    print(f"Loaded {len(results)} approaches:")
    for app, s_dict in results.items():
        print(f"  - {app}: {len(s_dict)} seeds ({sorted(list(s_dict.keys()))})")
        
    trajectories = compute_anytime_trajectories(results, max_trials=50)
    
    # Print summary table of final trial results
    print("\n=======================================================")
    print("ANYTIME PERFORMANCE SUMMARY AT TRIAL 50 (rbv2_super_1040)")
    print("=======================================================")
    print(f"{'Approach':<25} | {'Mean Best Cost':<15} | {'Std Error':<10} | {'Seeds':<5}")
    print("-" * 65)
    for app in sorted(trajectories.keys(), key=lambda a: trajectories[a]["mean"][-1]):
        m = trajectories[app]["mean"][-1]
        se = trajectories[app]["stderr"][-1]
        ns = trajectories[app]["n_seeds"]
        print(f"{app:<25} | {m:<15.6f} | {se:<10.6f} | {ns:<5}")
    print("=======================================================\n")
    
    plot_anytime_performance(trajectories, "results/epistemic_ei_highdim/anytime_rbv2_super_1040.png")

if __name__ == "__main__":
    main()
