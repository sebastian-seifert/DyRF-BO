#!/usr/bin/env python3
import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

# Set clean, publication-ready style parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['grid.color'] = '#e5e5e5'
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

def parse_results(results_dir="results"):
    """Scans results/ folder and parses all JSON reports to build a data structure."""
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' does not exist.")
        return None

    # Matches files like: uncertainty_quantification_results_rf1_k20_sparse_m12_linear_20260630_113000.json
    pattern = re.compile(
        r"uncertainty_quantification_results_rf(?P<rf>\d)_k(?P<k>\w+?)_(?P<gap>empty|sparse)(?:_m(?P<mult>\d+))?(?:_(?P<law>linear|fractional|leaf))?(?:_(?P<ds>ds))?_\d{8}_\d{6}\.json"
    )

    data_store = []

    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"):
            continue
        match = pattern.match(filename)
        if not match:
            continue

        meta = match.groupdict()
        filepath = os.path.join(results_dir, filename)
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = json.load(f)
            
            meta["rf"] = int(meta["rf"])
            try:
                meta["k"] = int(meta["k"])
            except ValueError:
                pass  # Keep as 'auto' or 'all'
            
            if meta["gap"] == "sparse":
                meta["mult"] = int(meta["mult"]) if meta["mult"] else 12
                meta["law"] = meta["law"] if meta["law"] else "linear"
            else:
                meta["mult"] = None
                meta["law"] = None

            meta["content"] = content
            data_store.append(meta)
        except Exception as e:
            print(f"Warning: Could not parse file {filename}: {e}")

    return data_store

def plot_sparsity_transition(data, figures_dir):
    """Generates a Sparsity Transition Plot (AUROC vs. Sparse Multiplier)."""
    # Filter only sparse runs with rf_config=1 and K=20
    sparse_runs = [d for d in data if d["gap"] == "sparse" and d["rf"] == 1 and d["k"] == 20]
    if not sparse_runs:
        print("Sparsity Transition Plot: No matching sparse results found.")
        return

    # Group by scaling law and multiplier
    laws = ["linear", "fractional", "leaf"]
    multipliers = sorted(list(set(d["mult"] for d in sparse_runs)))

    plt.figure(figsize=(8, 5))
    colors = {"linear": "#3498db", "fractional": "#9b59b6", "leaf": "#e67e22"}
    
    # Plot Standard approach (which stays stable across scaling laws)
    std_means = []
    for mult in multipliers:
        # Find runs with this multiplier
        runs = [r for r in sparse_runs if r["mult"] == mult]
        if runs:
            # Average Standard AUROC across available scaling laws for standard reference
            std_auroc = np.mean([np.mean(run["content"]["results_all"]["Standard"]["auroc"]) for run in runs])
            std_means.append(std_auroc)
        else:
            std_means.append(np.nan)
            
    plt.plot(multipliers, std_means, marker='s', linestyle='--', color='#7f8c8d', linewidth=1.5, label="Standard RF Variance")

    # Plot Proximity approach under each scaling law
    for law in laws:
        prox_means = []
        for mult in multipliers:
            run = next((r for r in sparse_runs if r["law"] == law and r["mult"] == mult), None)
            if run:
                mean_auroc = np.mean(run["content"]["results_all"]["Proximity"]["auroc"])
                prox_means.append(mean_auroc)
            else:
                prox_means.append(np.nan)
        
        if not all(np.isnan(prox_means)):
            plt.plot(multipliers, prox_means, marker='o', color=colors[law], linewidth=2, label=f"Proximity (Law: {law})")

    plt.title("Sparsity Transition: Impact of Sparse Gap Refilling on AUROC", pad=15)
    plt.xlabel("Sparse Multiplier (Scale Factor)")
    plt.ylabel("Overall AUROC (OOD Discrimination)")
    plt.axhline(0.5, color='#999999', linestyle=':', label="Random Guess")
    plt.grid(True, linestyle='-', alpha=0.4)
    plt.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5')
    
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, "sparsity_transition_auroc.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved sparsity transition plot to: {save_path}")

def plot_dimensionality_trend(data, figures_dir):
    """Plots AUROC vs Dimension comparing Standard vs Proximity under empty gap."""
    # Select empty gap with RF=1 and K=20
    empty_runs = [d for d in data if d["gap"] == "empty" and d["rf"] == 1 and d["k"] == 20]
    if not empty_runs:
        print("Dimensionality Trend: No empty gap results found.")
        return

    # Use the latest file
    run = empty_runs[-1]
    results_by_dim = run["content"]["results_by_dim"]

    dims = ["1D", "2D", "3D", "4D", "5D", "6D", "7D", "8D", "9D", "10D"]
    approaches = ["Standard", "Proximity"]
    colors = {"Standard": "#3498db", "Proximity": "#2ecc71"}

    plt.figure(figsize=(8, 5))

    for app in approaches:
        means, stds = [], []
        for dim in dims:
            if dim in results_by_dim and app in results_by_dim[dim]:
                auroc_vals = results_by_dim[dim][app]["auroc"]
                means.append(np.mean(auroc_vals))
                stds.append(np.std(auroc_vals))
            else:
                means.append(np.nan)
                stds.append(np.nan)

        means_arr = np.array(means, dtype=float)
        stds_arr = np.array(stds, dtype=float)
        x_dims = np.arange(len(dims))
        valid = ~np.isnan(means_arr)

        plt.plot(x_dims[valid], means_arr[valid], marker='o', linewidth=2, color=colors[app], label=f"{app} UQ")
        plt.fill_between(x_dims[valid], means_arr[valid] - stds_arr[valid], means_arr[valid] + stds_arr[valid],
                         color=colors[app], alpha=0.1)

    plt.title("Dimensionality Trend: OOD Discrimination Quality Across Dimensions", pad=15)
    plt.xlabel("Dimensionality of Target Function")
    plt.ylabel("AUROC (OOD Discrimination)")
    plt.xticks(np.arange(len(dims)), dims)
    plt.axhline(0.5, color='#999999', linestyle='--', label="Random Guess")
    plt.grid(True, linestyle='-', alpha=0.4)
    plt.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5')
    
    save_path = os.path.join(figures_dir, "dimensionality_trend_auroc.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved dimensionality trend plot to: {save_path}")

def plot_k_sensitivity(data, figures_dir):
    """Plots K Neighbors Sensitivity (AUROC vs K Neighbors)."""
    # Filter empty gap, RF config 1
    k_runs = [d for d in data if d["gap"] == "empty" and d["rf"] == 1]
    if not k_runs:
        print("K Sensitivity Plot: No empty gap results found.")
        return

    # Extract K values
    k_vals = []
    prox_means = []
    
    for run in k_runs:
        k_val = run["k"]
        if isinstance(k_val, int):
            k_vals.append(k_val)
            prox_means.append(np.mean(run["content"]["results_all"]["Proximity"]["auroc"]))

    if not k_vals:
        print("K Sensitivity Plot: No integer K values found.")
        return

    # Sort together
    sorted_idx = np.argsort(k_vals)
    k_vals = np.array(k_vals)[sorted_idx]
    prox_means = np.array(prox_means)[sorted_idx]

    plt.figure(figsize=(7, 4.5))
    plt.plot(k_vals, prox_means, marker='o', linewidth=2, color="#9b59b6", label="Proximity UQ")
    
    plt.title("Sensitivity Analysis: Impact of Neighborhood Size K on Proximity UQ", pad=15)
    plt.xlabel("K (Number of Nearest Neighbors)")
    plt.ylabel("Overall AUROC")
    plt.grid(True, linestyle='-', alpha=0.4)
    plt.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5')
    
    save_path = os.path.join(figures_dir, "k_sensitivity_auroc.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved K sensitivity plot to: {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Grid Results")
    parser.add_argument("--use_density_scaling", action="store_true", help="Plot density scaling results from results/density_scaling/")
    args = parser.parse_args()

    results_dir = "results/density_scaling" if args.use_density_scaling else "results"
    figures_dir = "figures/density_scaling" if args.use_density_scaling else "figures"

    print(f"Scanning results from '{results_dir}' and generating plots to '{figures_dir}'...")
    data = parse_results(results_dir)
    if not data:
        print("No structured JSON results found.")
        return

    os.makedirs(figures_dir, exist_ok=True)
    plot_sparsity_transition(data, figures_dir)
    plot_dimensionality_trend(data, figures_dir)
    plot_k_sensitivity(data, figures_dir)
    print(f"\n🎉 Visualizations successfully updated and saved in '{figures_dir}'!")

if __name__ == "__main__":
    main()
