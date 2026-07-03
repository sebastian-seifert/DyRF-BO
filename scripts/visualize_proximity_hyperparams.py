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

def load_data(results_dir="results"):
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' does not exist.")
        return None

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
                pass
            
            if meta["gap"] == "sparse":
                meta["mult"] = int(meta["mult"]) if meta["mult"] else 12
                meta["law"] = meta["law"] if meta["law"] else "linear"
            else:
                meta["mult"] = None
                meta["law"] = None

            # Compute average metric value across all functions
            for app in ["Standard", "Proximity"]:
                meta[f"{app}_auroc"] = np.nanmean(content["results_all"][app]["auroc"])
                meta[f"{app}_spearman"] = np.nanmean(content["results_all"][app]["spearman"])
                meta[f"{app}_brier"] = np.nanmean(content["results_all"][app]["brier"])
                meta[f"{app}_mi"] = np.nanmean(content["results_all"][app]["mi"])
                meta[f"{app}_jsd"] = np.nanmean(content["results_all"][app]["jsd"])

            meta["content"] = content
            data_store.append(meta)
        except Exception as e:
            print(f"Warning: Could not parse file {filename}: {e}")

    return data_store

def plot_standalone_hyperparams(data, metric, save_dir):
    """Plots the standalone impact of each hyperparameter on the specified metric."""
    # Mapping of user-facing metric names
    metric_labels = {
        "auroc": "AUROC",
        "brier": "Brier Score",
        "spearman": "Spearman Rank Correlation",
        "mi": "Mutual Information",
        "jsd": "Jensen-Shannon Divergence"
    }
    
    label = metric_labels.get(metric, metric.upper())
    key = f"Proximity_{metric}"
    
    # Standalone hyperparameters to analyze
    hyperparams = {
        "k": ("K (Neighborhood Size)", [20, 100, 500]),
        "rf": ("RF Trees Configuration", [1, 3, 5]),
        "gap": ("Gap Type", ["empty", "sparse"]),
        "law": ("Scaling Law (Sparse only)", ["linear", "leaf"]),
        "mult": ("Sparse Multiplier (Sparse only)", [5, 15, 50])
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (hp, (name, values)) in enumerate(hyperparams.items()):
        ax = axes[i]
        
        # Filter data for this hyperparam
        hp_means = []
        hp_stds = []
        
        for val in values:
            if hp in ["law", "mult"] and val is not None:
                # Only look at sparse runs for sparse-specific hyperparameters
                subset = [d for d in data if d[hp] == val and d["gap"] == "sparse"]
            else:
                subset = [d for d in data if d[hp] == val]
            
            vals = [d[key] for d in subset if not np.isnan(d[key])]
            if vals:
                hp_means.append(np.mean(vals))
                hp_stds.append(np.std(vals) / np.sqrt(len(vals)))  # Standard error of mean
            else:
                hp_means.append(np.nan)
                hp_stds.append(np.nan)
        
        # Plotting
        x_pos = np.arange(len(values))
        ax.bar(x_pos, hp_means, yerr=hp_stds, color='#2ecc71', alpha=0.8, edgecolor='#27ae60', capsize=5, width=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(v) for v in values])
        ax.set_title(name)
        ax.set_ylabel(label)
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        
    # Hide the 6th subplot
    fig.delaxes(axes[5])
    
    plt.suptitle(f"Standalone Hyperparameter Impact on Proximity UQ: {label}", fontsize=14, y=0.98)
    plt.tight_layout()
    
    filename = f"hyperparam_impact_{metric}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved hyperparameter impact plot for {metric} to: {os.path.join(save_dir, filename)}")

def plot_best_proximity_vs_standard(data, metric, save_dir):
    """Compares the best performing Proximity configuration for a metric against the corresponding Standard UQ across dimensions."""
    metric_labels = {
        "auroc": "AUROC",
        "brier": "Brier Score",
        "spearman": "Spearman Correlation",
        "mi": "Mutual Information",
        "jsd": "Jensen-Shannon Divergence"
    }
    
    label = metric_labels.get(metric, metric.upper())
    key = f"Proximity_{metric}"
    
    # Find best proximity config
    # For Brier Score, lower is better. For others, higher is better.
    if metric == "brier":
        best_run = min(data, key=lambda x: x[key])
    else:
        best_run = max(data, key=lambda x: x[key])
        
    print(f"Best Proximity config for {label}: RF={best_run['rf']}, K={best_run['k']}, gap={best_run['gap']} (mean {key}={best_run[key]:.4f})")
    
    results_by_dim = best_run["content"]["results_by_dim"]
    dims = sorted(results_by_dim.keys(), key=lambda x: int(x.replace("D", "")))
    
    plt.figure(figsize=(9, 5.5))
    
    prox_means, prox_stds = [], []
    std_means, std_stds = [], []
    
    for dim in dims:
        p_vals = results_by_dim[dim]["Proximity"][metric]
        s_vals = results_by_dim[dim]["Standard"][metric]
        
        prox_means.append(np.nanmean(p_vals))
        prox_stds.append(np.nanstd(p_vals) / np.sqrt(len(p_vals)))
        
        std_means.append(np.nanmean(s_vals))
        std_stds.append(np.nanstd(s_vals) / np.sqrt(len(s_vals)))
        
    x_pos = np.arange(len(dims))
    width = 0.35
    
    plt.bar(x_pos - width/2, prox_means, yerr=prox_stds, width=width, label="Best Proximity UQ", color='#2ecc71', edgecolor='#27ae60', capsize=4)
    plt.bar(x_pos + width/2, std_means, yerr=std_stds, width=width, label="Standard RF Variance", color='#3498db', edgecolor='#2980b9', capsize=4)
    
    plt.xticks(x_pos, dims)
    plt.xlabel("Target Function Dimension")
    plt.ylabel(label)
    
    gap_desc = f"gap={best_run['gap']}"
    if best_run['gap'] == 'sparse':
         gap_desc += f" ({best_run['law']}, mult={best_run['mult']})"
    plt.title(f"Comparison: Best Proximity vs. Standard RF Variance ({label})\nBest Config: RF={best_run['rf']}, K={best_run['k']}, {gap_desc}", pad=15)
    
    if metric == "auroc":
        plt.axhline(0.5, color='#e74c3c', linestyle='--', label="Random Guess (0.5)")
        
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')
    plt.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5')
    
    filename = f"best_vs_standard_{metric}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison plot for {metric} to: {os.path.join(save_dir, filename)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Proximity Hyperparameters")
    parser.add_argument("--use_density_scaling", action="store_true", help="Plot density scaling results from results/density_scaling/")
    args = parser.parse_args()

    results_dir = "results/density_scaling" if args.use_density_scaling else "results"
    save_dir = "results/density_scaling/proximity" if args.use_density_scaling else "results/proximity"

    print(f"Loading cluster results from: {results_dir}")
    data = load_data(results_dir)
    if not data:
        print("No results loaded.")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created subfolder: {save_dir}\n")
    
    metrics = ["auroc", "brier", "spearman", "mi", "jsd"]
    
    for metric in metrics:
        print(f"Generating plots for metric: {metric}")
        plot_standalone_hyperparams(data, metric, save_dir)
        plot_best_proximity_vs_standard(data, metric, save_dir)
        print("-" * 50)
        
    print(f"\n🎉 All proximity-specific hyperparameter and best-vs-standard plots saved in '{save_dir}'!")

if __name__ == "__main__":
    main()
