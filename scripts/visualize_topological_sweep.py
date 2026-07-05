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

def load_all_sweep_results(results_dir="results"):
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' does not exist.")
        return None

    pattern = re.compile(
        r"uncertainty_quantification_results_rf(?P<rf>\d)_k(?P<k>\w+?)_(?P<gap>empty|sparse)"
        r"(?:_m(?P<mult>\d+))?(?:_(?P<law>linear|fractional|leaf))?"
        r"(?:_lambda(?P<lambda>[\d\.]+))?(?:_(?P<ds>ds))?_\d{8}_\d{6}\.json"
    )

    data_store = []

    for root, dirs, files in os.walk(results_dir):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            match = pattern.match(filename)
            if not match:
                continue

            meta = match.groupdict()
            filepath = os.path.join(root, filename)
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = json.load(f)
                
                meta["rf"] = int(meta["rf"])
                try:
                    meta["k"] = int(meta["k"])
                except ValueError:
                    meta["k"] = str(meta["k"]) # 'auto'
                
                if meta["gap"] == "sparse":
                    meta["mult"] = int(meta["mult"]) if meta["mult"] else 12
                    meta["law"] = meta["law"] if meta["law"] else "linear"
                else:
                    meta["mult"] = 0
                    meta["law"] = "none"

                meta["lambda"] = float(meta["lambda"]) if meta["lambda"] else 0.0
                meta["ds"] = True if meta["ds"] else False
                
                # Identify Method label:
                if meta["lambda"] == 0.0 and not meta["ds"]:
                    meta["method_type"] = "Standard Proximity"
                elif meta["lambda"] > 0.0 and not meta["ds"]:
                    meta["method_type"] = f"Topological (lambda={meta['lambda']})"
                elif meta["lambda"] > 0.0 and meta["ds"]:
                    meta["method_type"] = f"Topological + DS"
                else:
                    meta["method_type"] = "Unknown"

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
                pass

    return data_store

def plot_best_vs_standard_all_metrics(data, save_dir):
    """Compares the best overall Proximity config (by AUROC) against Standard RFGAP across dimensions and metrics."""
    # Find best overall config by AUROC
    best_run = max(data, key=lambda x: x["Proximity_auroc"])
    
    print(f"Best AUROC config selected for line plot: RF={best_run['rf']}, K={best_run['k']}, lambda={best_run['lambda']}, ds={best_run['ds']}")
    
    results_by_dim = best_run["content"]["results_by_dim"]
    dims = sorted(results_by_dim.keys(), key=lambda x: int(x.replace("D", "")))
    
    metrics = ["auroc", "brier", "spearman"]
    metric_labels = {
        "auroc": "AUROC (Higher is Better)",
        "brier": "Brier Score (Lower is Better)",
        "spearman": "Spearman Correlation"
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        prox_means = []
        std_means = []
        
        for dim in dims:
            p_vals = results_by_dim[dim]["Proximity"][metric]
            s_vals = results_by_dim[dim]["Standard"][metric]
            prox_means.append(np.nanmean(p_vals))
            std_means.append(np.nanmean(s_vals))
            
        x_pos = np.arange(len(dims))
        ax.plot(x_pos, prox_means, marker='o', color='#2ecc71', linewidth=2, label="Topological Proximity + DS")
        ax.plot(x_pos, std_means, marker='s', color='#3498db', linewidth=1.5, linestyle='--', label="Standard RFGAP Baseline")
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(dims)
        ax.set_xlabel("Dimension")
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(f"{metric.upper()} across Dimensions")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5')
        
        if metric == "auroc":
            ax.axhline(0.5, color='#e74c3c', linestyle=':', label="Random Guess (0.5)")
            ax.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5')
            
    gap_desc = f"gap={best_run['gap']}"
    if best_run['gap'] == 'sparse':
         gap_desc += f" ({best_run['law']}, mult={best_run['mult']})"
    plt.suptitle(f"Performance Comparison: Best Topological Proximity UQ vs. Standard RFGAP Baseline\nBest Config: RF={best_run['rf']}, K={best_run['k']}, {gap_desc}, lambda={best_run['lambda']}, ds={best_run['ds']}", y=1.02, fontsize=13)
    plt.tight_layout()
    
    filename = "best_vs_standard_all_metrics.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved dimensional comparisons to: {os.path.join(save_dir, filename)}")

def plot_approaches_bar_chart(data, save_dir):
    """Generates a grouped bar chart comparing overall averages for the different method types."""
    # Group by key method categories:
    # 1. Standard RFGAP (can extract from any run)
    # 2. Standard Proximity (lambda=0, ds=False)
    # 3. Topological Proximity (lambda=1.0, ds=False)
    # 4. Topological Proximity + DS (lambda=5.0, ds=True)
    
    groups = {
        "Standard RFGAP": [d["Standard_auroc"] for d in data], # extract standard baselines
        "Standard Proximity": [d["Proximity_auroc"] for d in data if d["lambda"] == 0.0 and not d["ds"]],
        "Topological (lambda=1.0)": [d["Proximity_auroc"] for d in data if d["lambda"] == 1.0 and not d["ds"]],
        "Topological + DS (lambda=5.0)": [d["Proximity_auroc"] for d in data if d["lambda"] == 5.0 and d["ds"]]
    }
    
    metrics = ["auroc", "brier", "spearman"]
    metric_labels = ["AUROC", "Brier Score", "Spearman r"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.18
    colors = ["#34495e", "#bdc3c7", "#9b59b6", "#2ecc71"]
    
    for idx, (label, _) in enumerate(groups.items()):
        means = []
        stds = []
        for metric in metrics:
            if label == "Standard RFGAP":
                vals = [d[f"Standard_{metric}"] for d in data]
            else:
                subset = [d for d in data if d["lambda"] == (1.0 if "lambda=1.0" in label else (5.0 if "+ DS" in label else 0.0)) and d["ds"] == ("+ DS" in label)]
                vals = [d[f"Proximity_{metric}"] for d in subset]
                
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals) / np.sqrt(len(vals)) if vals else 0.0)
            
        offset = (idx - 1.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=label, color=colors[idx], alpha=0.9, capsize=3, error_kw=dict(lw=0.8))
        
    ax.set_ylabel('Overall Metric Value (Mean ± SEM)')
    ax.set_title('Overall Performance Comparison by UQ Method Type', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    filename = "approaches_comparison.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved approaches comparison bar chart to: {os.path.join(save_dir, filename)}")

def plot_hyperparameter_impacts(data, save_dir):
    """Plots the impact of different hyperparameter settings (K, RF, lambda, ds) on AUROC and Brier Score."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. K Neighbors
    k_vals = [20, 100, 500, "auto"]
    k_means_auroc = []
    k_se_auroc = []
    for k in k_vals:
        subset = [d for d in data if d["k"] == k]
        vals = [d["Proximity_auroc"] for d in subset]
        k_means_auroc.append(np.mean(vals) if vals else 0)
        k_se_auroc.append(np.std(vals)/np.sqrt(len(vals)) if vals else 0)
    axes[0, 0].bar(np.arange(len(k_vals)), k_means_auroc, yerr=k_se_auroc, color="#9b59b6", alpha=0.8, width=0.5, capsize=4)
    axes[0, 0].set_xticks(np.arange(len(k_vals)))
    axes[0, 0].set_xticklabels([str(k) for k in k_vals])
    axes[0, 0].set_title("Neighborhood Size K")
    axes[0, 0].set_ylabel("Overall AUROC")
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.4)

    # 2. RF configurations
    rf_vals = [1, 3, 5]
    rf_means_brier = []
    rf_se_brier = []
    for rf in rf_vals:
        subset = [d for d in data if d["rf"] == rf]
        vals = [d["Proximity_brier"] for d in subset]
        rf_means_brier.append(np.mean(vals) if vals else 0)
        rf_se_brier.append(np.std(vals)/np.sqrt(len(vals)) if vals else 0)
    axes[0, 1].bar(np.arange(len(rf_vals)), rf_means_brier, yerr=rf_se_brier, color="#e67e22", alpha=0.8, width=0.4, capsize=4)
    axes[0, 1].set_xticks(np.arange(len(rf_vals)))
    axes[0, 1].set_xticklabels([f"RF Config {rf}" for rf in rf_vals])
    axes[0, 1].set_title("Random Forest Configuration (Leaf Size Bounds)")
    axes[0, 1].set_ylabel("Brier Score (Lower is Better)")
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.4)

    # 3. Decay Lambda
    lambda_vals = [0.0, 1.0, 5.0]
    lambda_means_auroc = []
    lambda_se_auroc = []
    for l in lambda_vals:
        subset = [d for d in data if d["lambda"] == l]
        vals = [d["Proximity_auroc"] for d in subset]
        lambda_means_auroc.append(np.mean(vals) if vals else 0)
        lambda_se_auroc.append(np.std(vals)/np.sqrt(len(vals)) if vals else 0)
    axes[1, 0].bar(np.arange(len(lambda_vals)), lambda_means_auroc, yerr=lambda_se_auroc, color="#3498db", alpha=0.8, width=0.4, capsize=4)
    axes[1, 0].set_xticks(np.arange(len(lambda_vals)))
    axes[1, 0].set_xticklabels([f"lambda={l}" for l in lambda_vals])
    axes[1, 0].set_title("Topological Decay Parameter lambda")
    axes[1, 0].set_ylabel("Overall AUROC")
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.4)

    # 4. Density Scaling
    ds_vals = [False, True]
    ds_means_brier = []
    ds_se_brier = []
    for ds in ds_vals:
        subset = [d for d in data if d["ds"] == ds]
        vals = [d["Proximity_brier"] for d in subset]
        ds_means_brier.append(np.mean(vals) if vals else 0)
        ds_se_brier.append(np.std(vals)/np.sqrt(len(vals)) if vals else 0)
    axes[1, 1].bar(np.arange(len(ds_vals)), ds_means_brier, yerr=ds_se_brier, color="#2ecc71", alpha=0.8, width=0.3, capsize=4)
    axes[1, 1].set_xticks(np.arange(len(ds_vals)))
    axes[1, 1].set_xticklabels(["Standard (No DS)", "Density Scaled (DS)"])
    axes[1, 1].set_title("Density Scaling (Method C)")
    axes[1, 1].set_ylabel("Brier Score (Lower is Better)")
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.4)

    plt.suptitle("Hyperparameter Sensitivity & Impact Analysis (Overall Sweep Averages)", fontsize=14, y=0.98)
    plt.tight_layout()
    
    filename = "hyperparam_impact_comparison.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved hyperparameter impact comparisons to: {os.path.join(save_dir, filename)}")

def main():
    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading all results files recursively...")
    data = load_all_sweep_results("results")
    if not data:
        print("No results loaded. Exiting.")
        return
        
    print(f"Loaded {len(data)} configurations. Generating plots...")
    
    plot_best_vs_standard_all_metrics(data, save_dir)
    plot_approaches_bar_chart(data, save_dir)
    plot_hyperparameter_impacts(data, save_dir)
    
    print(f"\n🎉 All sweep visualization figures successfully saved in '{save_dir}/'!")

if __name__ == "__main__":
    main()
