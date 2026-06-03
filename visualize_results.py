import os
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
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# Create figures directory if not exists
os.makedirs("figures", exist_ok=True)

# -----------------------------------------------------------------------------
# Data from statistical_tests.txt
# -----------------------------------------------------------------------------
approaches = ["Standard", "Shaker", "Chen"]
colors = {
    "Standard": "#3498db",  # Steel Blue
    "Shaker": "#e74c3c",    # Coral Red
    "Chen": "#2ecc71"       # Emerald Green
}

# All Functions Summary
all_funcs_data = {
    "auroc":    {"mean": [0.6924, 0.6786, 0.6905], "std": [0.1886, 0.1943, 0.1882]},
    "spearman": {"mean": [0.0106, -0.0053, 0.0115], "std": [0.3493, 0.3214, 0.3479]},
    "brier":    {"mean": [0.0771, 0.0774, 0.0770], "std": [0.0759, 0.0761, 0.0759]},
    "mi":       {"mean": [0.3633, 0.3110, 0.3587], "std": [0.2459, 0.2101, 0.2487]},
    "jsd":      {"mean": [0.3013, 0.2637, 0.2960], "std": [0.1611, 0.1429, 0.1639]}
}

# Dimensional Breakdown Data
dim_data = {
    "1D": {
        "auroc": {"mean": [0.5711, 0.5452, 0.5717], "std": [0.1333, 0.1641, 0.1388]},
        "mi":    {"mean": [0.4738, 0.3861, 0.4804], "std": [0.1788, 0.1354, 0.1872]},
        "jsd":   {"mean": [0.3668, 0.3020, 0.3701], "std": [0.1310, 0.0906, 0.1364]}
    },
    "2D": {
        "auroc": {"mean": [0.6987, 0.6924, 0.6974], "std": [0.2044, 0.1900, 0.2025]},
        "mi":    {"mean": [0.3171, 0.2690, 0.3083], "std": [0.2447, 0.2085, 0.2427]},
        "jsd":   {"mean": [0.2619, 0.2264, 0.2531], "std": [0.1521, 0.1373, 0.1498]}
    },
    "3D": {
        "auroc": {"mean": [0.8074, 0.7983, 0.8025], "std": [0.1384, 0.1338, 0.1388]},
        "mi":    {"mean": [0.2991, 0.2780, 0.2875], "std": [0.2665, 0.2492, 0.2630]},
        "jsd":   {"mean": [0.2753, 0.2625, 0.2648], "std": [0.1763, 0.1770, 0.1768]}
    }
}

# -----------------------------------------------------------------------------
# PLOT 1: Grouped Bar Chart of All Metrics
# -----------------------------------------------------------------------------
print("Generating Plot 1: Metric Comparison...")
fig1, ax1 = plt.subplots(figsize=(9, 5.5))

metrics = ["auroc", "spearman", "brier", "mi", "jsd"]
metric_labels = ["AUROC\n(OOD Class.)", "Spearman r\n(Err. Corr.)", "Brier Score\n(Calibration)", "Mutual Info\n(Info. Content)", "JS Divergence\n(Separation)"]

x = np.arange(len(metrics))
width = 0.24

for idx, app in enumerate(approaches):
    means = [all_funcs_data[m]["mean"][idx] for m in metrics]
    stds = [all_funcs_data[m]["std"][idx] for m in metrics]
    
    rects = ax1.bar(x + (idx - 1) * width, means, width, yerr=stds,
                    label=app, color=colors[app], edgecolor="none", alpha=0.9,
                    error_kw=dict(ecolor='#555555', lw=0.8, capsize=3, capthick=0.8))

ax1.set_ylabel('Score / Value')
ax1.set_title('Epistemic Uncertainty Quantification: Performance Across 15 Functions', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(metric_labels)
ax1.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5')
ax1.grid(axis='y', linestyle='-', alpha=0.5)

# Add horizontal line at zero
ax1.axhline(0, color='#333333', linewidth=0.8, zorder=1)

# Annotations for statistical significance
# Standard/Chen are equivalent, both sig > Shaker in MI and JSD
# We label significance bars over MI and JSD
# MI significance brackets
ax1.text(3, 0.65, "**", ha="center", va="bottom", color="#333333", fontsize=10)
# JSD significance brackets
ax1.text(4, 0.50, "**", ha="center", va="bottom", color="#333333", fontsize=10)

plt.tight_layout()
fig1_path = "figures/uq_metric_comparison.png"
fig1.savefig(fig1_path, dpi=300)
plt.close(fig1)
print(f"✓ Saved: {fig1_path}")


# -----------------------------------------------------------------------------
# PLOT 2: Dimensionality Trend Line Plot (AUROC & MI & JSD)
# -----------------------------------------------------------------------------
print("Generating Plot 2: Dimensionality Trend...")
fig2, (ax_auroc, ax_mi, ax_jsd) = plt.subplots(1, 3, figsize=(12, 4.2))

dims = ["1D", "2D", "3D"]
x_dims = np.arange(len(dims))

# Setup subplots details
axes_config = [
    (ax_auroc, "auroc", "AUROC (OOD Discrimination)", "Higher is better"),
    (ax_mi, "mi", "Mutual Information (Info. Content)", "Higher is better"),
    (ax_jsd, "jsd", "Jensen-Shannon Divergence", "Higher is better")
]

for ax, metric_key, title, subtitle in axes_config:
    for idx, app in enumerate(approaches):
        means = [dim_data[d][metric_key]["mean"][idx] for d in dims]
        stds = [dim_data[d][metric_key]["std"][idx] for d in dims]
        
        # Plot line
        ax.plot(x_dims, means, marker='o', linewidth=1.5, markersize=5,
                label=app, color=colors[app], alpha=0.85)
        # Plot standard error band
        ax.fill_between(x_dims, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                        color=colors[app], alpha=0.08)
        
    ax.set_title(title, pad=10)
    ax.set_xticks(x_dims)
    ax.set_xticklabels(dims)
    ax.set_xlabel("Dimension")
    ax.grid(linestyle='-', alpha=0.4)
    if metric_key == "auroc":
        ax.set_ylabel("Metric Value")
        ax.axhline(0.5, color='#999999', linestyle='--', linewidth=0.8, label="Random Guess")
        ax.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5', loc="lower right")
    else:
        ax.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5')

plt.suptitle("Impact of Input Dimensionality on Epistemic Uncertainty Quality", y=0.98)
plt.tight_layout()
fig2_path = "figures/dimensionality_trend.png"
fig2.savefig(fig2_path, dpi=300)
plt.close(fig2)
print(f"✓ Saved: {fig2_path}")

print("\n🎉 All visualizations completed successfully!")
