#!/usr/bin/env python3
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
# Data from statistical_tests.txt (June 9, 2026 Run)
# -----------------------------------------------------------------------------
approaches = ["Standard", "Chen", "Credal"]
colors = {
    "Standard": "#3498db",  # Steel Blue
    "Chen": "#2ecc71",      # Emerald Green
    "Credal": "#9b59b6"     # Amethyst Purple
}

# All Functions Summary
all_funcs_data = {
    "auroc":    {"mean": [0.7137, 0.7081, 0.6040], "std": [0.1810, 0.1835, 0.1738]},
    "spearman": {"mean": [0.0731, 0.0729, 0.1316], "std": [0.3799, 0.3757, 0.4359]},
    "brier":    {"mean": [0.4424, 0.4423, 0.4421], "std": [0.0737, 0.0737, 0.0768]},
    "mi":       {"mean": [0.3356, 0.3286, 0.2792], "std": [0.2164, 0.2193, 0.2443]},
    "jsd":      {"mean": [0.4556, 0.4440, 0.3971], "std": [0.2216, 0.2236, 0.2576]}
}

# Dimensional Breakdown Data
dim_data = {
    "1D": {
        "auroc":    {"mean": [0.5711, 0.5717, 0.6430], "std": [0.1333, 0.1388, 0.2477]},
        "spearman": {"mean": [-0.1661, -0.1634, -0.0769], "std": [0.4335, 0.4361, 0.5117]},
        "brier":    {"mean": [0.3174, 0.3173, 0.3116], "std": [0.0254, 0.0255, 0.0320]},
        "mi":       {"mean": [0.4738, 0.4804, 0.6755], "std": [0.1788, 0.1872, 0.1242]},
        "jsd":      {"mean": [0.5292, 0.5340, 0.7415], "std": [0.1889, 0.1968, 0.1114]}
    },
    "2D": {
        "auroc":    {"mean": [0.6987, 0.6974, 0.5677], "std": [0.2044, 0.2025, 0.1268]},
        "spearman": {"mean": [0.2015, 0.1982, 0.1273], "std": [0.3200, 0.3165, 0.3467]},
        "brier":    {"mean": [0.4521, 0.4520, 0.4558], "std": [0.0185, 0.0186, 0.0157]},
        "mi":       {"mean": [0.3171, 0.3083, 0.1536], "std": [0.2447, 0.2427, 0.0721]},
        "jsd":      {"mean": [0.3778, 0.3652, 0.2595], "std": [0.2194, 0.2162, 0.1173]}
    },
    "3D": {
        "auroc":    {"mean": [0.8074, 0.8025, 0.5666], "std": [0.1384, 0.1388, 0.1028]},
        "spearman": {"mean": [-0.0035, -0.0003, 0.1472], "std": [0.0888, 0.0849, 0.2859]},
        "brier":    {"mean": [0.4895, 0.4895, 0.4905], "std": [0.0047, 0.0047, 0.0048]},
        "mi":       {"mean": [0.2991, 0.2875, 0.1012], "std": [0.2665, 0.2630, 0.1015]},
        "jsd":      {"mean": [0.3972, 0.3820, 0.2235], "std": [0.2544, 0.2551, 0.2039]}
    },
    "4D": {
        "auroc":    {"mean": [0.7998, 0.7935, 0.6339], "std": [0.0634, 0.0657, 0.1229]},
        "spearman": {"mean": [0.3856, 0.3798, 0.4600], "std": [0.3722, 0.3587, 0.4317]},
        "brier":    {"mean": [0.4984, 0.4984, 0.4984], "std": [0.0000, 0.0000, 0.0000]},
        "mi":       {"mean": [0.2335, 0.2162, 0.1275], "std": [0.0636, 0.0597, 0.0243]},
        "jsd":      {"mean": [0.5601, 0.5289, 0.3421], "std": [0.1069, 0.1075, 0.0511]}
    },
    "5D": {
        "auroc":    {"mean": [0.7342, 0.7104, 0.6320], "std": [0.1934, 0.2130, 0.2022]},
        "spearman": {"mean": [np.nan, np.nan, np.nan], "std": [np.nan, np.nan, np.nan]},
        "brier":    {"mean": [0.4999, 0.4999, 0.4999], "std": [0.0000, 0.0000, 0.0000]},
        "mi":       {"mean": [0.2991, 0.2904, 0.2761], "std": [0.0761, 0.0736, 0.0419]},
        "jsd":      {"mean": [np.nan, np.nan, np.nan], "std": [np.nan, np.nan, np.nan]}
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
ax1.set_title('Epistemic Uncertainty Quantification: Performance Across 20 Functions', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(metric_labels)
ax1.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5')
ax1.grid(axis='y', linestyle='-', alpha=0.5)

# Add horizontal line at zero
ax1.axhline(0, color='#333333', linewidth=0.8, zorder=1)

plt.tight_layout()
fig1_path = "figures/uq_metric_comparison.png"
fig1.savefig(fig1_path, dpi=300)
plt.close(fig1)
print(f"✓ Saved: {fig1_path}")


# -----------------------------------------------------------------------------
# PLOT 2: Dimensionality Trend Line Plot (All 5 Metrics)
# -----------------------------------------------------------------------------
print("Generating Plot 2: Dimensionality Trend...")
fig2, axes = plt.subplots(1, 5, figsize=(18, 4.5))

dims = ["1D", "2D", "3D", "4D", "5D"]
x_dims = np.arange(len(dims))

# Setup subplots details
axes_config = [
    (axes[0], "auroc", "AUROC (OOD Discrimination)", "Higher is better"),
    (axes[1], "spearman", "Spearman r (Error Correlation)", "Higher is better"),
    (axes[2], "brier", "Brier Score (Calibration)", "Lower is better"),
    (axes[3], "mi", "Mutual Information (Info. Content)", "Higher is better"),
    (axes[4], "jsd", "Jensen-Shannon Divergence", "Higher is better")
]

for ax, metric_key, title, subtitle in axes_config:
    for idx, app in enumerate(approaches):
        means = [dim_data[d][metric_key]["mean"][idx] for d in dims]
        stds = [dim_data[d][metric_key]["std"][idx] for d in dims]
        
        # Convert to numpy array to handle nans gracefully
        means_arr = np.array(means, dtype=float)
        stds_arr = np.array(stds, dtype=float)
        valid = ~np.isnan(means_arr)
        
        # Plot line
        ax.plot(x_dims[valid], means_arr[valid], marker='o', linewidth=1.5, markersize=5,
                label=app, color=colors[app], alpha=0.85)
        # Plot standard error band
        ax.fill_between(x_dims[valid], means_arr[valid] - stds_arr[valid], means_arr[valid] + stds_arr[valid],
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
    elif metric_key == "brier":
        ax.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5', loc="upper right")
    else:
        ax.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5')

plt.suptitle("Impact of Input Dimensionality on Epistemic Uncertainty Quality (All Metrics)", y=0.98)
plt.tight_layout()
fig2_path = "figures/dimensionality_trend.png"
fig2.savefig(fig2_path, dpi=300)
plt.close(fig2)
print(f"✓ Saved: {fig2_path}")

print("\n🎉 All visualizations completed successfully!")
