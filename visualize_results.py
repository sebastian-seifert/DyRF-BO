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
    "auroc":    {"mean": [0.6924, 0.6905, 0.5925], "std": [0.1886, 0.1882, 0.1750]},
    "spearman": {"mean": [0.0106, 0.0115, 0.0659], "std": [0.3493, 0.3479, 0.4060]},
    "brier":    {"mean": [0.0771, 0.0770, 0.0760], "std": [0.0759, 0.0759, 0.0740]},
    "mi":       {"mean": [0.3633, 0.3587, 0.3101], "std": [0.2459, 0.2487, 0.2784]},
    "jsd":      {"mean": [0.3013, 0.2960, 0.2829], "std": [0.1611, 0.1639, 0.1940]}
}

# Dimensional Breakdown Data
dim_data = {
    "1D": {
        "auroc": {"mean": [0.5711, 0.5717, 0.6430], "std": [0.1333, 0.1388, 0.2477]},
        "mi":    {"mean": [0.4738, 0.4804, 0.6755], "std": [0.1788, 0.1872, 0.1242]},
        "jsd":   {"mean": [0.3668, 0.3701, 0.5139], "std": [0.1310, 0.1364, 0.0772]}
    },
    "2D": {
        "auroc": {"mean": [0.6987, 0.6974, 0.5677], "std": [0.2044, 0.2025, 0.1268]},
        "mi":    {"mean": [0.3171, 0.3083, 0.1536], "std": [0.2447, 0.2427, 0.0721]},
        "jsd":   {"mean": [0.2619, 0.2531, 0.1798], "std": [0.1521, 0.1498, 0.0813]}
    },
    "3D": {
        "auroc": {"mean": [0.8074, 0.8025, 0.5666], "std": [0.1384, 0.1388, 0.1028]},
        "mi":    {"mean": [0.2991, 0.2875, 0.1012], "std": [0.2665, 0.2630, 0.1015]},
        "jsd":   {"mean": [0.2753, 0.2648, 0.1549], "std": [0.1763, 0.1768, 0.1414]}
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
