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
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8.5

# Create figures directory if not exists
os.makedirs("figures", exist_ok=True)

# -----------------------------------------------------------------------------
# Data from Cluster Run (June 12, 2026 Run with 4 Credal Variants)
# -----------------------------------------------------------------------------
approaches = [
    "Standard", 
    "Chen", 
    "Credal_GL_Bisect", 
    "Credal_GL_Newton", 
    "Credal_Trapz_Bisect", 
    "Credal_Trapz_Newton"
]

colors = {
    "Standard": "#2b5c8f",           # Deep Steel Blue
    "Chen": "#3a9d5d",               # Emerald Sage
    "Credal_GL_Bisect": "#7d3c98",   # Amethyst Purple
    "Credal_GL_Newton": "#a569bd",   # Lavender Purple
    "Credal_Trapz_Bisect": "#b85a25", # Deep Terracotta
    "Credal_Trapz_Newton": "#e59866"  # Soft Apricot
}

# All Functions Summary
all_funcs_data = {
    "auroc": {
        "mean": [0.7125, 0.6993, 0.6695, 0.6766, 0.6692, 0.6786], 
        "std":  [0.1775, 0.1833, 0.1717, 0.1752, 0.1717, 0.1728]
    },
    "spearman": {
        "mean": [0.0870, 0.0857, 0.0977, 0.0535, 0.0900, 0.0652], 
        "std":  [0.3821, 0.3711, 0.3853, 0.3602, 0.3852, 0.3619]
    },
    "brier": {
        "mean": [0.4426, 0.4423, 0.4418, 0.4435, 0.4418, 0.4435], 
        "std":  [0.0736, 0.0737, 0.0759, 0.0734, 0.0758, 0.0734]
    },
    "mi": {
        "mean": [0.3324, 0.3253, 0.3096, 0.3154, 0.3113, 0.3154], 
        "std":  [0.2163, 0.2217, 0.2210, 0.2191, 0.2234, 0.2111]
    },
    "jsd": {
        "mean": [0.4518, 0.4413, 0.4370, 0.4350, 0.4392, 0.4372], 
        "std":  [0.2232, 0.2264, 0.2314, 0.2247, 0.2328, 0.2169]
    }
}

# Dimensional Breakdown Data
dim_data = {
    "1D": {
        "auroc":    {"mean": [0.5702, 0.5619, 0.6490, 0.6058, 0.6474, 0.6143], "std": [0.1339, 0.1420, 0.2002, 0.1729, 0.1993, 0.1676]},
        "spearman": {"mean": [-0.1316, -0.1200, -0.1155, -0.1585, -0.1418, -0.1267], "std": [0.4391, 0.4365, 0.4502, 0.4246, 0.4377, 0.4356]},
        "brier":    {"mean": [0.3175, 0.3174, 0.3128, 0.3184, 0.3130, 0.3183], "std": [0.0251, 0.0252, 0.0291, 0.0258, 0.0287, 0.0258]},
        "mi":       {"mean": [0.4718, 0.4803, 0.6231, 0.5720, 0.6281, 0.5617], "std": [0.1810, 0.1956, 0.1561, 0.1953, 0.1606, 0.1767]},
        "jsd":      {"mean": [0.5263, 0.5310, 0.6877, 0.6309, 0.6918, 0.6232], "std": [0.1928, 0.2054, 0.1480, 0.1992, 0.1545, 0.1805]}
    },
    "2D": {
        "auroc":    {"mean": [0.6977, 0.6955, 0.6627, 0.6727, 0.6627, 0.6732], "std": [0.1929, 0.1922, 0.1651, 0.1784, 0.1664, 0.1789]},
        "spearman": {"mean": [0.1894, 0.1860, 0.1868, 0.1566, 0.1889, 0.1704], "std": [0.3305, 0.3293, 0.3189, 0.2696, 0.3195, 0.2794]},
        "brier":    {"mean": [0.4528, 0.4520, 0.4536, 0.4552, 0.4536, 0.4552], "std": [0.0180, 0.0189, 0.0160, 0.0156, 0.0160, 0.0156]},
        "mi":       {"mean": [0.3109, 0.3059, 0.2215, 0.2460, 0.2248, 0.2542], "std": [0.2439, 0.2411, 0.1211, 0.1564, 0.1228, 0.1539]},
        "jsd":      {"mean": [0.3668, 0.3610, 0.3156, 0.3221, 0.3209, 0.3333], "std": [0.2198, 0.2161, 0.1165, 0.1376, 0.1193, 0.1360]}
    },
    "3D": {
        "auroc":    {"mean": [0.8077, 0.8022, 0.6887, 0.7240, 0.6886, 0.7241], "std": [0.1391, 0.1419, 0.1583, 0.1672, 0.1583, 0.1673]},
        "spearman": {"mean": [0.0004, 0.0040, -0.0007, -0.0429, -0.0004, -0.0428], "std": [0.1040, 0.1000, 0.1035, 0.1711, 0.1042, 0.1717]},
        "brier":    {"mean": [0.4895, 0.4896, 0.4902, 0.4903, 0.4902, 0.4903], "std": [0.0047, 0.0047, 0.0047, 0.0047, 0.0047, 0.0047]},
        "mi":       {"mean": [0.2983, 0.2880, 0.1801, 0.2226, 0.1809, 0.2232], "std": [0.2674, 0.2662, 0.1625, 0.2097, 0.1627, 0.2108]},
        "jsd":      {"mean": [0.3965, 0.3832, 0.3076, 0.3359, 0.3087, 0.3370], "std": [0.2542, 0.2582, 0.2310, 0.2357, 0.2306, 0.2376]}
    },
    "4D": {
        "auroc":    {"mean": [0.7993, 0.7862, 0.6793, 0.7159, 0.6798, 0.7157], "std": [0.0562, 0.0648, 0.0932, 0.0726, 0.0924, 0.0726]},
        "spearman": {"mean": [0.4246, 0.3975, 0.4682, 0.3953, 0.4622, 0.3893], "std": [0.3622, 0.3399, 0.3514, 0.2885, 0.3530, 0.2936]},
        "brier":    {"mean": [0.4984, 0.4984, 0.4984, 0.4984, 0.4984, 0.4984], "std": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]},
        "mi":       {"mean": [0.2311, 0.2096, 0.1682, 0.1785, 0.1656, 0.1822], "std": [0.0546, 0.0550, 0.0734, 0.0623, 0.0707, 0.0616]},
        "jsd":      {"mean": [0.5614, 0.5224, 0.4371, 0.4617, 0.4329, 0.4678], "std": [0.1014, 0.1095, 0.1260, 0.1017, 0.1228, 0.0980]}
    },
    "5D": {
        "auroc":    {"mean": [0.7289, 0.6763, 0.6735, 0.6831, 0.6735, 0.6821], "std": [0.1905, 0.2123, 0.2052, 0.2151, 0.2051, 0.2128]},
        "spearman": {"mean": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], "std": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]},
        "brier":    {"mean": [0.4999, 0.4999, 0.4999, 0.4999, 0.4999, 0.4999], "std": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]},
        "mi":       {"mean": [0.2941, 0.2774, 0.2915, 0.2948, 0.2908, 0.2940], "std": [0.0696, 0.0611, 0.0500, 0.0568, 0.0480, 0.0546]},
        "jsd":      {"mean": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], "std": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}
    }
}

# -----------------------------------------------------------------------------
# PLOT 1: Grouped Bar Chart of All Metrics
# -----------------------------------------------------------------------------
print("Generating Plot 1: Metric Comparison...")
fig1, ax1 = plt.subplots(figsize=(10, 6))

metrics = ["auroc", "spearman", "brier", "mi", "jsd"]
metric_labels = [
    "AUROC\n(OOD Class.)", 
    "Spearman r\n(Err. Corr.)", 
    "Brier Score\n(Calibration)", 
    "Mutual Info\n(Info. Content)", 
    "JS Divergence\n(Separation)"
]

x = np.arange(len(metrics))
width = 0.13  # Thinner bars to fit 6 approaches

for idx, app in enumerate(approaches):
    means = [all_funcs_data[m]["mean"][idx] for m in metrics]
    stds = [all_funcs_data[m]["std"][idx] for m in metrics]
    
    # Calculate offset
    offset = (idx - 2.5) * width
    
    rects = ax1.bar(x + offset, means, width, yerr=stds,
                    label=app.replace("_", " "), color=colors[app], edgecolor="none", alpha=0.9,
                    error_kw=dict(ecolor='#555555', lw=0.7, capsize=2, capthick=0.7))

ax1.set_ylabel('Score / Value')
ax1.set_title('Uncertainty Quantification Benchmark: Comparison of 6 Approaches (10 Seeds)', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(metric_labels)
ax1.legend(frameon=True, facecolor='white', edgecolor='#e5e5e5', loc='upper right')
ax1.grid(axis='y', linestyle='-', alpha=0.5)

# Add horizontal line at zero
ax1.axhline(0, color='#333333', linewidth=0.8, zorder=1)

plt.tight_layout()
fig1_path = "figures/variants_metric_comparison.png"
fig1.savefig(fig1_path, dpi=300)
plt.close(fig1)
print(f"✓ Saved: {fig1_path}")


# -----------------------------------------------------------------------------
# PLOT 2: Dimensionality Trend Line Plot (All 5 Metrics)
# -----------------------------------------------------------------------------
print("Generating Plot 2: Dimensionality Trend...")
fig2, axes = plt.subplots(1, 5, figsize=(20, 5))

dims = ["1D", "2D", "3D", "4D", "5D"]
x_dims = np.arange(len(dims))

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
        ax.plot(x_dims[valid], means_arr[valid], marker='o', linewidth=1.2, markersize=4,
                label=app.replace("_", " "), color=colors[app], alpha=0.85)
        # Plot standard error band
        ax.fill_between(x_dims[valid], means_arr[valid] - stds_arr[valid], means_arr[valid] + stds_arr[valid],
                        color=colors[app], alpha=0.05)
        
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

plt.suptitle("Impact of Input Dimensionality on UQ Quality (All 6 Approaches Compared)", y=0.98, fontsize=15)
plt.tight_layout()
fig2_path = "figures/variants_dimensionality_trend.png"
fig2.savefig(fig2_path, dpi=300)
plt.close(fig2)
print(f"✓ Saved: {fig2_path}")

print("\n🎉 All visualizations completed successfully!")
