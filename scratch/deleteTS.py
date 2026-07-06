import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data extracted from the WS 25/26 Klausurergebnisse
grades = (
    [1.0] * 2
    + [1.3] * 2
    + [1.7] * 2
    + [2.0] * 5
    + [2.3] * 5
    + [2.7] * 2
    + [3.0] * 10
    + [3.3] * 12
    + [3.7] * 10
    + [4.0] * 11
    + [5.0] * 34
)

# Set up the figure and axes
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Histogram
sns.histplot(
    grades,
    bins=np.arange(0.8, 5.4, 0.2),
    kde=False,
    ax=axes[0],
    color="skyblue",
    edgecolor="black",
)
axes[0].set_title("Histogram of Grades", fontsize=14)
axes[0].set_xlabel("Grade", fontsize=12)
axes[0].set_ylabel("Number of Students", fontsize=12)
axes[0].set_xticks([1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0, 5.0])
axes[0].tick_params(axis="x", rotation=45)

# 2. Boxplot
sns.boxplot(y=grades, ax=axes[1], color="lightgreen", width=0.4)
axes[1].set_title("Boxplot of Grades", fontsize=14)
axes[1].set_ylabel("Grade", fontsize=12)
axes[
    1
].invert_yaxis()  # Inverting so 1.0 (best) is at the top, optional but common for grades

# Display the plots
plt.tight_layout()
plt.show()
