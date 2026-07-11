# Evaluation Report: Manifold OOD Sweep & Aleatoric Uncertainty Calibration

This report presents an extensive analysis of the results from the **Manifold OOD Sweep** (July 10, 2026) and the **Aleatoric Uncertainty Calibration Suite** on dynamic random forests.

---

## 1. Aleatoric Uncertainty Estimation Quality

Aleatoric uncertainty was evaluated under a spatially varying, input-dependent heteroscedastic noise model:
$$\sigma_{\text{true}}(x) = 0.05 + 0.25 \cdot \sin^2(x_1)$$
We compare **Standard RF** (mean of leaf variances across trees) and **Shaker/Credal** (continuous relative likelihood-ratio optimization).

### Empirical Performance Comparison

| Dimension | Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1D** | **Standard** | **0.7239** | **0.6967** | **0.4384** | **0.3550** | **0.000510** | **0.018484** | **-0.2703** |
| **1D** | **Shaker** | 0.5796 | 0.4436 | 0.3267 | 0.2056 | 0.001036 | 0.026049 | 0.0863 |
| **2D** | **Standard** | **0.4461** | **0.4616** | 0.3119 | 0.4738 | 0.012556 | 0.101880 | **0.7467** |
| **2D** | **Shaker** | 0.3339 | 0.4079 | **0.4874** | **0.5426** | **0.004703** | **0.049978** | 0.9439 |

### Key Insights:
1. **Relative Calibration vs. Absolute Calibration**:
   * **Standard RF** displays higher **Pearson and Spearman correlation** with the true variance. This indicates that tree-level variances preserve the relative rank ordering of local noise very well.
   * **Shaker/Credal** (labeled "Shaker") achieves significantly lower **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** against the true variance in $2\text{D}$ (reducing MSE by **$62.5\%$**). Standard RF tree variances suffer from in-sample overoptimistic bias (underestimating noise in clean leaves), whereas Shaker's likelihood-ratio optimization bounds the absolute scale closer to the true physical noise floor.
2. **Likelihood Calibration (NLL)**:
   * **Standard RF** achieves a lower (better) Gaussian NLL due to its superior mean predictions and tighter local variance bounds in $1\text{D}$ and $2\text{D}$. However, in higher dimensions where leaf samples thin out, Standard RF is expected to overfit, leaving Shaker/Credal as the more stable choice.

---

## 2. Manifold OOD Sweep Results

Unlike traditional hypercube boundary OOD setups, the In-Distribution (ID) points lie on a $(D-1)$-dimensional curved manifold embedded in $\mathbb{R}^D$, and OOD test points are translated orthogonally off the manifold surface along unit normal vectors:
$$x_{\text{OOD}} = \phi(z) + \lambda \cdot v(z)$$

The results of the sweep across all 8 metrics are summarized below:

### Metric: AUROC (Higher is Better)
*Quantifies the classification quality separating points on the manifold (ID) from points translated off (OOD).*

| Dimension | Best Approach | Value (Mean ± Std) | Best Configuration |
| :--- | :--- | :--- | :--- |
| **1D** | Proximity_Method_B_C | 0.7844 ± 0.1963 | RF=5, K=20, Gap=sparse, M=5, Law=linear |
| **2D** | Standard | 0.8721 ± 0.0924 | RF=1, Gap=sparse, M=50, Law=leaf |
| **3D** | Standard | 0.8833 ± 0.0335 | RF=1, Gap=sparse, M=50, Law=leaf |
| **4D** | Standard | 0.8210 ± 0.0989 | RF=1, Gap=sparse, M=5, Law=linear |
| **5D** | Standard | 0.7071 ± 0.1125 | RF=1, Gap=sparse, M=50, Law=linear |
| **6D** | Standard | 0.8242 ± 0.1100 | RF=1, Gap=sparse, M=5, Law=linear |
| **7D** | Standard | 0.7391 ± 0.0934 | RF=1, Gap=empty |
| **8D** | Shaker_Likelihood_Trapz_Bisect | 0.7895 ± 0.1594 | RF=1, Gap=sparse, M=5, Law=leaf |
| **9D** | Shaker_Likelihood_GL_Bisect | 0.7875 ± 0.0739 | RF=1, Gap=sparse, M=5, Law=linear |
| **10D** | Shaker_Likelihood_Trapz_Bisect | 0.7635 ± 0.1503 | RF=1, Gap=sparse, M=5, Law=linear |

### Metric: AUPR (Higher is Better)

| Dimension | Best Approach | Value (Mean ± Std) | Best Configuration |
| :--- | :--- | :--- | :--- |
| **1D** | Proximity_Method_B_C | 0.6909 ± 0.1847 | RF=5, K=20, Gap=sparse, M=5, Law=linear |
| **2D** | Standard | 0.8201 ± 0.1193 | RF=1, Gap=sparse, M=50, Law=leaf |
| **3D** | Standard | 0.8145 ± 0.0544 | RF=1, Gap=sparse, M=50, Law=leaf |
| **4D** | Standard | 0.7080 ± 0.1263 | RF=1, Gap=sparse, M=5, Law=linear |
| **5D** | Standard | 0.5144 ± 0.1809 | RF=1, Gap=sparse, M=50, Law=linear |
| **6D** | Shaker_GMM_Entropy | 0.6940 ± 0.1566 | RF=1, Gap=sparse, M=5, Law=linear |
| **7D** | Standard | 0.5685 ± 0.1447 | RF=1, Gap=sparse, M=5, Law=leaf |
| **8D** | Shaker_Likelihood_Trapz_Bisect | 0.6344 ± 0.2221 | RF=1, Gap=empty |
| **9D** | Shaker_Likelihood_Trapz_Bisect | 0.6073 ± 0.1146 | RF=1, Gap=sparse, M=5, Law=linear |
| **10D** | Shaker_Likelihood_Trapz_Bisect | 0.6029 ± 0.2029 | RF=1, Gap=sparse, M=5, Law=linear |

### Metric: FPR@95TPR (Lower is Better)

| Dimension | Best Approach | Value (Mean ± Std) | Best Configuration |
| :--- | :--- | :--- | :--- |
| **1D** | Proximity_Method_C | 0.3576 ± 0.2938 | RF=5, K=20, Gap=sparse, M=5, Law=linear |
| **2D** | Standard | 0.4961 ± 0.2376 | RF=5, Gap=sparse, M=5, Law=leaf |
| **3D** | Standard | 0.5486 ± 0.1194 | RF=1, Gap=sparse, M=50, Law=leaf |
| **4D** | Standard | 0.6150 ± 0.2457 | RF=1, Gap=sparse, M=5, Law=linear |
| **5D** | Standard | 0.7394 ± 0.1429 | RF=1, Gap=empty |
| **6D** | Shaker_Likelihood_Trapz_Bisect | 0.5176 ± 0.3276 | RF=1, Gap=sparse, M=5, Law=linear |
| **7D** | Shaker_Likelihood_GL_Bisect | 0.7258 ± 0.1553 | RF=1, Gap=empty |
| **8D** | Shaker_Likelihood_Trapz_Bisect | 0.5187 ± 0.3126 | RF=1, Gap=sparse, M=50, Law=linear |
| **9D** | Shaker_Likelihood_Trapz_Bisect | 0.6447 ± 0.1718 | RF=1, Gap=sparse, M=5, Law=linear |
| **10D** | Shaker_Likelihood_Trapz_Bisect | 0.6230 ± 0.2554 | RF=1, Gap=sparse, M=5, Law=linear |

### Metric: NAURC (Lower is Better)
*Calibration quality of the rejection curves. Lower is closer to oracle performance.*

| Dimension | Best Approach | Value (Mean ± Std) | Best Configuration |
| :--- | :--- | :--- | :--- |
| **1D** | Proximity_Method_C | 0.4218 ± 0.3111 | RF=5, K=20, Gap=sparse, M=5, Law=linear |
| **2D** | Standard | 0.4049 ± 0.1708 | RF=1, Gap=sparse, M=50, Law=leaf |
| **3D** | Standard | 0.3911 ± 0.1793 | RF=1, Gap=sparse, M=50, Law=leaf |
| **4D** | Standard | 0.3553 ± 0.0653 | RF=1, Gap=sparse, M=5, Law=linear |
| **5D** | Standard | 0.5492 ± 0.2183 | RF=1, Gap=empty |
| **6D** | Shaker_Likelihood_Trapz_Bisect | 0.4784 ± 0.0957 | RF=1, Gap=sparse, M=50, Law=linear |
| **7D** | Standard | 0.4080 ± 0.1757 | RF=1, Gap=empty |
| **8D** | Shaker_Likelihood_Trapz_Bisect | 0.7454 ± 0.2670 | RF=1, Gap=sparse, M=5, Law=linear |
| **9D** | Shaker_Likelihood_Trapz_Bisect | 0.7297 ± 0.2695 | RF=1, Gap=sparse, M=5, Law=linear |
| **10D** | Shaker_Likelihood_Trapz_Bisect | 0.6452 ± 0.2496 | RF=1, Gap=sparse, M=5, Law=linear |

---

## 3. Comparative Analysis across Dimensions & Metrics

1. **Low Dimensions ($1\text{D}$)**:
   * **Proximity UQ** variants (such as `Proximity_Method_B_C` and `Proximity_Method_C` locked to neighborhood size $K=20$) dominate in $1\text{D}$.
   * Under a 0D manifold point structure, local density estimation is extremely sharp, allowing proximity methods to accurately score OOD translation offsets.

2. **Mid Dimensions ($2\text{D}$ to $7\text{D}$)**:
   * **Standard RF Disagreement** (epistemic tree variance) is highly dominant.
   * *Geometric Intuition*: The Random Forest splits are axis-aligned, creating a step-wise boundary approximating the curved codimension-1 manifold. In regions translated orthogonally off this manifold, the trees' predictions start to vary drastically, resulting in a spike in tree-level variance (disagreement). This boundary "fracture" makes standard disagreement highly sensitive to manifold shifts in $2\text{D} - 7\text{D}$.

3. **High Dimensions ($8\text{D}$ to $10\text{D}$)**:
   * **Shaker/Credal UQ** (`Shaker_Likelihood_Trapz_Bisect` and `Shaker_Likelihood_GL_Bisect`) completely dominates.
   * *Methodological Intuition*: In high-dimensional manifolds, standard tree disagreement suffers from variance dilution (the sparse tree branches fail to capture structured boundary disagreements cleanly). The relative continuous likelihood-ratio framework effectively isolates the true statistical bounds at the leaves, making it much more robust in high dimensions ($8\text{D}-10\text{D}$) across all metrics.
