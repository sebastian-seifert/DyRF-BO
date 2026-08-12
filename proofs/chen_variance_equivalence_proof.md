# Proof of Expectation Equivalence: Normalized Chen Paired Stability vs. Standard Ensemble Variance in Correlated Random Forests

**Author**: James & Sebastian  
**Context**: Bachelor Thesis - Active Learning & Epistemic Uncertainty Quantification in Random Forests (DyRF-BO)  
**Date**: August 12, 2026  

---

## 1. Model Setup & Stochastic Assumptions

Let $\mathcal{H} = \{h_1(x), h_2(x), \dots, h_M(x)\}$ be an ensemble of $M$ decision trees, where $M$ is an even integer ($M = 2K, \, K \in \mathbb{N}$). 

In a Random Forest regressor, each tree $h_i(x)$ is trained on a bootstrap sample $\mathcal{D}_i^*$ drawn from the training dataset $\mathcal{D}_{\text{train}}$ and splits nodes over random feature subsets.

Before training, because every tree is constructed using the exact same randomized algorithm:
1. The tree predictions $h_1(x), \dots, h_M(x)$ for a target input $x$ are **identically distributed random variables**.
2. **Marginal Mean**: $\mathbb{E}[h_i(x)] = \mu(x)$ for all $i = 1, \dots, M$.
3. **Marginal Variance**: $\text{Var}(h_i(x)) = \mathbb{E}[(h_i(x) - \mu(x))^2] = \sigma^2(x)$ for all $i = 1, \dots, M$.
4. **Breiman's Tree Correlation**: Due to overlapping bootstrap samples and correlated feature spaces, distinct trees $i \neq j$ share a non-zero pairwise correlation $\rho(x) = \text{Corr}(h_i(x), h_j(x)) \in [0, 1]$, yielding pairwise covariance:
   $$\text{Cov}(h_i(x), h_j(x)) = \mathbb{E}\left[(h_i(x) - \mu(x))(h_j(x) - \mu(x))\right] = \rho(x) \cdot \sigma^2(x) \quad (\forall i \neq j)$$

---

## 2. Estimator Definitions

### Definition A: Standard Unbiased Sample Variance ($V_{\text{standard}}$)
Standard Random Forest surrogate variance computes the sample variance of the $M$ tree predictions around the empirical ensemble mean $\bar{h}(x) = \frac{1}{M} \sum_{i=1}^M h_i(x)$:
$$V_{\text{standard}}(x) = \frac{1}{M-1} \sum_{i=1}^M \left( h_i(x) - \bar{h}(x) \right)^2$$

### Definition B: Normalized Chen Paired Stability ($V_{\text{chen}}$)
Chen (2025) partitions the $M = 2K$ trees into $K = \frac{M}{2}$ non-overlapping pairs $(h_{2i-1}(x), h_{2i}(x))$. The normalized Chen paired variance estimator is defined as:
$$V_{\text{chen}}(x) = \frac{1}{M} \sum_{i=1}^{M/2} \left( h_{2i-1}(x) - h_{2i}(x) \right)^2$$

---

## 3. Derivation of Expected Value for Chen Paired Stability ($\mathbb{E}[V_{\text{chen}}]$)

Take the expectation of $V_{\text{chen}}(x)$:
$$\mathbb{E}\left[V_{\text{chen}}(x)\right] = \mathbb{E}\left[ \frac{1}{M} \sum_{i=1}^{M/2} \left( h_{2i-1}(x) - h_{2i}(x) \right)^2 \right]$$

By linearity of expectation:
$$\mathbb{E}\left[V_{\text{chen}}(x)\right] = \frac{1}{M} \sum_{i=1}^{M/2} \mathbb{E}\left[ \left( h_{2i-1}(x) - h_{2i}(x) \right)^2 \right]$$

Consider the expected squared difference for a single tree pair $(h_{2i-1}, h_{2i})$:
$$\mathbb{E}\left[ (h_{2i-1} - h_{2i})^2 \right] = \mathbb{E}\left[ \Big( (h_{2i-1} - \mu) - (h_{2i} - \mu) \Big)^2 \right]$$

Expanding the square:
$$\mathbb{E}\left[ (h_{2i-1} - h_{2i})^2 \right] = \mathbb{E}\left[ (h_{2i-1} - \mu)^2 - 2(h_{2i-1} - \mu)(h_{2i} - \mu) + (h_{2i} - \mu)^2 \right]$$

Applying linearity of expectation and substituting $\text{Var}(h_i) = \sigma^2(x)$ and $\text{Cov}(h_i, h_j) = \rho(x)\sigma^2(x)$:
$$\mathbb{E}\left[ (h_{2i-1} - h_{2i})^2 \right] = \sigma^2(x) - 2\rho(x)\sigma^2(x) + \sigma^2(x) = 2\sigma^2(x) \left( 1 - \rho(x) \right)$$

Substituting this result back into the summation over the $\frac{M}{2}$ pairs:
$$\mathbb{E}\left[V_{\text{chen}}(x)\right] = \frac{1}{M} \sum_{i=1}^{M/2} 2\sigma^2(x) \left( 1 - \rho(x) \right) = \frac{1}{M} \cdot \left( \frac{M}{2} \right) \cdot 2\sigma^2(x) \left( 1 - \rho(x) \right)$$

$$\mathbf{\mathbb{E}\left[V_{\text{chen}}(x)\right] = \sigma^2(x) \left( 1 - \rho(x) \right)}$$

---

## 4. Derivation of Expected Value for Standard Sample Variance ($\mathbb{E}[V_{\text{standard}}]$)

To compute $\mathbb{E}\left[V_{\text{standard}}(x)\right] = \mathbb{E}\left[ \frac{1}{M-1} \sum_{i=1}^M (h_i(x) - \bar{h}(x))^2 \right]$, we use **Steiner's Translation Theorem** (Sum of Squares Decomposition):

### Step 4.1: Sum of Squares Algebraic Expansion
For any reference constant $\mu$:
$$(h_i - \bar{h}) = (h_i - \mu) - (\bar{h} - \mu)$$

Squaring and summing over all $M$ trees:
$$\sum_{i=1}^M (h_i - \bar{h})^2 = \sum_{i=1}^M \left[ (h_i - \mu) - (\bar{h} - \mu) \right]^2$$
$$\sum_{i=1}^M (h_i - \bar{h})^2 = \sum_{i=1}^M (h_i - \mu)^2 - 2(\bar{h} - \mu) \sum_{i=1}^M (h_i - \mu) + \sum_{i=1}^M (\bar{h} - \mu)^2$$

Since $\sum_{i=1}^M (h_i - \mu) = M(\bar{h} - \mu)$:
$$\sum_{i=1}^M (h_i - \bar{h})^2 = \sum_{i=1}^M (h_i - \mu)^2 - 2M(\bar{h} - \mu)^2 + M(\bar{h} - \mu)^2$$
$$\sum_{i=1}^M (h_i - \bar{h})^2 = \sum_{i=1}^M (h_i - \mu)^2 - M(\bar{h} - \mu)^2$$

### Step 4.2: Expectation of Ensemble Mean Variance $\text{Var}(\bar{h})$
Compute $\text{Var}(\bar{h}(x)) = \mathbb{E}[(\bar{h}(x) - \mu(x))^2]$ under pairwise tree correlation $\rho(x)$:
$$\text{Var}(\bar{h}) = \text{Var}\left( \frac{1}{M} \sum_{i=1}^M h_i \right) = \frac{1}{M^2} \left[ \sum_{i=1}^M \text{Var}(h_i) + \sum_{i \neq j} \text{Cov}(h_i, h_j) \right]$$

Since there are $M$ variance terms and $M(M-1)$ pairwise covariance terms:
$$\text{Var}(\bar{h}) = \frac{1}{M^2} \left[ M\sigma^2(x) + M(M-1)\rho(x)\sigma^2(x) \right] = \frac{\sigma^2(x)}{M} + \frac{M-1}{M} \rho(x)\sigma^2(x)$$

### Step 4.3: Substituting $\text{Var}(\bar{h})$ into $\mathbb{E}\left[\sum (h_i - \bar{h})^2\right]$
Take the expectation of the expanded sum of squares:
$$\mathbb{E}\left[ \sum_{i=1}^M (h_i - \bar{h})^2 \right] = \mathbb{E}\left[ \sum_{i=1}^M (h_i - \mu)^2 \right] - M \cdot \text{Var}(\bar{h})$$
$$\mathbb{E}\left[ \sum_{i=1}^M (h_i - \bar{h})^2 \right] = M\sigma^2(x) - M \left( \frac{\sigma^2(x)}{M} + \frac{M-1}{M} \rho(x)\sigma^2(x) \right)$$
$$\mathbb{E}\left[ \sum_{i=1}^M (h_i - \bar{h})^2 \right] = M\sigma^2(x) - \sigma^2(x) - (M-1)\rho(x)\sigma^2(x) = (M-1)\sigma^2(x)\left( 1 - \rho(x) \right)$$

Dividing by $(M-1)$ for $V_{\text{standard}}$:
$$\mathbf{\mathbb{E}\left[V_{\text{standard}}(x)\right] = \frac{1}{M-1} \cdot (M-1)\sigma^2(x)\left( 1 - \rho(x) \right) = \sigma^2(x)\left( 1 - \rho(x) \right)}$$

---

## 5. Main Theorem Conclusion

$$\mathbf{\mathbb{E}\left[V_{\text{standard}}(x)\right] = \mathbb{E}\left[V_{\text{chen}}(x)\right] = \sigma^2(x)\left( 1 - \rho(x) \right)} \quad \blacksquare$$

### Theoretical & Computational Implications

1. **Exact Unbiasedness**: Both Standard Sample Variance and Normalized Chen Paired Stability estimate the exact same expected within-ensemble variance quantity $\sigma^2(x)(1 - \rho(x))$.
2. **Computational Speedup**: 
   * Standard sample variance requires **2 passes** over all $M$ trees (Pass 1: compute mean $\bar{h}$, Pass 2: compute squared differences $(h_i - \bar{h})^2$).
   * Chen Paired Stability requires **1 pass** over $M/2$ pairs, executing with **$50\%$ fewer arithmetic operations** and zero memory overhead for tracking the ensemble mean.
## 6. Derivation of Estimator Variances ($\text{Var}(\hat{V})$)

To evaluate the precision and convergence speed of both estimators, we derive their exact variances $\text{Var}(\hat{V}_{\text{chen}})$ and $\text{Var}(\hat{V}_{\text{standard}})$ under a Gaussian tree model $h_i(x) \sim \mathcal{N}(\mu(x), \sigma^2(x))$.

Let $\tilde{\sigma}^2 = \sigma^2(x)(1 - \rho(x))$ denote the within-ensemble residual variance.

---

### 6.1 Variance of Normalized Chen Paired Stability ($\text{Var}(\hat{V}_{\text{chen}})$)

#### Step 6.1.1: Distribution of a Single Pair Difference
Consider pair $i$: $D_i = h_{2i-1}(x) - h_{2i}(x)$.
Since $h_{2i-1}$ and $h_{2i}$ are jointly normal with variance $\sigma^2$ and covariance $\rho\sigma^2$:
- $\mathbb{E}[D_i] = 0$
- $\text{Var}(D_i) = \sigma^2 + \sigma^2 - 2\rho\sigma^2 = 2\sigma^2(1 - \rho) = 2\tilde{\sigma}^2$

Therefore, $D_i \sim \mathcal{N}(0, 2\tilde{\sigma}^2)$, which implies the normalized square follows a Chi-Square distribution with 1 degree of freedom:
$$\frac{D_i^2}{2\tilde{\sigma}^2} \sim \chi^2_1$$

#### Step 6.1.2: Variance of $D_i^2$
The variance of a $\chi^2_1$ distribution is $2$. Using $\text{Var}(aZ) = a^2 \text{Var}(Z)$:
$$\text{Var}(D_i^2) = (2\tilde{\sigma}^2)^2 \cdot \text{Var}\left( \frac{D_i^2}{2\tilde{\sigma}^2} \right) = 4\tilde{\sigma}^4 \cdot 2 = 8\tilde{\sigma}^4$$

#### Step 6.1.3: Variance of the Sum of $K = M/2$ Independent Pairs
Because the pairs $(h_1, h_2), (h_3, h_4), \dots, (h_{M-1}, h_M)$ involve non-overlapping trees, the squared differences $D_1^2, D_2^2, \dots, D_{M/2}^2$ are independent random variables. Summing their variances:
$$\text{Var}\left( \sum_{i=1}^{M/2} D_i^2 \right) = \sum_{i=1}^{M/2} \text{Var}(D_i^2) = \frac{M}{2} \cdot 8\tilde{\sigma}^4 = 4M\tilde{\sigma}^4$$

#### Step 6.1.4: Scaling by $\frac{1}{M}$
Recall $V_{\text{chen}} = \frac{1}{M} \sum_{i=1}^{M/2} D_i^2$. Using $\text{Var}(c Z) = c^2 \text{Var}(Z)$:
$$\text{Var}(\hat{V}_{\text{chen}}) = \frac{1}{M^2} \text{Var}\left( \sum_{i=1}^{M/2} D_i^2 \right) = \frac{1}{M^2} \cdot 4M\tilde{\sigma}^4 = \frac{4\tilde{\sigma}^4}{M}$$

Substituting $\tilde{\sigma}^2 = \sigma^2(1 - \rho)$:
$$\mathbf{\text{Var}(\hat{V}_{\text{chen}}) = \frac{4\sigma^4(x)(1 - \rho(x))^2}{M}}$$

---

### 6.2 Variance of Standard Sample Variance ($\text{Var}(\hat{V}_{\text{standard}})$)

For $M$ jointly normal variables $h_1, \dots, h_M$ with residual variance $\tilde{\sigma}^2 = \sigma^2(1 - \rho)$, the standard sample variance $V_{\text{standard}} = \frac{1}{M-1} \sum_{i=1}^M (h_i - \bar{h})^2$ follows a scaled Chi-Square distribution with $M-1$ degrees of freedom:

$$\frac{(M-1) V_{\text{standard}}}{\tilde{\sigma}^2} \sim \chi^2_{M-1}$$

The variance of a $\chi^2_{M-1}$ distribution is $2(M-1)$:
$$\text{Var}\left( \frac{(M-1) V_{\text{standard}}}{\tilde{\sigma}^2} \right) = 2(M-1)$$

Pulling out the constants:
$$\frac{(M-1)^2}{\tilde{\sigma}^4} \text{Var}(\hat{V}_{\text{standard}}) = 2(M-1) \implies \text{Var}(\hat{V}_{\text{standard}}) = \frac{2\tilde{\sigma}^4}{M-1}$$

Substituting $\tilde{\sigma}^2 = \sigma^2(1 - \rho)$:
$$\mathbf{\text{Var}(\hat{V}_{\text{standard}}) = \frac{2\sigma^4(x)(1 - \rho(x))^2}{M - 1}}$$

---

### 6.3 Estimator Variance Ratio

Comparing the two estimator variances:

$$\frac{\text{Var}(\hat{V}_{\text{chen}})}{\text{Var}(\hat{V}_{\text{standard}})} = \frac{\frac{4\tilde{\sigma}^4}{M}}{\frac{2\tilde{\sigma}^4}{M-1}} = \frac{2(M-1)}{M} = 2 - \frac{2}{M}$$

As forest size $M \to \infty$:
$$\lim_{M \to \infty} \frac{\text{Var}(\hat{V}_{\text{chen}})}{\text{Var}(\hat{V}_{\text{standard}})} = \mathbf{2}$$

---

## 7. Final Summary & Trade-Off Matrix

| Metric / Property | Standard Sample Variance ($V_{\text{standard}}$) | Normalized Chen Paired Stability ($V_{\text{chen}}$) |
| :--- | :---: | :---: |
| **Expected Value $\mathbb{E}[\hat{V}]$** | $\sigma^2(x)(1 - \rho(x))$ | $\sigma^2(x)(1 - \rho(x))$ |
| **Unbiasedness** | Unbiased | Unbiased |
| **Estimator Variance $\text{Var}(\hat{V})$** | $\mathbf{\frac{2\sigma^4(1-\rho)^2}{M - 1}}$ | $\mathbf{\frac{4\sigma^4(1-\rho)^2}{M}}$ |
| **Asymptotic Efficiency Ratio** | $1.0$ (Baseline Minimum) | $\approx 2.0$ ($2\times$ higher variance) |
| **Passes over Trees** | 2 Passes (Requires mean $\bar{h}$) | **1 Pass (No mean required)** |
| **Arithmetic Operations** | $O(M)$ | **$O(M/2)$ ($50\%$ fewer operations)** |

