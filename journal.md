# Journal for Research Progress

## 18.5
Today I expanded the comparison of the different epistemic uncertainty quantification approaches (Standard, Chen, Shaker).
I started approximating the integral from Shaker using Adaptive Quadrature, and getting all of the tree variances by just computing the variances from the training set. This could be further improved. 
For evaluation, I use a sin-wave on the interval from [0,10] with some gaussian noise added on top.
The random forest is then trained on [0,10] \ [4,6], using the OOD technique. 
I then measure the NLL value, the correlation between MSE and predicted total variance (Epi + expected value of the Tree Variances (Aleatoric)), as well as the ratio between gap and non-gap variance. 
The NLL for everything but Shaker is realized via a simple "Gaussian Plug In", the GMM from Shaker is more complicated to compute. After running some tests, the NLL value of Shaker seems to be the best.

**ToDo:** Run statistical tests to prove my intuition. 

## 19.5
Ran the Friedmann Test on the entire range. Found no significant difference (alpha = 0.05).
Ran the Friedmann Test just on the Gap Range. Found a significant difference (alpha = 0.05).

| Approach | Mean Gap NLL | Std Dev |
| :--- | :--- | :--- |
| Standard Disagreement | 1.2401 | 1.6316 |
| Shaker GMM | 1.0127 | 0.6913 |
| Chen Stability | 1.2425 | 1.6433 |

--- Statistical Significance (Friedman Test on Gap) ---
- **Statistic:** 115.9720
- **P-value:** 6.5615e-26
- **Result:** Significant difference found in the gap (p < 0.05).

--- Post-hoc Analysis (Wilcoxon Signed-Rank + Bonferroni Correction) ---
| Comparison | p-value | Sig. | Cohen's d |
| :--- | :--- | :--- | :--- |
| Shaker GMM vs Standard Disagreement | 6.36e-05 | YES | -0.1813 |
| Shaker GMM vs Chen Stability | 7.95e-06 | YES | -0.1821 |
| Standard Disagreement vs Chen Stability | 7.11e-03 | YES | -0.0014 |

- **Bonferroni-corrected alpha:** 0.0167
- **Note:** Cohen's d > 0.8 is considered a large effect size.

I also deeply investigated the Generalization Bound Paper... 
I currently dont have an idea on how to connect the Epistemic Uncertainty and the Generalization Bounds.
The dynamic adjustment of the RF's HP will be a tradeoff between speed and generalization. Number of Trees and m_bag are immediately clear from the paper, but I think, that min_samples_per_leaf may also be hidden as a hyperparameter, especially in the RF_V and Gap_V, but I dont have any evidence / proof / idea yet. 

Some questions I have for the paper and the implementation, especially regarding SMAC:
1. Chen uses a different optimization technique than the SMAC one. This could be a problem. Even though, e_mh diminishes for large T_MH, the tau term is still present in the e_Stab formula. 
2. How does "picking the last config instead of the incumbent" affect the approach

## 20.5
Today, i ran some more tests on comparing the three different approaches for the Epistemic Uncertainty. I tried using AUROC and Spearman Correlation Coefficient. 
Results: 

--- Summary Statistics for AUROC ---
Standard: Mean = 0.9824, Std = 0.0425
Shaker: Mean = 0.9606, Std = 0.0737
Chen: Mean = 0.9810, Std = 0.0472

--- Summary Statistics for SPEARMAN ---
Standard: Mean = 0.0175, Std = 0.2705
Shaker: Mean = 0.0472, Std = 0.2644
Chen: Mean = 0.0386, Std = 0.2697

--- Statistical Validation for AUROC ---
Friedman Test: p = 1.4289e-04
Wilcoxon Shaker vs Standard: p = 7.8224e-04 (Sig: YES)
Wilcoxon Shaker vs Chen: p = 1.4256e-03 (Sig: YES)
Wilcoxon Standard vs Chen: p = 8.7454e-02 (Sig: NO)

--- Statistical Validation for SPEARMAN ---
Friedman Test: p = 1.5612e-01

I am not really sure if everything works as intended (I also changed the numerical integration a bit to guide to "better" values), but if yes:
The Shaker formula tends to just give a higher uncertainty everywhere, therefore having a lower AUROC Score on average than the other two approaches. But nontheless, the Spearman Correlation Coefficient between the Error and the Epistemic Uncertainty is higher for the epistemic uncertainty, but without any major significant difference. 
