# Journal for Research Progress

## 18.5
Today i expanded the comparison of the different epistemic uncertainty quantification approaches. (Standard, Chen, Shaker)
I started approximating the integral from Shaker using Adaptive Quadrature, and getting all of the tree variances by just computing the variances from the training set. This could be further improved. 
For evaluation, i use a sin-wave on the interval from [0,10] with some gaussian noise added on top.
The random forest is then trained on [0,10] \ [4,6], using the OOD technique. 
I then measure the NLL value, the correlation between MSE and predicted total variance (Epi + expected value of the Tree Variances (Aleatoric)), as well as the ratio between gap and non-gap variance. 
The NLL for everything but Shaker is realized via a simple "Gaussian Plug In", the GMM from Shaker is more complicated to compute. After running some tests, the NLL value of Shaker seems to be the best.

ToDo: Run statistical tests to prove my intuition. 

## 19.5
Ran the Friedmann Test on the entire range. Found no signifiant difference. (alpha = 0.05)
Ran the Friedmann Test just on the Gap Range. Found a significant difference. (alpha = 0.05)

Approach                  | Mean Gap NLL | Std Dev   
-------------------------------------------------------
Standard Disagreement     | 1.2401       | 1.6316    
Shaker GMM                | 1.0127       | 0.6913    
Chen Stability            | 1.2425       | 1.6433    

--- Statistical Significance (Friedman Test on Gap) ---
Statistic: 115.9720
P-value:   6.5615e-26
Result: Significant difference found in the gap (p < 0.05).

--- Post-hoc Analysis (Wilcoxon Signed-Rank + Bonferroni Correction) ---
Comparison                                    | p-value    | Sig.  | Cohen's d 
--------------------------------------------------------------------------------
Shaker GMM vs Standard Disagreement           | 6.36e-05   | YES   | -0.1813   
Shaker GMM vs Chen Stability                  | 7.95e-06   | YES   | -0.1821   
Standard Disagreement vs Chen Stability       | 7.11e-03   | YES   | -0.0014   

Bonferroni-corrected alpha: 0.0167
Note: Cohen's d > 0.8 is considered a large effect size.

I also deeply investigated the Generalization Bound Paper... 
I currently dont have an idea on how to connect the Epistemic Uncertainty and the Generalization Bounds
The dynamic adjustment of the RF's HP will be a tradeoff between speed and generalization. Number of Trees and m_bag are immediatly clear from the paper, but i think, that min_samples_per_leaf may also be hidden as a hyperparameter, especially in the RF_V and Gap_V, but i dont have any evidence / proof / idea yet. 

Some questions i have for the paper and the implementation, especially regarding SMAC:
1. Chen uses a different optimization technique than the SMAC one. This could be a problem. Even though, e_mh diminishes for large T_MH, the tau term is still present in the e_Stab formula. 
2. How does "picking the last config instead of the incumbent" affect the approach




