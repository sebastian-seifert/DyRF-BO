# Aleatoric Uncertainty Estimation Quality Report

This report evaluates the accuracy of aleatoric uncertainty estimations under input-dependent heteroscedastic noise:
$$\sigma_{\text{true}}(x) = 0.05 + 0.25 \cdot \sin^2(x_1)$$

## 1D Function: sin

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.8745 | 0.9403 | 0.3627 | 0.5172 | 0.000301 | 0.011212 | -0.3383 |
| **Shaker** | 0.8630 | 0.9344 | 0.3562 | 0.5146 | 0.001108 | 0.024945 | 0.4732 |

## 2D Function: sin_cos

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.6983 | 0.7803 | 0.2814 | 0.3918 | 0.000542 | 0.017467 | -0.2375 |
| **Shaker** | 0.5874 | 0.6888 | 0.2254 | 0.3426 | 0.001038 | 0.022772 | 0.1077 |

## 3D Function: sin_cos_sin

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.5577 | 0.5767 | 0.4256 | 0.5023 | 0.003162 | 0.046850 | 0.2621 |
| **Shaker** | 0.5829 | 0.6079 | 0.5106 | 0.5444 | 0.001024 | 0.025072 | 0.3377 |

## 4D Function: sin_cos_4d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.5493 | 0.5796 | 0.3653 | 0.4615 | 0.001118 | 0.026828 | 0.1835 |
| **Shaker** | 0.6110 | 0.6437 | 0.4190 | 0.4979 | 0.000658 | 0.020289 | 0.4199 |

## 5D Function: sin_cos_5d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.5014 | 0.5470 | 0.2770 | 0.3590 | 0.000744 | 0.023032 | 0.0594 |
| **Shaker** | 0.6053 | 0.6323 | 0.3295 | 0.4055 | 0.000957 | 0.023818 | 0.3358 |

## 6D Function: sin_cos_6d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.4047 | 0.4775 | 0.2520 | 0.3244 | 0.000934 | 0.024998 | 0.0026 |
| **Shaker** | 0.5466 | 0.5850 | 0.3205 | 0.3801 | 0.001223 | 0.026071 | 0.2806 |

## 7D Function: sin_cos_7d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.4941 | 0.5251 | 0.2130 | 0.2935 | 0.000993 | 0.024753 | -0.0724 |
| **Shaker** | 0.6321 | 0.6414 | 0.2550 | 0.3471 | 0.001295 | 0.026540 | 0.1984 |

## 8D Function: sin_cos_8d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.4136 | 0.4542 | 0.1836 | 0.2567 | 0.001144 | 0.025975 | -0.0265 |
| **Shaker** | 0.5613 | 0.5768 | 0.2416 | 0.3223 | 0.001402 | 0.027300 | 0.2612 |

## 9D Function: sin_cos_9d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.3475 | 0.3666 | 0.1723 | 0.2202 | 0.001228 | 0.026915 | -0.0933 |
| **Shaker** | 0.5011 | 0.5035 | 0.2395 | 0.2862 | 0.001474 | 0.028113 | 0.1631 |

## 10D Function: sin_cos_10d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.3486 | 0.3726 | 0.1462 | 0.1844 | 0.001258 | 0.027156 | -0.0898 |
| **Shaker** | 0.4962 | 0.4921 | 0.2045 | 0.2472 | 0.001491 | 0.028369 | 0.1709 |

