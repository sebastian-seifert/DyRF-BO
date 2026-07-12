# Aleatoric Uncertainty Estimation Quality Report

This report evaluates the accuracy of aleatoric uncertainty estimations under input-dependent heteroscedastic noise:
$$\sigma_{\text{true}}(x) = 0.05 + 0.25 \cdot \sin^2(x_1)$$

## 1D Function: sin

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.8686 | 0.9397 | 0.3702 | 0.5186 | 0.000318 | 0.011581 | -0.3447 |
| **Shaker** | 0.8572 | 0.9330 | 0.3612 | 0.5146 | 0.001132 | 0.025258 | 0.4601 |

## 2D Function: sin_cos

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.6978 | 0.7727 | 0.2906 | 0.4001 | 0.000537 | 0.017474 | -0.2318 |
| **Shaker** | 0.5844 | 0.6790 | 0.2299 | 0.3485 | 0.001038 | 0.022938 | 0.1167 |

## 3D Function: sin_cos_sin

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.5748 | 0.5882 | 0.4306 | 0.5035 | 0.003079 | 0.046258 | 0.2715 |
| **Shaker** | 0.5947 | 0.6184 | 0.5113 | 0.5449 | 0.000986 | 0.024639 | 0.3630 |

## 4D Function: sin_cos_4d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.4972 | 0.5313 | 0.3662 | 0.4446 | 0.001175 | 0.027587 | 0.1983 |
| **Shaker** | 0.5653 | 0.6001 | 0.4201 | 0.4844 | 0.000693 | 0.020904 | 0.4425 |

## 5D Function: sin_cos_5d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.5064 | 0.5529 | 0.2914 | 0.3724 | 0.000737 | 0.022912 | 0.0642 |
| **Shaker** | 0.6043 | 0.6355 | 0.3364 | 0.4185 | 0.000944 | 0.023576 | 0.3435 |

## 6D Function: sin_cos_6d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.4067 | 0.4777 | 0.2503 | 0.3244 | 0.000914 | 0.024786 | -0.0117 |
| **Shaker** | 0.5511 | 0.5866 | 0.3180 | 0.3798 | 0.001202 | 0.025900 | 0.2587 |

## 7D Function: sin_cos_7d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.4820 | 0.4999 | 0.2167 | 0.2852 | 0.000991 | 0.024881 | -0.0738 |
| **Shaker** | 0.6134 | 0.6152 | 0.2587 | 0.3393 | 0.001293 | 0.026628 | 0.1950 |

## 8D Function: sin_cos_8d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.4203 | 0.4548 | 0.1791 | 0.2557 | 0.001119 | 0.025892 | -0.0421 |
| **Shaker** | 0.5691 | 0.5790 | 0.2444 | 0.3226 | 0.001376 | 0.027150 | 0.2274 |

## 9D Function: sin_cos_9d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.3341 | 0.3562 | 0.1704 | 0.2235 | 0.001231 | 0.026975 | -0.0549 |
| **Shaker** | 0.4846 | 0.4885 | 0.2322 | 0.2854 | 0.001482 | 0.028216 | 0.2289 |

## 10D Function: sin_cos_10d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.3230 | 0.3423 | 0.1331 | 0.1774 | 0.001270 | 0.027266 | -0.0644 |
| **Shaker** | 0.4633 | 0.4577 | 0.1880 | 0.2415 | 0.001510 | 0.028530 | 0.2161 |

## 11D Function: sin_cos_11d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.2892 | 0.3011 | 0.1269 | 0.1618 | 0.001307 | 0.027344 | -0.1016 |
| **Shaker** | 0.4313 | 0.4239 | 0.1876 | 0.2335 | 0.001523 | 0.028412 | 0.1343 |

## 12D Function: sin_cos_12d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.2650 | 0.2764 | 0.1107 | 0.1431 | 0.001327 | 0.027724 | -0.0503 |
| **Shaker** | 0.3991 | 0.3953 | 0.1668 | 0.2073 | 0.001530 | 0.028725 | 0.1863 |

## 13D Function: sin_cos_13d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.1793 | 0.2160 | 0.0813 | 0.1242 | 0.001380 | 0.028054 | 0.0701 |
| **Shaker** | 0.3329 | 0.3426 | 0.1431 | 0.1875 | 0.001551 | 0.028897 | 0.3267 |

## 14D Function: sin_cos_14d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.1886 | 0.2065 | 0.0847 | 0.0986 | 0.001378 | 0.028174 | 0.0541 |
| **Shaker** | 0.3246 | 0.3198 | 0.1502 | 0.1746 | 0.001560 | 0.029098 | 0.3008 |

## 15D Function: sin_cos_15d

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.2153 | 0.2521 | 0.0950 | 0.1475 | 0.001390 | 0.028161 | 0.0188 |
| **Shaker** | 0.3628 | 0.3655 | 0.1486 | 0.2042 | 0.001551 | 0.029019 | 0.2368 |

