# Aleatoric Uncertainty Estimation Quality Report

This report evaluates the accuracy of aleatoric uncertainty estimations under input-dependent heteroscedastic noise:
$$\sigma_{\text{true}}(x) = 0.05 + 0.25 \cdot \sin^2(x_1)$$

## 1D Function: sin

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.7239 | 0.6967 | 0.4384 | 0.3550 | 0.000510 | 0.018484 | -0.2703 |
| **Shaker** | 0.5796 | 0.4436 | 0.3267 | 0.2056 | 0.001036 | 0.026049 | 0.0863 |

## 2D Function: sin_cos

| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 0.4461 | 0.4616 | 0.3119 | 0.4738 | 0.012556 | 0.101880 | 0.7467 |
| **Shaker** | 0.3339 | 0.4079 | 0.4874 | 0.5426 | 0.004703 | 0.049978 | 0.9439 |

