# Multi-Budget Horizon Ranking & Statistical Analysis

## 1. Cross-Horizon Mean Rank Evolution

| Optimizer | Rank (T=11) | Rank (T=15) | Rank (T=20) | Rank (T=25) | Rank (T=35) | Rank (T=50) | Δ (Early T=11 → Final T=50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | 8.81 | 7.42 | 8.19 | 7.56 | 6.56 | 5.97 | 🟢 -2.83 (Improved) |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | 8.81 | 9.81 | 9.19 | 8.92 | 7.44 | 7.00 | 🟢 -1.81 (Improved) |
| `SMAC20_CustomUncertainty_ei_proximity_b` | 8.81 | 7.83 | 7.25 | 7.25 | 7.14 | 7.17 | 🟢 -1.64 (Improved) |
| `SMAC20_CustomUncertainty_ei_likelihood_credal` | 8.81 | 7.94 | 7.69 | 7.11 | 7.31 | 7.31 | 🟢 -1.50 (Improved) |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | 8.81 | 6.53 | 9.31 | 7.64 | 7.03 | 7.36 | 🟢 -1.44 (Improved) |
| `SMAC20_CustomUncertainty_ei_proximity_auto_lambda` | 8.81 | 5.94 | 8.56 | 6.58 | 7.17 | 7.42 | 🟢 -1.39 (Improved) |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | 7.75 | 6.67 | 6.64 | 6.53 | 6.64 | 7.53 | ⚪ -0.22 (Stable) |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b` | 6.17 | 9.56 | 7.42 | 8.50 | 9.03 | 7.89 | 🔴 +1.72 (Degraded) |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | 6.44 | 8.97 | 8.36 | 9.47 | 8.72 | 7.94 | 🔴 +1.50 (Degraded) |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | 8.81 | 6.64 | 8.14 | 7.17 | 7.33 | 8.42 | 🟢 -0.39 (Improved) |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_disagreement` | 8.17 | 9.78 | 7.83 | 9.11 | 8.69 | 8.78 | 🔴 +0.61 (Degraded) |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | 7.47 | 9.17 | 9.33 | 8.89 | 10.11 | 8.81 | 🔴 +1.33 (Degraded) |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_proximity` | 6.72 | 8.42 | 6.47 | 7.97 | 8.86 | 9.03 | 🔴 +2.31 (Degraded) |
| `SMAC3_HPOFacade_ei` | 8.81 | 6.61 | 7.97 | 8.39 | 8.33 | 9.47 | 🔴 +0.67 (Degraded) |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_auto_lambda` | 6.83 | 8.72 | 7.64 | 8.92 | 9.64 | 9.92 | 🔴 +3.08 (Degraded) |

## 2. Omnibus Friedman & Iman-Davenport Tests per Budget Horizon

| Budget Horizon | Tasks | Optimizers | Friedman $\chi_F^2$ | $p_{\text{Friedman}}$ | Iman-Davenport $F_F$ | $p_{\text{Iman-Davenport}}$ | Critical Difference ($CD$) | Global Significance (α=0.05) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T=11** | 18 | 15 | 19.13 | 1.5993e-01 | 1.40 | 1.5522e-01 | 4.518 | ✗ NO |
| **T=15** | 18 | 15 | 22.13 | 7.6059e-02 | 1.64 | 7.0461e-02 | 4.518 | ✗ NO |
| **T=20** | 18 | 15 | 9.90 | 7.6931e-01 | 0.70 | 7.7815e-01 | 4.518 | ✗ NO |
| **T=25** | 18 | 15 | 12.26 | 5.8560e-01 | 0.87 | 5.9311e-01 | 4.518 | ✗ NO |
| **T=35** | 18 | 15 | 16.99 | 2.5655e-01 | 1.23 | 2.5459e-01 | 4.518 | ✗ NO |
| **T=50** | 18 | 15 | 15.40 | 3.5159e-01 | 1.11 | 3.5277e-01 | 4.518 | ✗ NO |

## 3. Pairwise Holm-Bonferroni Corrected Wilcoxon Tests (vs. `SMAC3_HPOFacade_ei`)

### Horizon $T = 11$ (Iman-Davenport $p = 1.5522e-01$)
| Candidate Optimizer | $r_{\text{rb}}$ (Effect Size) | $W$ Stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_proximity` | -0.66 | 18.0 | 0.0303 | 0.4246 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | -0.59 | 16.0 | 0.0712 | 0.9255 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | -0.32 | 31.0 | 0.3109 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_auto_lambda` | -0.32 | 31.0 | 0.3109 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | -0.41 | 31.0 | 0.1771 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b` | -0.47 | 28.0 | 0.1240 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_disagreement` | -0.36 | 29.0 | 0.2489 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_likelihood_credal` | +0.00 | 0.0 | 1.0000 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_auto_lambda` | +0.00 | 0.0 | 1.0000 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_b` | +0.00 | 0.0 | 1.0000 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | +0.00 | 0.0 | 1.0000 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | +0.00 | 0.0 | 1.0000 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | +0.00 | 0.0 | 1.0000 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | +0.00 | 0.0 | 1.0000 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |

### Horizon $T = 15$ (Iman-Davenport $p = 7.0461e-02$)
| Candidate Optimizer | $r_{\text{rb}}$ (Effect Size) | $W$ Stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | +0.85 | 13.0 | 0.0007 | 0.0094 | ✓ **YES** | *LOSS (Inferior)* ❌ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b` | +0.50 | 43.0 | 0.0665 | 0.8650 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_auto_lambda` | +0.23 | 66.0 | 0.4171 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | +0.11 | 76.0 | 0.7019 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | +0.28 | 55.0 | 0.3088 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_disagreement` | +0.31 | 53.0 | 0.2659 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_proximity` | +0.15 | 65.0 | 0.5862 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | +0.19 | 69.0 | 0.4951 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_likelihood_credal` | +0.28 | 43.0 | 0.3343 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_auto_lambda` | -0.08 | 55.0 | 0.7764 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | +0.01 | 76.0 | 0.9811 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_b` | -0.01 | 67.0 | 0.9588 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | +0.30 | 42.0 | 0.3066 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | +0.23 | 59.0 | 0.4074 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |

### Horizon $T = 20$ (Iman-Davenport $p = 7.7815e-01$)
| Candidate Optimizer | $r_{\text{rb}}$ (Effect Size) | $W$ Stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | -0.20 | 61.0 | 0.4631 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_auto_lambda` | +0.06 | 80.0 | 0.8317 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b` | -0.22 | 53.0 | 0.4380 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | +0.06 | 80.0 | 0.8317 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | +0.11 | 76.0 | 0.7019 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_disagreement` | -0.04 | 82.0 | 0.8986 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_proximity` | -0.06 | 72.0 | 0.8313 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_likelihood_credal` | +0.10 | 54.0 | 0.7333 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_auto_lambda` | -0.03 | 74.0 | 0.9058 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_b` | -0.45 | 29.0 | 0.1401 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | +0.06 | 72.0 | 0.8313 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | +0.31 | 47.0 | 0.2775 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | +0.01 | 76.0 | 0.9811 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | -0.03 | 74.0 | 0.9058 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |

### Horizon $T = 25$ (Iman-Davenport $p = 5.9311e-01$)
| Candidate Optimizer | $r_{\text{rb}}$ (Effect Size) | $W$ Stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | -0.28 | 55.0 | 0.3088 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_auto_lambda` | +0.23 | 59.0 | 0.4074 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b` | -0.02 | 75.0 | 0.9434 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | +0.15 | 58.0 | 0.6051 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | -0.06 | 72.0 | 0.8313 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_disagreement` | +0.03 | 74.0 | 0.9058 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_proximity` | -0.01 | 76.0 | 0.9811 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_likelihood_credal` | -0.07 | 63.0 | 0.7960 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_auto_lambda` | -0.24 | 58.0 | 0.3812 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_b` | -0.12 | 60.0 | 0.6791 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | -0.29 | 54.0 | 0.2868 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | -0.24 | 58.0 | 0.3812 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | -0.08 | 70.0 | 0.7583 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | -0.38 | 53.0 | 0.1674 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |

### Horizon $T = 35$ (Iman-Davenport $p = 2.5459e-01$)
| Candidate Optimizer | $r_{\text{rb}}$ (Effect Size) | $W$ Stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | -0.25 | 57.0 | 0.3560 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_auto_lambda` | +0.03 | 74.0 | 0.9058 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b` | +0.03 | 66.0 | 0.9176 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | -0.10 | 61.0 | 0.7174 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | -0.06 | 72.0 | 0.8313 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_disagreement` | -0.18 | 63.0 | 0.5228 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_proximity` | +0.01 | 76.0 | 0.9811 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_likelihood_credal` | +0.24 | 52.0 | 0.4080 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_auto_lambda` | -0.45 | 42.0 | 0.1024 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_b` | -0.20 | 61.0 | 0.4631 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | -0.20 | 61.0 | 0.4631 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | +0.22 | 53.0 | 0.4380 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | -0.46 | 37.0 | 0.1089 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | -0.25 | 51.0 | 0.3794 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |

### Horizon $T = 50$ (Iman-Davenport $p = 3.5277e-01$)
| Candidate Optimizer | $r_{\text{rb}}$ (Effect Size) | $W$ Stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | -0.79 | 14.0 | 0.0052 | 0.0733 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | -0.58 | 25.0 | 0.0468 | 0.6087 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | -0.13 | 59.0 | 0.6417 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_auto_lambda` | +0.15 | 65.0 | 0.5862 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | -0.03 | 66.0 | 0.9176 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_disagreement` | -0.06 | 72.0 | 0.8313 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b` | +0.06 | 64.0 | 0.8361 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | -0.19 | 55.0 | 0.5014 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_likelihood_credal` | -0.29 | 48.0 | 0.3011 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_standard_proximity` | -0.12 | 60.0 | 0.6791 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_auto_lambda` | -0.35 | 39.0 | 0.2330 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_b` | -0.47 | 32.0 | 0.1118 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | -0.27 | 44.0 | 0.3635 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | -0.07 | 56.0 | 0.8203 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
