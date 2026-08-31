# Multi-Budget Horizon Ranking & Statistical Analysis

## 1. Cross-Horizon Mean Rank Evolution

| Optimizer | Rank (T=11) | Rank (T=15) | Rank (T=20) | Rank (T=25) | Rank (T=35) | Rank (T=50) | Δ (Early T=11 → Final T=50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | 5.25 | 3.44 | 4.25 | 3.25 | 3.69 | 3.75 | 🟢 -1.50 (Improved) |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | 5.44 | 4.19 | 3.88 | 3.81 | 3.75 | 3.75 | 🟢 -1.69 (Improved) |
| `SMAC3_HPOFacade_ei` | 5.31 | 3.75 | 3.94 | 4.31 | 3.69 | 3.75 | 🟢 -1.56 (Improved) |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | 5.19 | 4.19 | 4.06 | 3.88 | 4.69 | 4.25 | 🟢 -0.94 (Improved) |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | 5.38 | 4.62 | 3.62 | 3.81 | 3.81 | 4.81 | 🟢 -0.56 (Improved) |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | 2.84 | 5.56 | 5.88 | 5.62 | 5.19 | 5.06 | 🔴 +2.22 (Degraded) |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | 3.72 | 4.78 | 4.78 | 5.66 | 5.38 | 5.19 | 🔴 +1.47 (Degraded) |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | 2.88 | 5.47 | 5.59 | 5.66 | 5.81 | 5.44 | 🔴 +2.56 (Degraded) |

## 2. Omnibus Friedman & Iman-Davenport Tests per Budget Horizon

| Budget Horizon | Tasks | Optimizers | Friedman $\chi_F^2$ | $p_{\text{Friedman}}$ | Iman-Davenport $F_F$ | $p_{\text{Iman-Davenport}}$ | Critical Difference ($CD$) | Global Significance (α=0.05) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T=11** | 16 | 8 | 32.51 | 3.2658e-05 | 6.13 | 5.0557e-06 | 2.625 | ✓ **YES** |
| **T=15** | 16 | 8 | 10.80 | 1.4736e-01 | 1.60 | 1.4296e-01 | 2.625 | ✗ NO |
| **T=20** | 16 | 8 | 13.06 | 7.0742e-02 | 1.98 | 6.4699e-02 | 2.625 | ✗ NO |
| **T=25** | 16 | 8 | 18.34 | 1.0520e-02 | 2.94 | 7.5363e-03 | 2.625 | ✓ **YES** |
| **T=35** | 16 | 8 | 14.27 | 4.6569e-02 | 2.19 | 4.0816e-02 | 2.625 | ✓ **YES** |
| **T=50** | 16 | 8 | 9.38 | 2.2684e-01 | 1.37 | 2.2558e-01 | 2.625 | ✗ NO |

## 3. Pairwise Holm-Bonferroni Corrected Wilcoxon Tests (vs. `SMAC3_HPOFacade_ei`)

### Horizon $T = 11$ (Iman-Davenport $p = 5.0557e-06$)
| Candidate Optimizer | $r_{\text{rb}}$ (Effect Size) | $W$ Stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | -0.72 | 19.0 | 0.0092 | 0.0643 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | -0.53 | 32.0 | 0.0654 | 0.3924 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | -0.53 | 32.0 | 0.0654 | 0.3924 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | -1.00 | 0.0 | 0.3173 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | +1.00 | 0.0 | 0.3173 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | -1.00 | 0.0 | 0.3173 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | +1.00 | 0.0 | 0.3173 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |

### Horizon $T = 15$ (Iman-Davenport $p = 1.4296e-01$)
| Candidate Optimizer | $r_{\text{rb}}$ (Effect Size) | $W$ Stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | +0.53 | 32.0 | 0.0654 | 0.4578 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | +0.35 | 44.0 | 0.2312 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | +0.06 | 64.0 | 0.8603 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | -0.31 | 47.0 | 0.2979 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | +0.15 | 58.0 | 0.6322 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | +0.07 | 63.0 | 0.8209 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | +0.31 | 47.0 | 0.2979 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |

### Horizon $T = 20$ (Iman-Davenport $p = 6.4699e-02$)
| Candidate Optimizer | $r_{\text{rb}}$ (Effect Size) | $W$ Stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | +0.63 | 25.0 | 0.0250 | 0.1747 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | +0.47 | 36.0 | 0.1046 | 0.6275 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | +0.13 | 59.0 | 0.6685 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | +0.21 | 54.0 | 0.4954 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | -0.06 | 64.0 | 0.8603 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | -0.12 | 60.0 | 0.7057 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | -0.25 | 51.0 | 0.4037 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |

### Horizon $T = 25$ (Iman-Davenport $p = 7.5363e-03$)
| Candidate Optimizer | $r_{\text{rb}}$ (Effect Size) | $W$ Stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | +0.65 | 24.0 | 0.0214 | 0.1497 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | +0.47 | 36.0 | 0.1046 | 0.6275 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | +0.44 | 38.0 | 0.1297 | 0.6487 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | -0.40 | 41.0 | 0.1754 | 0.7014 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | -0.19 | 55.0 | 0.5282 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | -0.21 | 54.0 | 0.4954 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | -0.26 | 50.0 | 0.3755 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |

### Horizon $T = 35$ (Iman-Davenport $p = 4.0816e-02$)
| Candidate Optimizer | $r_{\text{rb}}$ (Effect Size) | $W$ Stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | +0.53 | 32.0 | 0.0654 | 0.4578 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | +0.41 | 40.0 | 0.1591 | 0.9543 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | +0.41 | 40.0 | 0.1591 | 0.9543 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | +0.35 | 44.0 | 0.2312 | 0.9543 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | +0.07 | 63.0 | 0.8209 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | +0.10 | 61.0 | 0.7436 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | +0.18 | 56.0 | 0.5619 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |

### Horizon $T = 50$ (Iman-Davenport $p = 2.2558e-01$)
| Candidate Optimizer | $r_{\text{rb}}$ (Effect Size) | $W$ Stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Significant (α=0.05) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `SMAC20_CustomUncertainty_ei_standard_proximity` | +0.56 | 30.0 | 0.0507 | 0.3546 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_standard_disagreement` | +0.41 | 40.0 | 0.1591 | 0.9543 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal` | +0.31 | 47.0 | 0.2979 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc` | +0.32 | 46.0 | 0.2744 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_proximity_bc` | +0.10 | 61.0 | 0.7436 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy` | +0.32 | 46.0 | 0.2744 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
| `SMAC20_CustomUncertainty_ei_shaker_entropy` | +0.12 | 60.0 | 0.7057 | 1.0000 | ✗ NO | TIE (Equivalent) ⚪ |
