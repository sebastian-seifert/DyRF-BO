# Parameter Sweep Summary Report
**Directory**: `results/sweep_20260711_140514`  
**Processed Files**: 27 runs  

This report presents the consolidated mean and standard deviation of all evaluated metrics across all dataset configurations and seeds present in this folder.

| Metric | Chen | Proximity_Baseline | Proximity_Method_A | Proximity_Method_B | Proximity_Method_B_C | Proximity_Method_C | Shaker_GMM_Entropy | Shaker_Likelihood_GL_Bisect | Shaker_Likelihood_GL_Newton | Shaker_Likelihood_Trapz_Bisect | Shaker_Likelihood_Trapz_Newton | Standard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **AUROC** | 0.6597 &plusmn; 0.2121 | 0.5849 &plusmn; 0.1398 | 0.5861 &plusmn; 0.1391 | 0.6109 &plusmn; 0.1540 | 0.6165 &plusmn; 0.1345 | 0.6032 &plusmn; 0.1284 | 0.6800 &plusmn; 0.1970 | 0.7076 &plusmn; 0.1513 | 0.7037 &plusmn; 0.1608 | 0.7070 &plusmn; 0.1539 | 0.7058 &plusmn; 0.1593 | 0.6699 &plusmn; 0.2175 |
| **FPR95** | 0.7653 &plusmn; 0.1908 | 0.8411 &plusmn; 0.1693 | 0.8404 &plusmn; 0.1713 | 0.7886 &plusmn; 0.1963 | 0.8016 &plusmn; 0.1934 | 0.8297 &plusmn; 0.1785 | 0.7471 &plusmn; 0.2080 | 0.6917 &plusmn; 0.2264 | 0.7068 &plusmn; 0.2190 | 0.6923 &plusmn; 0.2271 | 0.7044 &plusmn; 0.2204 | 0.7478 &plusmn; 0.1993 |
| **AUPR** | 0.5244 &plusmn; 0.1833 | 0.4073 &plusmn; 0.1285 | 0.4086 &plusmn; 0.1297 | 0.4238 &plusmn; 0.1450 | 0.4255 &plusmn; 0.1388 | 0.4173 &plusmn; 0.1323 | 0.5284 &plusmn; 0.1985 | 0.5448 &plusmn; 0.1636 | 0.5500 &plusmn; 0.1665 | 0.5449 &plusmn; 0.1641 | 0.5525 &plusmn; 0.1673 | 0.5375 &plusmn; 0.1865 |
| **SPEARMAN** | 0.0667 &plusmn; 0.2705 | 0.0556 &plusmn; 0.2568 | 0.0556 &plusmn; 0.2571 | 0.0464 &plusmn; 0.2688 | 0.0535 &plusmn; 0.2653 | 0.0613 &plusmn; 0.2569 | 0.0694 &plusmn; 0.2396 | 0.0711 &plusmn; 0.2696 | 0.0618 &plusmn; 0.2700 | 0.0711 &plusmn; 0.2697 | 0.0658 &plusmn; 0.2693 | 0.0681 &plusmn; 0.2709 |
| **BRIER** | 0.2370 &plusmn; 0.0527 | 0.2660 &plusmn; 0.0398 | 0.2660 &plusmn; 0.0390 | 0.2574 &plusmn; 0.0499 | 0.2620 &plusmn; 0.0441 | 0.2656 &plusmn; 0.0392 | 0.2328 &plusmn; 0.0565 | 0.2331 &plusmn; 0.0540 | 0.2325 &plusmn; 0.0527 | 0.2328 &plusmn; 0.0540 | 0.2322 &plusmn; 0.0533 | 0.2330 &plusmn; 0.0542 |
| **MI** | 0.2452 &plusmn; 0.2230 | 0.1896 &plusmn; 0.2340 | 0.1922 &plusmn; 0.2372 | 0.2271 &plusmn; 0.2379 | 0.1994 &plusmn; 0.2415 | 0.1804 &plusmn; 0.2379 | 0.2258 &plusmn; 0.1808 | 0.2739 &plusmn; 0.2329 | 0.2700 &plusmn; 0.2311 | 0.2744 &plusmn; 0.2323 | 0.2711 &plusmn; 0.2307 | 0.2614 &plusmn; 0.2245 |
| **JSD** | 0.2514 &plusmn; 0.2350 | 0.1987 &plusmn; 0.2474 | 0.2011 &plusmn; 0.2501 | 0.2394 &plusmn; 0.2501 | 0.2100 &plusmn; 0.2546 | 0.1890 &plusmn; 0.2509 | 0.2311 &plusmn; 0.1861 | 0.2848 &plusmn; 0.2449 | 0.2790 &plusmn; 0.2433 | 0.2853 &plusmn; 0.2444 | 0.2802 &plusmn; 0.2427 | 0.2680 &plusmn; 0.2363 |
| **NAURC** | 0.7571 &plusmn; 0.4320 | 0.8552 &plusmn; 0.2850 | 0.8555 &plusmn; 0.2847 | 0.8203 &plusmn; 0.3323 | 0.7920 &plusmn; 0.2969 | 0.8166 &plusmn; 0.2654 | 0.7447 &plusmn; 0.4028 | 0.6750 &plusmn; 0.3016 | 0.6870 &plusmn; 0.3351 | 0.6785 &plusmn; 0.3101 | 0.6833 &plusmn; 0.3289 | 0.7551 &plusmn; 0.4422 |
