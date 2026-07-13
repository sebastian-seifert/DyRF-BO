# Gemini Sessions Log

## Session: 2026-07-10
* **Goal**: Process topological cluster sweep results, generate comparative tables, optimize configuration spaces, and integrate profiling timers.

### Accomplishments
1. **Sweep Processing**:
   * Executed the parser script `parse_logs_to_best.py` to extract best-performing UQ configs for each dimension (1D to 10D) across all 8 metrics.
   * Filtered out default/unused shell parameters ($K$ and $\alpha$) from baseline configurations in both the report artifact (`best_tuned_approaches.md`) and progress journal (`journal.md`).
2. **Archiving**:
   * Moved raw cluster results from `results/` to a gitignored `local_results/` directory using the archiver script, and committed the deletions to prevent git bloat.
3. **Hyperparameter Truncation**:
   * Identified neighborhood parameter $K=20$ as optimal for localized OOD detection.
   * Truncated the search space to `K_VALUES=(10 20 30)` in `run_unified_cluster_sweep.sh` to speed up future runs.
4. **Execution Profiling**:
   * Integrated sub-section execution timers in:
     - `Credal_Regression_UQ.py` (measuring leaf retrieval, host-to-device transfers, grid setups, Newton/Bisection solver iterations, and integration steps).
     - `Epistemic_Quantifier.py` (measuring tree variance lookups, GMM MC sampling prep, and vectorized loop execution).
   * Appended the `--debug_timing` flag to all runner scripts in `run_unified_cluster_sweep.sh` to automatically log timing profiles during sweeps.
5. **Validation**:
   * Executed the UQ benchmark tests via `bash run_tests.sh` confirming 100% CPU-GPU parity and correct execution.
6. **Manifold OOD Generation**:
   * Implemented codimension-1 manifold OOD generation (where the manifold has dimension $d = D - 1$) in `data_generator.py` and `Uncertainty_Quantification.py`.
   * Generated ID training/test sets lying on the manifold curve/surface (graph of a sum of sinusoids) and OOD test points by translating manifold points along unit normal vectors by orthogonal distance $\lambda$.
   * Created new unit test suite `tests/test_manifold_ood.py` and updated `run_tests.sh` to include discover-based test runs, confirming successful parity and geometric validity.
7. **Aleatoric Uncertainty Quality Evaluation**:
   * Developed a dedicated evaluation driver in `evaluate_aleatoric.py` to assess standard vs. Shaker/Credal aleatoric uncertainty estimation accuracy under heteroscedastic noise: $\sigma_{\text{true}}(x) = 0.05 + 0.25 \cdot \sin^2(x_1)$.
   * Evaluates metrics including Pearson/Spearman correlations with true variance and squared residuals, MSE/MAE, and Gaussian Negative Log-Likelihood (NLL).
   * Automatically queries first representative functions across all dimensions from $1\text{D}$ to $10\text{D}$ using imports from `synthetic_functions.py`.
   * Added `run_aleatoric_evaluation.sh` for cluster dispatches, adding support for a `--quick` flag to run fast TDD unit test runs in `tests/test_aleatoric_evaluation.py`.

## Session: 2026-07-11
* **Goal**: Expand synthetic dimension functions to 11D-15D, add pure aleatoric uncertainty approaches for manifold OOD evaluation, and set up a 7-seed cluster run.

### Accomplishments
1. **Dimension Expansion (11D-15D)**:
   * Created new functions `get_11d_functions` through `get_15d_functions` in [synthetic_functions.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/synthetic_functions.py). Each defines one diverse dimension function (extended sine-cosine product waves).
   * Updated [data_generator.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/data_generator.py) to dynamically scale dataset size linearly for higher dimensions (`n_samples = ndim * 1000`).
   * Integrated 11D-15D functions into [Uncertainty_Quantification.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/Uncertainty_Quantification.py) and [evaluate_aleatoric.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/evaluate_aleatoric.py) (including parsing, printing, and result mappings).
2. **Aleatoric UQ Evaluation in OOD Sweep**:
   * Added support for `Standard_Aleatoric` and `Shaker_Aleatoric` as standalone approaches in [Uncertainty_Quantification.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/Uncertainty_Quantification.py). This enables direct OOD and manifold evaluation of pure data noise estimators.
3. **TDD Validation**:
   * Created a unit test suite [tests/test_new_synthetic_functions.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/tests/test_new_synthetic_functions.py) to verify the shapes, definitions, and integration of the 11D-15D functions and new aleatoric approaches before code changes.
   * Ran the verification test suite via `bash run_tests.sh` confirming all 25 unit and smoke tests passed successfully.
4. **Cluster Sweeping Configuration**:
   * Configured the epistemic sweep script [run_unified_cluster_sweep.sh](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/run_unified_cluster_sweep.sh) to run with 7 runs/seeds (`--n_runs 7`) instead of 5.
   * Configured the dedicated aleatoric sweep script [run_aleatoric_evaluation.sh](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/run_aleatoric_evaluation.sh) to run with 7 runs/seeds (`--n_runs 7`) instead of 5.

## Session: 2026-07-12
* **Goal**: Retrieve cluster sweep results via `git pull`, parse and analyze performance across epistemic UQ on OOD manifolds (1D-10D) and aleatoric UQ under heteroscedastic noise (1D-15D).

### Accomplishments
1. **Cluster Data Sync**:
   * Pulled the completed results of the massive 7-seed sweep from GitHub, resolving merge conflicts from untracked markdown reports.
2. **Epistemic Sweep Analysis**:
   * Parsed the cluster logs for 10 distinct UQ approaches across 8 metrics.
   * Documented results in the thesis progress journal, indicating that Proximity UQ dominates in 1D, standard RF disagreement remains the most robust choice in 2D-5D, and Shaker Likelihood (continuous relative likelihood-ratio) dominates in higher dimensions ($D \ge 6$ and $D \ge 8$).
3. **Aleatoric Sweep Analysis**:
   * Evaluated standard leaf-level variance against Shaker's continuous credal aleatoric uncertainty over 1D to 15D.
   * Discovered a dimensional crossover point: standard variance is slightly superior or comparable in 1D-2D, but Shaker heavily dominates standard variance in all higher dimensions ($3D$ to $15D$), showing significantly higher resilience against partition boundary noise and leaf sample scarcity.
4. **Student-t Likelihood Model Integration**:
   * Integrated **Student-t distribution** and **Corrected Student-t distribution** (with scale inflation $s \cdot \sqrt{1 + 1/n}$) as alternative models in [Credal_Regression_UQ.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/Credal_Regression_UQ.py).
   * Developed a highly optimized and fast normal CDF approximation based on Abramowitz & Stegun 26.7.10 to replace SciPy's slow incomplete beta function (`betainc`) for CPU execution, dropping test run times from minutes to milliseconds.
   * Registered `Shaker_Likelihood_Normal`, `Shaker_Likelihood_StudentT`, and `Shaker_Likelihood_StudentT_Corrected` in [Uncertainty_Quantification.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/Uncertainty_Quantification.py).
   * Created [tests/test_student_t_likelihood.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/tests/test_student_t_likelihood.py) verifying both Student-t and Corrected scale models against edge cases (like count=1 leaves) under 0.1s.
   * Wrote the cluster dispatch script [run_student_t_sweep.sh](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/run_student_t_sweep.sh) ready to sbatch on 10 seeds over a 4x A100 GPU partition.

## Session: 2026-07-13
* **Goal**: Implement depth-normalized proximity UQ approaches (B, C, B+C), develop the Hybrid Proximity-Epistemic neighborhood-blended UQ model, and resolve test runner failures.

### Accomplishments
1. **Depth-Normalized Proximity Decay**:
   * Added the `normalize_by_depth` hyperparameter to `GPUProximityRegressionUQ` to scale the topological decay kernel $p_{\text{walk}} = \exp\left(-\frac{\lambda \cdot d(x_i, x_j)}{2 \cdot \text{max\_tree\_depth}}\right)$ and ensure scale-independence of $\lambda$ across trees of varying depths.
   * Registered `Proximity_Method_B_Norm`, `Proximity_Method_C_Norm`, and `Proximity_Method_B_C_Norm` approaches in `Uncertainty_Quantification.py`.
2. **Hybrid Proximity-Epistemic UQ Model**:
   * Designed and implemented `HybridProximityEpistemicUQ` in [Hybrid_Proximity_Epistemic_UQ.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/Hybrid_Proximity_Epistemic_UQ.py). Blends weighted epistemic signals from the KNN (retrieved via RF Proximity KNN) with the query point's own epistemic uncertainty using a convex combination:
     $$U_{\text{hybrid}}(x) = \lambda_1 \cdot U_{\text{neighbors}}(x) + (1.0 - \lambda_1) \cdot U_{\text{query}}(x)$$
   * Added a `compute_proximity_matrix` method to `GPUProximityRegressionUQ` to compute and expose the raw test-to-train proximity matrix of shape `(n_test, n_train)` in a vectorized way.
   * Registered the 6 requested hybrid configurations (`Hybrid_Shaker_Entropy_L20`, `_L40`, `_L70` and `Hybrid_Likelihood_L20`, `_L40`, `_L70` using Proximity Method B) in [Uncertainty_Quantification.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/Uncertainty_Quantification.py).
3. **Cluster Sweep Setup**:
   * Created [generate_hybrid_sweep_params.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/generate_hybrid_sweep_params.py) and [run_hybrid_sweep.sh](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/run_hybrid_sweep.sh) to execute the 246 parameter sweep configurations on a single SLURM job allocation requesting 6 A100 GPUs, executing background processes concurrently in a round-robin routing layout.
4. **TDD Validation & Parity**:
   * Created a unit test suite [tests/test_hybrid_proximity_epistemic.py](file:///home/sebastians/Projects/university/bachelorthesis/DyRF-BO/tests/test_hybrid_proximity_epistemic.py) to mathematically assert the blending kernel, neighborhood mapping, and boundary conditions.
   * Refactored tests to subclass `unittest.TestCase` so that the local test runner (`bash run_tests.sh`) automatically runs all 36 test cases cleanly.
5. **Data Generator Bugfix**:
   * Resolved a `NameError: name 'X_test' is not defined` in `data_generator.py`'s hypercube boundary OOD generation path by restoring the missing test set construction code block.






