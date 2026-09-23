# Cluster Parallelization and Aggregation Plan for Extrapolation UQ Sweep

**Author**: James (Senior Software Engineer)  
**Branch**: `feat/extrapolation-uq-calibration`  
**Date**: September 23, 2026  
**Status**: APPROVED FOR IMPLEMENTATION  

---

## 1. Executive Summary & Cluster Architecture

The Extrapolation Uncertainty Quantification study tests the hypothesis that **Proximity LCB ($U_{\text{PLCB}}$)** preserves calibrated exploration under severe domain extrapolation in high dimensions ($D \ge 16$) where **Standard LCB ($U_{\text{SLCB}}$)** miscalibrates due to RF tree consensus collapse.

The full experimental matrix contains **1,920 runs**:
- **Dimensions ($D$)**: $2, 3, 5, 8, 16, 32$ (6)
- **Training Budgets ($N$)**: $112, 224, 448, 896$ (4)
- **Benchmark Objectives**: `sphere`, `rosenbrock`, `rastrigin`, `ackley` (4)
- **Sampling Strategies**: `natural` (uniform in $[-1, 1]^D$), `stratified` (Dirichlet + ray-casting) (2)
- **Random Seeds**: $0, 1, 2, \dots, 9$ (10 paired seeds)
- **Test Sample Size per Run**: $M = 10,000$ points ($19.2 \times 10^6$ total evaluated test points).

### Cluster Specifications (LUIS Cluster, Leibniz Universität Hannover)
- **Scheduler**: Slurm (`#SBATCH -p ai` or CPU compute partition).
- **Array Limits**: LUIS enforces a maximum array size of **300 tasks** per job array, with task index $< 1,000,000$.
- **Chunking Strategy**: 1,920 tasks partitioned into **10 chunks** of 200 tasks each (last chunk: 120 tasks), submitted with concurrency limit `%25` (max 25 active tasks per user array).
- **Runtime per Task**: Single-thread runtime is ~5–15 seconds for $N=112–448$, ~25–35 seconds for $N=896, D=32$.
- **Total Compute**: ~8–12 CPU core-hours across the entire grid. Under 25 parallel workers, total wall clock time is **~20–30 minutes**.

---

## 2. System Architecture & Component Design

```
                                  [1,920 Run Matrix Grid]
                                             │
                       scripts/generate_extrapolation_sweep_tasks.py
                                             │
                                             ▼
                       results/extrapolation_sweep_tasks.txt (1,920 lines)
                                             │
                  ┌──────────────────────────┴──────────────────────────┐
                  ▼                                                     ▼
      [LUIS Cluster (Slurm)]                                [Local Workstation]
scripts/submit_extrapolation_sweep_all.sh             scripts/run_extrapolation_sweep_local.py
(10 chunks of 200, array %25)                         (Multi-core ProcessPoolExecutor)
                  │                                                     │
                  ▼                                                     ▼
scripts/submit_extrapolation_sweep_array.sbatch                         │
                  │                                                     │
                  └──────────────────────────┬──────────────────────────┘
                                             │
                                             ▼
                        scripts/run_extrapolation_experiment.py
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
        results/extrapolation_uq/raw/           results/extrapolation_uq/summaries/
         *.parquet (11 columns per point)         *.json (compact per-run scorecard)
                         └───────────────────┬───────────────────┘
                                             │
                                             ▼
                       scripts/aggregate_extrapolation_results.py
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
         extrapolation_uq_scorecard.csv           HYPOTHESIS_EVALUATION_REPORT.md
```

---

## 3. Storage Hierarchy & Schemas

### 3.1 Raw Telemetry (Parquet)
Path: `results/extrapolation_uq/raw/dim={D}/n={N}/func={func}/extrapolation_{func}_d{D}_n{N}_{strategy}_s{seed}.parquet`
Footprint: ~80 KB per file $\times$ 1,920 files $\approx$ 150 MB total.

Columns:
1. `point_id` (`int64`): Unique index $0 \dots M-1$.
2. `stratum` (`int64`): 0 (interior), 1 (near), 2 (medium), 3 (far).
3. `d_norm` (`float64`): Normalized distance $d_{\text{raw}} / \sqrt{D}$.
4. `d_rel` (`float64`): Relative domain distance $d_{\text{raw}} / (2\sqrt{D})$.
5. `d_inf` (`float64`): Chebyshev distance.
6. `is_interpolating` (`bool`): True if $d_{\text{norm}} < 10^{-6}$.
7. `y_true` (`float64`): Ground truth standardized target $\tilde{y}$.
8. `y_hat` (`float64`): RF ensemble mean prediction $\hat{\mu}$.
9. `abs_error` (`float64`): Residual absolute error $|\tilde{y} - \hat{\mu}|$.
10. `u_slcb` (`float64`): Standard SMAC uncertainty $1.96 \cdot \sqrt{\sigma^2_{\text{between}} + \sigma^2_{\text{within}}}$.
11. `u_plcb` (`float64`): Proximity LCB exploration uncertainty $\max(\delta_{\text{floor}}, -q_{\text{lower}})$.

### 3.2 Run Summary Telemetry (JSON)
Path: `results/extrapolation_uq/summaries/summary_{func}_d{D}_n{N}_{strategy}_s{seed}.json`
Footprint: ~2 KB per file $\times$ 1,920 files $\approx$ 3.8 MB total.
Enables rapid scorecard generation and Wilcoxon signed-rank statistical tests without loading multi-gigabyte raw arrays.

---

## 4. Phased Milestones & Strict TDD Plan

### Milestone 5: Experiment CLI & Task Generation Pipeline
* **Tests**: `tests/test_extrapolation_cli_and_tasks.py`
  - Test CLI argument parsing, defaults, and type casting.
  - Test invocation of `run_single_experiment` with mock configs.
  - Test idempotent skip logic (`--skip-if-exists`).
  - Test task generator line count (1,920 lines for full grid; 4 lines for `--pilot`).
  - Test formatting and executable command structure in task output file.
* **Implementations**:
  - `scripts/run_extrapolation_experiment.py`
  - `scripts/generate_extrapolation_sweep_tasks.py`

### Milestone 6: Cluster Slurm Execution Harness & Local Runner
* **Tests**: `tests/test_extrapolation_cluster_scripts.py`
  - Test file existence, execution permissions (`chmod +x`), and shell syntax validation (`bash -n`).
  - Test Slurm array chunk calculation logic (validating chunk size $\le 300$ for any arbitrary task count).
  - Test task command extraction (`sed -n "${TASK_ID}p"`).
  - Test local parallel worker pool execution on a 2-task test mock.
* **Implementations**:
  - `scripts/submit_extrapolation_sweep_array.sbatch`
  - `scripts/submit_extrapolation_sweep_all.sh`
  - `scripts/submit_extrapolation_sweep_slurm.sh`
  - `scripts/run_extrapolation_sweep_local.py`

### Milestone 7: Aggregation, Statistical Testing & Scorecard Pipeline
* **Tests**: `tests/test_extrapolation_aggregation.py`
  - Test JSON summary ingestion into aggregated DataFrame.
  - Test calculation of slice metrics (mean rank correlation, PICP delta, Winkler score).
  - Test paired Wilcoxon signed-rank tests across dimensions ($D \in \{2, 3, 5, 8, 16, 32\}$).
  - Test output serialization to CSV scorecard and Markdown report.
* **Implementations**:
  - `scripts/aggregate_extrapolation_results.py`

---

## 5. Review & Verification Criteria

Between milestones, dedicated review agents will verify:
1. **Security & Reproducibility**: Absolute path handling, decoupled seeds, unbuffered stdout.
2. **Cluster Compliance**: Adherence to LUIS Slurm constraints (single CPU thread pinning to prevent oversubscription, chunking $\le 300$).
3. **Idempotence**: Graceful handling of existing outputs for fault-tolerant resumption.
4. **Statistical Rigor**: Accurate two-sided Wilcoxon signed-rank tests with effect size estimation.
