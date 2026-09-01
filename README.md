# DyRF-BO: Epistemic Uncertainty Quantification & Dynamic Random Forests for Bayesian Optimization

**Author**: Sebastian Seifert  
**Supervisor**: Leona Hennig  
**Institution**: Institute of Computer Science, University of Potsdam  

---

## 📌 Overview

**DyRF-BO** is a research framework investigating principled epistemic and aleatoric uncertainty disentanglement in Random Forest (RF) surrogate models for Bayesian Optimization (BO). Traditional Tree-based BO surrogates conflate stochastic aleatoric noise with epistemic model ignorance. DyRF-BO develops and benchmarks:

1. **Second-Order Epistemic Uncertainty Extractors**:
   - *Ensemble Disagreement*: Empirical variance across tree predictions.
   - *Shaker Generalized Entropy*: Information-theoretic dispersion over tree prediction distributions.
   - *Likelihood Credal Sets*: Second-order credal sets bounding imprecise posterior probabilities.
   - *Localized Proximity-Weighted Kernels*: Leaf-co-occurrence similarity and depth-scaled kernel distances with automated $\lambda$-decay tuning.

2. **Decoupled Additive Epistemic Acquisition Functions**:
   - Decoupled acquisition formulations ($EI_{\text{add}}, LCB_{\text{add}}, PI_{\text{add}}$) combining total surrogate variance with time-decayed epistemic exploration bonuses $\beta_t \cdot U_{\text{epistemic}}(x)$.
   - Translation-invariant acquisition transformations for continuous and hierarchical hyperparameter spaces.

3. **Benchmarking & Statistical Multiplicity Suite**:
   - CARP-S BBOB/HPOBench benchmark integration with SMAC3.
   - Closed-form heteroscedastic testbeds (`hetGP`: Branin 2D, Yuan-Wahba 1D, Goldstein-Price 2D, Scalable Sinusoid).
   - Rigorous non-parametric statistical testing (Wilcoxon Signed-Rank tests with Holm-Bonferroni family-wise error rate control and Rank-Biserial effect sizes).

---

## 🚀 Installation & Setup

### Prerequisites
- Linux OS (Ubuntu 22.04+ or Debian-based cluster environment recommended)
- Python $\ge 3.10$

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/Neterich04/DyRF-BO.git
cd DyRF-BO

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade packaging tools and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Optional GPU Acceleration (Credal UQ)
For accelerated Credal likelihood set calculations on NVIDIA GPUs (e.g. A100 / RTX 4090):
```bash
# Via pip (CUDA 12.x)
pip install cupy-cuda12x

# Or via conda
conda install -c conda-forge cupy
```

---

## 🏗️ Repository Architecture

```
DyRF-BO/
├── carps_integration/                # CARP-S and SMAC3 integration layer
│   ├── acquisitions.py               # Additive epistemic acquisitions & WarmupCosineScheduler
│   ├── custom_uncertainty_model.py   # CustomUncertaintyRandomForest SMAC3 surrogate
│   ├── noisy_objective.py            # CARPSNoisyObjectiveFunction adapter
│   ├── optimizer.py                  # CARPSDynamicRFOptimizer implementation
│   └── configs/                      # Hydra configuration hierarchies
├── ep_extractors/                    # Second-order Epistemic UQ Extractors
│   ├── base.py                       # UQExtractor base class & registry
│   ├── disagreement.py               # Ensemble variance extractor
│   ├── shaker.py                     # Shaker generalized entropy extractor
│   ├── credal.py                     # Likelihood-based credal sets
│   ├── proximity.py                  # Localized proximity kernel extractors
│   └── auto_lambda.py                # Fast analytical auto-lambda heuristic
├── noisy_benchmarks/                 # Heteroscedastic & Synthetic Testbeds
│   ├── base.py                       # NoisyBenchmarkProblem abstract base
│   ├── bbob.py                       # BBOB continuous noisy benchmarks
│   ├── hetgp.py                      # Canonical hetGP problems (Branin, Yuan-Wahba, Goldstein-Price)
│   ├── noise_models.py               # Heteroscedastic & mixture noise distributions
│   ├── registry.py                   # Problem discovery and factory registry
│   └── telemetry.py                  # High-resolution instantaneous/incumbent regret logger
├── scripts/                          # Analysis, Sweeps, and Statistical Testing
│   ├── run_1v1_wilcoxon_analysis.py  # Pairwise Wilcoxon tests with Holm-Bonferroni correction
│   ├── compute_pairwise_wilcoxon_suite.py # Global pairwise test suite generator
│   ├── generate_all_anytime_plots.py # Anytime regret trajectory plotting
│   └── run_carps_patched.py          # Patched entry point for CARP-S sweeps
├── tests/                            # Strict TDD Test Suites (Unit & Integration)
│   ├── test_custom_uncertainty_smac3.py
│   ├── test_acquisitions.py
│   ├── test_carps_noisy_integration.py
│   ├── test_noisy_benchmarks.py
│   └── test_1v1_wilcoxon_analysis.py
└── requirements.txt                  # Strict pinned dependency manifest
```

---

## 📊 Reproduction Guidelines for Figures & Tables

### 1. Running Unit & Integration Tests
To verify all algorithmic modules and statistical correctness:
```bash
pytest tests/ -v
```

### 2. Executing Single Benchmark Runs
Execute an isolated CARP-S optimization task with Additive Epistemic EI:
```bash
python scripts/run_carps_patched.py \
  --config-dir carps_integration/configs \
  +task/Noisy/hetgp=cfg_branin_2d \
  +optimizer=dyrf_additive_epistemic_ei \
  ++optimizer.extractor_name=proximity_bc \
  ++optimizer.beta_max=1.0 \
  ++optimizer.warmup_ratio=0.2 \
  task.optimization_resources.n_trials=50 \
  seed=1 \
  outdir=results/demo_run
```

### 3. Computing Statistical Multiplicity Analyses
To generate publication-ready LaTeX, Markdown, and CSV statistical summary tables comparing candidate approaches against SMAC3 baselines under Holm-Bonferroni family-wise error control:
```bash
python scripts/run_1v1_wilcoxon_analysis.py \
  --input results/bbsubset_results_ei_pi_lcb_9_approaches \
  --output results/statistical_reports \
  --alpha 0.05
```

### 4. Generating Anytime Regret Figures & Critical Difference (CD) Diagrams
```bash
# Generate anytime regret trajectory plots
python scripts/generate_all_anytime_plots.py \
  --input results/carps_summary/ \
  --output figures/anytime_plots/

# Combine generated anytime plots into thesis-ready PDF
python scripts/combine_anytime_plots_pdf.py
```

---

## 🌿 Git Branching & Sweep Workflow

To keep the codebase stable and prevent Git history bloat from massive text/JSON sweep report files, we follow a feature-branch and local-archiving workflow.

### 1. Branching Strategy
* **`main`**: The stable branch. Code here must always pass all unit tests. Benchmark sweeps are only pulled from `main` to the cluster for final production runs.
* **`feat/<feature-name>`**: Development branches. All experimental code, parameter sweep setups, and temporary tests are executed here.

### 2. Launching a Sweep (on the Cluster)
1. Create a feature branch and push it to the remote repository:
   ```bash
   git checkout -b feat/my-new-feature
   git push -u origin feat/my-new-feature
   ```
2. Pull the branch on the cluster and launch the sweep with a descriptive folder name:
   ```bash
   git checkout feat/my-new-feature
   git pull
   ./run_unified_cluster_sweep.sh --name my_new_feature_sweep
   ```
3. Stage and commit only the isolated sweep results folder on the cluster:
   ```bash
   git add results/my_new_feature_sweep
   git commit -m "Upload results for my_new_feature_sweep"
   git push
   ```

### 3. Pulling and Archiving Results (on the Local Machine)
Once the sweep completes and the results are pushed to GitHub, pull them down to your laptop, archive them locally, and clean up the repository history before merging to `main`:
1. Pull the results to your local laptop:
   ```bash
   git checkout feat/my-new-feature
   git pull
   ```
2. Run the archive script to move the files from `results/` to `local_results/` (which is gitignored):
   ```bash
   .venv/bin/python scratch/archive_results.py
   ```
3. Commit the deletion of the results folder and push:
   ```bash
   git commit -m "Archive my_new_feature_sweep results locally"
   git push
   ```
4. Merge the clean, tested feature branch into `main`:
   ```bash
   git checkout main
   git merge feat/my-new-feature
   git push origin main
   ```
5. Clean up the branch:
   ```bash
   git branch -d feat/my-new-feature
   git push origin --delete feat/my-new-feature
   ```
