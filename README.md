# DyRF-BO
Bachelor Thesis from Sebastian Seifert.
Supervisor: Leona Hennig

In this Bachelor Thesis I strive to find new, novel ways to dynamically adjust the Hyperparameters of the Random Forest Surrogate Model,
to increase the effiency of Bayesian Optimization.
This could lead to a more efficent energy usage and could enable faster Automated Machine Learning Models. 

## Git Branching & Sweep Workflow
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
