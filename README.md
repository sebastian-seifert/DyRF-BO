# DyRF-BO
Bachelor Thesis from Sebastian Seifert.
Supervisor: Leona Hennig

In this Bachelor Thesis I strive to find new, novel ways to dynamically adjust the Hyperparameters of the Random Forest Surrogate Model,
to increase the effiency of Bayesian Optimization.
This could lead to a more efficent energy usage and could enable faster Automated Machine Learning Models. 

## Sweep & Archiving Workflow
To prevent Git history bloat and avoid cluttering the GitHub repository with massive text and JSON sweep report files, we use an isolated-sweep and local-archiving workflow.

### 1. Launching a Sweep (on the Cluster)
When launching a sweep, pass a descriptive name via the `--name` flag to create an isolated folder containing all JSON results, report files, and individual job logs:
```bash
# Run a sweep with a descriptive folder name
./run_unified_cluster_sweep.sh --name proximity_sensitivity_alpha

# Stage and commit only the isolated sweep results folder
git add results/proximity_sensitivity_alpha
git commit -m "Upload results for proximity sensitivity alpha sweep"
git push
```

### 2. Pulling and Archiving Results (on the Local Machine)
Once the sweep completes and the results are pushed to GitHub, pull them down and run the archiver to move them into the gitignored local folder and prune them from git:
```bash
# 1. Pull the new results down to your local machine
git pull

# 2. Run the archive script to move files from results/ to local_results/ (gitignored)
.venv/bin/python scratch/archive_results.py

# 3. Stage the deletion and push to clean up GitHub
git commit -m "Archive proximity_sensitivity_alpha results locally"
git push
```
All your results will remain fully accessible in the local `local_results/` folder, but they will never be stored or committed to the remote repository.
