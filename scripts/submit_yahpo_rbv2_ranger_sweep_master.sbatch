#!/bin/bash
# ==============================================================================
# Master Slurm Orchestrator for 26,180-Task YAHPO Gym rbv2_ranger Sweep
# ------------------------------------------------------------------------------
# LUIS limits total user jobs to 5,000 at any time.
# Uses scripts/master_slurm_scheduler.py to dynamically manage array chunk submissions
# in batches of 250 tasks (%25 concurrency), monitoring queue until completion.
# ==============================================================================
#SBATCH -p ai
#SBATCH --job-name=rngr_master
#SBATCH --output=results/sweep_yahpo_rbv2_ranger/logs/master_orchestrator_%j.out
#SBATCH --error=results/sweep_yahpo_rbv2_ranger/logs/master_orchestrator_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p results/sweep_yahpo_rbv2_ranger/logs

if [ -f "/bigwork/nhwpseis/.conda/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="/bigwork/nhwpseis/.conda/envs/dyrf/bin/python"
elif [ -f "$HOME/.conda/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="$HOME/.conda/envs/dyrf/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="$(which python3 || which python)"
fi

exec "$PYTHON_BIN" scripts/master_slurm_scheduler.py \
    --manifest results/sweep_yahpo_rbv2_ranger/tasks.txt \
    --sbatch scripts/submit_yahpo_rbv2_ranger_sweep_array.sbatch \
    --chunk-size 250 \
    --max-tasks 4500 \
    --trigger-jobs 10 \
    --concurrency 25 \
    --state-file results/sweep_yahpo_rbv2_ranger/scheduler_state.json \
    --job-prefix rngr \
    "$@"
