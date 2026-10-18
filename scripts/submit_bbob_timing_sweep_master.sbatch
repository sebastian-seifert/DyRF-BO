#!/bin/bash
# ==============================================================================
# Master Slurm Orchestrator for 1,100-Task BBOB Dimensionality Timing Sweep
# ------------------------------------------------------------------------------
# Uses scripts/master_slurm_scheduler.py to dynamically manage array chunk submissions
# in batches of 100 tasks (%25 concurrency), monitoring queue until completion.
# ==============================================================================
#SBATCH -p ai
#SBATCH --job-name=bbob_time_master
#SBATCH --output=results/sweep_bbob_timing/logs/master_orchestrator_%j.out
#SBATCH --error=results/sweep_bbob_timing/logs/master_orchestrator_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p results/sweep_bbob_timing/logs

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
    --manifest results/sweep_bbob_timing/tasks.txt \
    --sbatch scripts/submit_bbob_timing_sweep_array.sbatch \
    --chunk-size 100 \
    --max-tasks 1100 \
    --trigger-jobs 5 \
    --concurrency 25 \
    --state-file results/sweep_bbob_timing/scheduler_state.json \
    --job-prefix bbob_time \
    "$@"
