#!/bin/bash
# ==============================================================================
# Master Slurm Orchestrator for 21,600-Task Extrapolation UQ Sweep
# ------------------------------------------------------------------------------
# Uses scripts/master_slurm_scheduler.py to dynamically manage array chunk submissions
# in batches of 200 tasks (%25 concurrency, max 4500 active tasks), monitoring queue
# until full completion with state persistence in results/extrapolation_uq/scheduler_state.json.
# ==============================================================================
#SBATCH -p ai
#SBATCH --job-name=extrap_master
#SBATCH --output=results/extrapolation_uq/logs/master_orchestrator_%j.out
#SBATCH --error=results/extrapolation_uq/logs/master_orchestrator_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p results/extrapolation_uq/logs

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
    --manifest results/extrapolation_sweep_tasks.txt \
    --sbatch scripts/submit_extrapolation_sweep_array.sbatch \
    --chunk-size 200 \
    --max-tasks 4500 \
    --trigger-jobs 10 \
    --concurrency 25 \
    --state-file results/extrapolation_uq/scheduler_state.json \
    --job-prefix extrap_uq \
    "$@"

