#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=noisy_ei_sweep
#SBATCH --output=results/sweep_noisy_ei_head_to_head/logs/%x_%A_%a.log
#SBATCH --error=results/sweep_noisy_ei_head_to_head/logs/%x_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Defensive directory creation
mkdir -p results/sweep_noisy_ei_head_to_head/logs
mkdir -p results/sweep_noisy_ei_head_to_head/runs

# Environment Activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dyrf 2>/dev/null || true
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate dyrf 2>/dev/null || true
else
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate dyrf 2>/dev/null || true
fi

# Python Executable Resolution
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -f "$HOME/.conda/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="$HOME/.conda/envs/dyrf/bin/python"
elif [ -f "$HOME/miniconda3/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="$HOME/miniconda3/envs/dyrf/bin/python"
elif [ -f "$HOME/anaconda3/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="$HOME/anaconda3/envs/dyrf/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "ERROR: Could not find a valid python binary!" >&2
    exit 1
fi

export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TASK_FILE="results/sweep_noisy_ei_head_to_head/tasks.txt"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "ERROR: Task file $TASK_FILE does not exist or is empty!" >&2
    exit 1
fi

TASK_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")

if [ -z "$TASK_ARGS" ]; then
    echo "ERROR: Empty task arguments for task index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

echo "=========================================================="
echo "Starting Task Index: ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Command Line: scripts/run_carps_patched.py ${TASK_ARGS}"
echo "=========================================================="

$PYTHON_BIN scripts/run_carps_patched.py $TASK_ARGS
