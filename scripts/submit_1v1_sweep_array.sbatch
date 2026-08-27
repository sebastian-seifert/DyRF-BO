#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=1v1_sweep
#SBATCH --output=results/slurm_logs/%x_%A_%a.log
#SBATCH --error=results/slurm_logs/%x_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# 1. Initialize Conda / Virtualenv robustly on compute node
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

# 2. Select exact Python binary containing CARP-S, SMAC3, and HPOBench
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

mkdir -p results/slurm_logs

if [ -z "$SWEEP_TASK_FILE" ]; then
    echo "ERROR: SWEEP_TASK_FILE environment variable is not set!" >&2
    exit 1
fi

if [ ! -f "$SWEEP_TASK_FILE" ] || [ ! -s "$SWEEP_TASK_FILE" ]; then
    echo "ERROR: Task file $SWEEP_TASK_FILE does not exist or is empty!" >&2
    exit 1
fi

TASK_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SWEEP_TASK_FILE")

if [ -z "$TASK_ARGS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID returned empty task args!" >&2
    exit 1
fi

echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID | Task Index: $SLURM_ARRAY_TASK_ID"
echo "Python Executable: $PYTHON_BIN"
echo "Sweep Task File: $SWEEP_TASK_FILE"
echo "Running arguments: $TASK_ARGS"
echo "=================================================="

$PYTHON_BIN scripts/run_carps_patched.py $TASK_ARGS

echo "=================================================="
echo "Array Task Index $SLURM_ARRAY_TASK_ID Finished"
echo "=================================================="
