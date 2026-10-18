#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=hpml_sweep
#SBATCH --output=results/sweep_hpobench_ml/logs/hpml_%A_%a.log
#SBATCH --error=results/sweep_hpobench_ml/logs/hpml_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Environment / Module / Conda Setup
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

# 1. Initialize Conda / Virtualenv robustly on compute node
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate dyrf 2>/dev/null || true
elif [ -f "/bigwork/nhwpseis/miniforge3/etc/profile.d/conda.sh" ]; then
    source "/bigwork/nhwpseis/miniforge3/etc/profile.d/conda.sh"
    conda activate dyrf 2>/dev/null || true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dyrf 2>/dev/null || true
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate dyrf 2>/dev/null || true
else
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate dyrf 2>/dev/null || true
fi

# 2. Explicitly resolve Python interpreter containing the dyrf environment
if [ -f "/bigwork/nhwpseis/.conda/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="/bigwork/nhwpseis/.conda/envs/dyrf/bin/python"
elif [ -f "$HOME/.conda/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="$HOME/.conda/envs/dyrf/bin/python"
elif [ -f "$HOME/miniforge3/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="$HOME/miniforge3/envs/dyrf/bin/python"
elif [ -f "/bigwork/nhwpseis/miniforge3/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="/bigwork/nhwpseis/miniforge3/envs/dyrf/bin/python"
elif [ -f "$HOME/miniconda3/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="$HOME/miniconda3/envs/dyrf/bin/python"
elif [ -f "$HOME/anaconda3/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="$HOME/anaconda3/envs/dyrf/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="$(which python3 || which python)"
fi

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Constrain thread execution per worker to prevent CPU oversubscription
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p results/sweep_hpobench_ml/logs

# Allow overriding TASK_FILE via environment variable, defaulting to results/sweep_hpobench_ml/tasks.txt
TASK_FILE="${TASK_FILE:-results/sweep_hpobench_ml/tasks.txt}"

if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: Task file $TASK_FILE does not exist!" >&2
    exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
CMD_ARGS=$(sed -n "${TASK_ID}p" "$TASK_FILE")

if [ -z "$CMD_ARGS" ]; then
    echo "ERROR: No command found at line $TASK_ID in $TASK_FILE" >&2
    exit 1
fi

echo "=================================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Task ID: $TASK_ID"
echo "Command Args: $CMD_ARGS"
echo "Python: $PYTHON_BIN"
echo "=================================================="

# Execute benchmark run
exec "$PYTHON_BIN" scripts/run_carps_patched.py $CMD_ARGS
