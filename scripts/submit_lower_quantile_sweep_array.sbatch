#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=lower_quant_sweep
#SBATCH --output=results/lower_quantile_sweep/logs/array_%A_%a.log
#SBATCH --error=results/lower_quantile_sweep/logs/array_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

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

# Constrain single-thread execution per worker to prevent CPU oversubscription
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TASK_FILE="${TASK_FILE:-results/lower_quantile_sweep/tasks.txt}"
if [ ! -f "$TASK_FILE" ]; then
    echo "Task file ${TASK_FILE} missing. Generating tasks..."
    "$PYTHON_BIN" scripts/generate_lower_quantile_sweep_tasks.py --output-file "$TASK_FILE"
fi

mkdir -p results/lower_quantile_sweep/logs
mkdir -p results/lower_quantile_sweep/raw
mkdir -p results/lower_quantile_sweep/summaries

TASK_CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")

# Explicitly substitute leading python/python3 with the resolved $PYTHON_BIN
if [[ "$TASK_CMD" == python\ * ]]; then
    EXEC_CMD="\"$PYTHON_BIN\" ${TASK_CMD#python }"
elif [[ "$TASK_CMD" == python3\ * ]]; then
    EXEC_CMD="\"$PYTHON_BIN\" ${TASK_CMD#python3 }"
else
    EXEC_CMD="$TASK_CMD"
fi

echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID | Task Index: $SLURM_ARRAY_TASK_ID"
echo "Resolved Python: $PYTHON_BIN"
echo "Running Command: $EXEC_CMD"
echo "=================================================="

eval "$EXEC_CMD"

EXIT_CODE=$?
echo "=================================================="
echo "Task Index $SLURM_ARRAY_TASK_ID Finished with Exit Code: $EXIT_CODE"
echo "=================================================="
exit $EXIT_CODE
