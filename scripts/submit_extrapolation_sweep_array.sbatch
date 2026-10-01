#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=extrap_uq
#SBATCH --output=results/extrapolation_uq/logs/array_%A_%a.log
#SBATCH --error=results/extrapolation_uq/logs/array_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

# Environment / Module / Conda Setup
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

# Explicitly resolve Python interpreter (prioritizing cluster conda dyrf, then active CONDA_PREFIX, then .venv, then python3)
if [ -f "/bigwork/nhwpseis/.conda/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="/bigwork/nhwpseis/.conda/envs/dyrf/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [ -f "$HOME/.conda/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="$HOME/.conda/envs/dyrf/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
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

TASK_FILE="results/extrapolation_sweep_tasks.txt"
if [ ! -f "$TASK_FILE" ]; then
    echo "Task file ${TASK_FILE} missing. Generating tasks..."
    "$PYTHON_BIN" scripts/generate_extrapolation_sweep_tasks.py
fi

mkdir -p results/extrapolation_uq/logs

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
