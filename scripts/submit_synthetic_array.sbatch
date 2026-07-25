#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=uq_sweep_task
#SBATCH --output=results/%x_logs/array_%A_%a.log
#SBATCH --error=results/%x_logs/array_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SWEEP_DIR=$1
if [ -z "$SWEEP_DIR" ]; then
    echo "Usage: sbatch submit_synthetic_array.sbatch <sweep_dir>"
    exit 1
fi

TASK_FILE="${SWEEP_DIR}/tasks.txt"
if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: ${TASK_FILE} missing." >&2
    exit 1
fi

TASK_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")

echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID | Task Index: $SLURM_ARRAY_TASK_ID"
echo "Running arguments: $TASK_ARGS"
echo "=================================================="

python3 Uncertainty_Quantification.py $TASK_ARGS

echo "=================================================="
echo "Task Index $SLURM_ARRAY_TASK_ID Finished"
echo "=================================================="
