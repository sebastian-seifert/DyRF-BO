#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=dyrf_lambda_array
#SBATCH --output=results/array_dynamic_lambda_%A_%a.log
#SBATCH --error=results/array_dynamic_lambda_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=1-130%15

# Polite Slurm Job Array runner for Proximity Auto Lambda.
# %15 restricts concurrency to a maximum of 15 running jobs at any given time.

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ ! -f "results/array_tasks_dynamic_lambda.txt" ]; then
    echo "ERROR: results/array_tasks_dynamic_lambda.txt missing. Attempting auto-generation..." >&2
    python3 scripts/generate_dynamic_lambda_array_tasks.py || exit 1
fi

TASK_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" results/array_tasks_dynamic_lambda.txt)

echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID | Task Index: $SLURM_ARRAY_TASK_ID"
echo "Running arguments: $TASK_ARGS"
echo "=================================================="

./scripts/run_hpobench_carps_sweep.sh $TASK_ARGS

echo "=================================================="
echo "Array Task Index $SLURM_ARRAY_TASK_ID Finished"
echo "=================================================="
