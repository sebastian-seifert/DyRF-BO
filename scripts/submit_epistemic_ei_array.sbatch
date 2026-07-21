#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=epistemic_ei_array
#SBATCH --output=results/epistemic_ei_array_%A_%a.log
#SBATCH --error=results/epistemic_ei_array_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=1-2210%15

# Initialize conda
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ ! -f "results/epistemic_ei_array_tasks.txt" ]; then
    echo "ERROR: results/epistemic_ei_array_tasks.txt missing. Attempting auto-generation..." >&2
    python3 scripts/generate_epistemic_ei_array_tasks.py || exit 1
fi

# Retrieve the specific command arguments corresponding to this array task index
TASK_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" results/epistemic_ei_array_tasks.txt)

echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID | Task Index: $SLURM_ARRAY_TASK_ID"
echo "Running arguments: $TASK_ARGS"
echo "=================================================="

# Run the execution wrapper
./scripts/run_hpobench_carps_sweep.sh $TASK_ARGS

echo "=================================================="
echo "Array Task Index $SLURM_ARRAY_TASK_ID Finished"
echo "=================================================="
