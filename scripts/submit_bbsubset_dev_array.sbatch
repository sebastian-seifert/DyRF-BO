#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=bbsubset_dev
#SBATCH --output=results/bbsubset_dev/logs/dev_%A_%a.log
#SBATCH --error=results/bbsubset_dev/logs/dev_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Initialize conda
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ ! -f "results/bbsubset_dev_tasks.txt" ] || [ ! -s "results/bbsubset_dev_tasks.txt" ]; then
    echo "ERROR: results/bbsubset_dev_tasks.txt missing or empty. Attempting auto-generation..." >&2
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_bbsubset_dev_tasks.py || exit 1
    else
        python3 scripts/generate_bbsubset_dev_tasks.py || exit 1
    fi
fi

# Retrieve the specific command arguments corresponding to this array task index
TASK_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" results/bbsubset_dev_tasks.txt)

if [ -z "$TASK_ARGS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID returned empty task args from results/bbsubset_dev_tasks.txt!" >&2
    exit 1
fi

echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID | Task Index: $SLURM_ARRAY_TASK_ID"
echo "Running arguments: $TASK_ARGS"
echo "=================================================="

# Run the execution wrapper
python3 scripts/run_carps_patched.py $TASK_ARGS

echo "=================================================="
echo "Array Task Index $SLURM_ARRAY_TASK_ID Finished"
echo "=================================================="
