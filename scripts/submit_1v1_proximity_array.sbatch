#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=1v1_prox
#SBATCH --output=results/sweep_1v1_proximity/logs/prox_%A_%a.log
#SBATCH --error=results/sweep_1v1_proximity/logs/prox_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Initialize conda / venv
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p results/sweep_1v1_proximity/logs

if [ ! -f "results/sweep_1v1_proximity/tasks.txt" ] || [ ! -s "results/sweep_1v1_proximity/tasks.txt" ]; then
    echo "Auto-generating results/sweep_1v1_proximity/tasks.txt..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_1v1_sweep_tasks.py results 30 || exit 1
    else
        python3 scripts/generate_1v1_sweep_tasks.py results 30 || exit 1
    fi
fi

TASK_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" results/sweep_1v1_proximity/tasks.txt)

if [ -z "$TASK_ARGS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID returned empty task args!" >&2
    exit 1
fi

echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID | Task Index: $SLURM_ARRAY_TASK_ID"
echo "Running arguments: $TASK_ARGS"
echo "=================================================="

python3 scripts/run_carps_patched.py $TASK_ARGS

echo "=================================================="
echo "Array Task Index $SLURM_ARRAY_TASK_ID Finished"
echo "=================================================="
