#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=ei_compare
#SBATCH --output=results/ei_head_to_head/logs/ei_%A_%a.log
#SBATCH --error=results/ei_head_to_head/logs/ei_%A_%a.err
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

mkdir -p results/ei_head_to_head/logs

if [ ! -f "results/ei_comparison_sweep_tasks.txt" ] || [ ! -s "results/ei_comparison_sweep_tasks.txt" ]; then
    echo "Auto-generating results/ei_comparison_sweep_tasks.txt..."
    python3 scripts/generate_ei_comparison_sweep_tasks.py || exit 1
fi

TASK_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" results/ei_comparison_sweep_tasks.txt)

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
