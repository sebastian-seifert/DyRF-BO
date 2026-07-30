#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=epistemic_full_acq
#SBATCH --output=results/epistemic_ei_pi_lcb_all_dim/logs/acq_%A_%a.log
#SBATCH --error=results/epistemic_ei_pi_lcb_all_dim/logs/acq_%A_%a.err
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

if [ ! -f "results/epistemic_full_acq_array_tasks.txt" ] || [ ! -s "results/epistemic_full_acq_array_tasks.txt" ]; then
    echo "ERROR: results/epistemic_full_acq_array_tasks.txt missing or empty. Attempting auto-generation..." >&2
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_epistemic_full_acq_array_tasks.py || exit 1
    else
        python3 scripts/generate_epistemic_full_acq_array_tasks.py || exit 1
    fi
fi

# Retrieve the specific command arguments corresponding to this array task index
TASK_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" results/epistemic_full_acq_array_tasks.txt)

if [ -z "$TASK_ARGS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID returned empty task args from results/epistemic_full_acq_array_tasks.txt!" >&2
    exit 1
fi

echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID | Task Index: $SLURM_ARRAY_TASK_ID"
echo "Running arguments: $TASK_ARGS"
echo "=================================================="

# Run the execution wrapper
./scripts/run_hpobench_carps_sweep.sh $TASK_ARGS

echo "=================================================="
echo "Array Task Index $SLURM_ARRAY_TASK_ID Finished"
echo "=================================================="
