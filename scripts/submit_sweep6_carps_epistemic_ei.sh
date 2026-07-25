#!/bin/bash
# Submit Sweep 6 (CARP-S Epistemic EI Re-run) as 1 single SLURM Array Job
set -e

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

SWEEP_DIR="results/carps_epistemic_ei_scaled"
mkdir -p "${SWEEP_DIR}/logs"

echo "Generating Sweep 6 array tasks..."
python3 scripts/generate_sweep6_tasks.py

TOTAL_TASKS=$(wc -l < "${SWEEP_DIR}/tasks.txt")
PARENT_DEP=$1

echo "Submitting Sweep 6 (${TOTAL_TASKS} tasks) as a single SLURM Job Array..."

DEP_FLAG=""
if [ -n "$PARENT_DEP" ]; then
    DEP_FLAG="--dependency=afterany:${PARENT_DEP}"
fi

JOB_ID=$(sbatch --parsable ${DEP_FLAG} --job-name=carps_scaled \
    --output="${SWEEP_DIR}/logs/array_%A_%a.log" \
    --error="${SWEEP_DIR}/logs/array_%A_%a.err" \
    --array=1-${TOTAL_TASKS}%15 \
    scripts/submit_hpobench_array.sbatch)

echo "--------------------------------------------------------"
echo "Sweep 6 successfully scheduled -> SLURM Job ID: ${JOB_ID}"
echo "Admin progress logging command:"
echo "python3 scripts/monitor_progress.py ${SWEEP_DIR}"
echo "LAST_JOB_ID:${JOB_ID}"
echo "--------------------------------------------------------"
