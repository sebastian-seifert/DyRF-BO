#!/bin/bash
# Submit Sweep 6 (CARP-S Epistemic EI Re-run) in dependent 250-task chunks (MaxArraySize=300 limit)
set -e

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

SWEEP_DIR="results/carps_epistemic_ei_scaled"
mkdir -p "${SWEEP_DIR}/logs"

echo "Generating Sweep 6 array tasks..."
python3 scripts/generate_sweep6_tasks.py

TOTAL_TASKS=$(wc -l < "${SWEEP_DIR}/tasks.txt")
CHUNK_SIZE=250

PARENT_DEP=$1

echo "Submitting ${TOTAL_TASKS} tasks for Sweep 6 in chunks of ${CHUNK_SIZE} (MaxArraySize compliant)..."

PREV_JOB=""
for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi

    DEP_FLAG=""
    if [ -n "$PREV_JOB" ]; then
        DEP_FLAG="--dependency=afterany:${PREV_JOB}"
    elif [ -n "$PARENT_DEP" ]; then
        DEP_FLAG="--dependency=afterany:${PARENT_DEP}"
    fi

    if [ -n "$DEP_FLAG" ]; then
        JOB_ID=$(sbatch --parsable ${DEP_FLAG} --job-name=carps_scaled --output="${SWEEP_DIR}/logs/array_%A_%a.log" --error="${SWEEP_DIR}/logs/error_%A_%a.err" --array=${start}-${end}%15 scripts/submit_hpobench_array.sbatch)
        echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID} (${DEP_FLAG})"
    else
        JOB_ID=$(sbatch --parsable --job-name=carps_scaled --output="${SWEEP_DIR}/logs/array_%A_%a.log" --error="${SWEEP_DIR}/logs/error_%A_%a.err" --array=${start}-${end}%15 scripts/submit_hpobench_array.sbatch)
        echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
    fi
    PREV_JOB=$JOB_ID
done

echo "--------------------------------------------------------"
echo "Sweep 6 successfully scheduled."
echo "Admin progress logging command:"
echo "python3 scripts/monitor_progress.py ${SWEEP_DIR}"
echo "LAST_JOB_ID:${PREV_JOB}"
echo "--------------------------------------------------------"
